import json
import os
import shutil
import threading
import time

import pytest
from conftest import fingerprint

from soundtrack_studio.errors import LibraryError, OperationCancelled
from soundtrack_studio.library import scanner as scanner_mod
from soundtrack_studio.library.cache import AnalysisCache
from soundtrack_studio.library.scanner import LibraryScanner
from soundtrack_studio.project.store import Project
from soundtrack_studio.testing.fixtures import set_flac_tags, write_test_flac


@pytest.fixture
def project(paths):
    p = Project.create(paths, "Lib")
    yield p
    p.close()


@pytest.fixture
def cache(paths):
    c = AnalysisCache(paths.cache / "audio_analysis.sqlite3")
    yield c
    c.close()


def tracks(project):
    return {r["rel_path"]: dict(r) for r in project.query(
        "SELECT t.*, m.title, m.artist, m.duration_s, m.sample_rate, m.channels, m.bit_depth FROM track t"
        " LEFT JOIN track_metadata m ON m.track_id=t.id")}


def make_library(root):
    write_test_flac(root / "A" / "one.flac", seconds=6, tags={"TITLE": "One", "ARTIST": "X"})
    write_test_flac(root / "A" / "B" / "C" / "two – 二.flac", seconds=4, bpm=90, channels=1)
    write_test_flac(root / "Ünïcødé" / "三 three.flac", seconds=3, sample_rate=48000, subtype="PCM_24")
    (root / "A" / "broken.flac").write_bytes(b"not a flac at all")
    (root / "A" / "._one.flac").write_bytes(b"\x00\x05\x16\x07 resource fork")
    (root / "A" / "cover.jpg").write_bytes(b"\xff\xd8")
    return root


def test_recursive_scan_metadata_and_errors(tmp_path, project, cache):
    lib = make_library(tmp_path / "Music")
    before = fingerprint(lib)
    stats = LibraryScanner(project, cache, workers=2).scan(lib)
    assert stats.files_found == 4 and stats.analyzed == 3 and stats.errors == 1 and stats.skipped_system_files == 1
    t = tracks(project)
    assert set(t) == {"A/one.flac", "A/B/C/two – 二.flac", "Ünïcødé/三 three.flac", "A/broken.flac"}
    assert t["A/one.flac"]["title"] == "One" and t["A/one.flac"]["duration_s"] == pytest.approx(6.0)
    assert t["A/B/C/two – 二.flac"]["channels"] == 1 and t["A/B/C/two – 二.flac"]["title"] is None
    assert t["Ünïcødé/三 three.flac"]["bit_depth"] == 24 and t["Ünïcødé/三 three.flac"]["sample_rate"] == 48000
    assert t["A/broken.flac"]["status"] == "error" and "not a valid FLAC" in t["A/broken.flac"]["error"]
    features = project.query_one("SELECT features_json FROM track_features f JOIN track t ON t.id=f.track_id"
                                 " WHERE t.rel_path='A/one.flac'")[0]
    stored = json.loads(features)
    assert stored["decoded_duration_s"] == pytest.approx(6.0) and stored["rms_dbfs"] < 0
    assert stored["tempo_bpm"] is None and "too short for a tempo estimate" in stored["notes"]  # < 8 s: not guessed
    assert fingerprint(lib) == before  # user files untouched


def test_rescan_is_cached_and_detects_changes(tmp_path, project, cache, monkeypatch):
    lib = make_library(tmp_path / "Music")
    scanner = LibraryScanner(project, cache, workers=1)
    scanner.scan(lib)
    second = scanner.scan(lib)
    assert second.unchanged == 3 and second.analyzed == 0

    # re-tagging changes the file but not the audio: metadata re-read, analysis reused
    set_flac_tags(lib / "A" / "one.flac", {"TITLE": "Renamed"})
    third = scanner.scan(lib)
    assert third.analyzed == 1 and third.features_from_cache == 1
    assert tracks(project)["A/one.flac"]["title"] == "Renamed"

    # moving a file: new path, same audio -> no new signal analysis
    calls = []
    real = scanner_mod.analyze_blocks
    monkeypatch.setattr(scanner_mod, "analyze_blocks", lambda *a, **k: calls.append(1) or real(*a, **k))
    shutil.move(str(lib / "Ünïcødé" / "三 three.flac"), str(lib / "moved.flac"))
    fourth = scanner.scan(lib)
    assert not calls and fourth.missing == 1
    t = tracks(project)
    assert t["Ünïcødé/三 three.flac"]["status"] == "missing" and t["moved.flac"]["status"] == "ok"


def test_duplicates(tmp_path, project, cache):
    lib = tmp_path / "Music"
    write_test_flac(lib / "a.flac", seconds=2)
    shutil.copyfile(lib / "a.flac", lib / "copy.flac")
    (lib / "sub").mkdir()
    shutil.copyfile(lib / "a.flac", lib / "sub" / "copy2.flac")
    stats = LibraryScanner(project, cache).scan(lib)
    assert stats.duplicates == 2
    t = tracks(project)
    assert t["a.flac"]["duplicate_of"] is None
    assert t["copy.flac"]["duplicate_of"] == t["a.flac"]["id"]


def test_very_long_paths(tmp_path, project, cache):
    deep = tmp_path / "Music"
    for i in range(8):
        deep = deep / (f"{i:02d} a rather long folder name for a soundtrack album disc " + "x" * 10)
    write_test_flac(deep / ("a long file name " * 6 + ".flac"), seconds=1)
    assert len(str(deep)) > 260
    stats = LibraryScanner(project, cache).scan(tmp_path / "Music")
    assert stats.analyzed == 1 and stats.errors == 0


def test_missing_folder(tmp_path, project, cache):
    with pytest.raises(LibraryError):
        LibraryScanner(project, cache).scan(tmp_path / "nope")


def test_cancellation(tmp_path, project, cache):
    lib = tmp_path / "Music"
    for i in range(6):
        write_test_flac(lib / f"{i}.flac", seconds=2, tone_hz=200 + i)
    flag = threading.Event()
    seen = []

    def progress(p):
        seen.append(p)
        if p.done >= 1:
            flag.set()

    with pytest.raises(OperationCancelled):
        LibraryScanner(project, cache, workers=1).scan(lib, progress, flag.is_set)
    assert project.query_one("SELECT status FROM library_scan ORDER BY id DESC")[0] == "cancelled"


def test_corrupt_cache_is_rebuilt(paths):
    file = paths.cache / "audio_analysis.sqlite3"
    file.write_bytes(b"garbage" * 100)
    c = AnalysisCache(file)
    assert c.get_features("x", 1) is None
    c.close()


@pytest.mark.slow
def test_large_library(tmp_path, project, cache):
    lib = tmp_path / "Music"
    source = write_test_flac(tmp_path / "src.flac", seconds=0.5, sample_rate=8000, channels=1, bpm=None)
    data = source.read_bytes()
    count = 400
    for i in range(count):
        target = lib / f"Disc {i // 50:02d}" / f"Track {i:03d}.flac"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    started = time.monotonic()
    stats = LibraryScanner(project, cache).scan(lib)
    first = time.monotonic() - started
    assert stats.files_found == count and stats.analyzed == count and stats.duplicates == count - 1
    started = time.monotonic()
    again = LibraryScanner(project, cache).scan(lib)
    assert again.unchanged == count
    assert time.monotonic() - started < max(5.0, first)
    os.remove(source)
