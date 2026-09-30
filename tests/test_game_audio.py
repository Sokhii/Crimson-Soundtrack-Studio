"""Read-only game audio decoding and analysis (gameaudio/)."""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

from conftest import fingerprint
from soundtrack_studio.gameaudio.analysis import GameAudioCache, SourceLocation, _verify
from soundtrack_studio.compiler.archive import CompileError
from soundtrack_studio.testing.fixtures import ANALYZER_FAKE_INSTALL_DB, extract_analyzer_fake_install

VGMSTREAM = os.environ.get("CSS_VGMSTREAM")

FAKE_DECODER = textwrap.dedent('''
    import sys, wave, math, struct
    args = sys.argv[1:]
    if "-V" in args:
        print('{"version": "fake"}'); sys.exit(0)
    out = args[args.index("-o") + 1]
    src = args[-1]
    data = open(src, "rb").read()
    if data[:4] != b"RIFF":
        print("failed opening " + src, file=sys.stderr); sys.exit(1)
    seconds = 12
    with wave.open(out, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000)
        frames = bytearray()
        for i in range(48000 * seconds):
            v = int(12000 * math.sin(2 * math.pi * 220 * i / 48000) * (1 if (i // 12000) % 2 else 0.3))
            frames += struct.pack("<hh", v, v)
        w.writeframes(bytes(frames))
''')


def _fake_decoder(folder: Path) -> Path:
    script = folder / "fake_vgmstream.py"
    script.write_text(FAKE_DECODER, encoding="utf-8")
    if os.name == "nt":
        exe = folder / "vgmstream-cli.bat"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        exe = folder / "vgmstream-cli"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    return exe


def _broken_decoder(folder: Path) -> Path:
    folder = folder / "broken"
    folder.mkdir(exist_ok=True)
    exe = _fake_decoder(folder)
    (folder / "fake_vgmstream.py").write_text("import sys\nprint('failed opening', file=sys.stderr)\nsys.exit(1)\n",
                                               encoding="utf-8")
    return exe


@pytest.fixture
def fake_game(tmp_path):
    return extract_analyzer_fake_install(tmp_path / "Crimson Desert")


@pytest.fixture
def game_studio(studio, fake_game):
    studio.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
    studio.set_game_path(fake_game)
    return studio


def test_unavailable_without_decoder(game_studio, monkeypatch):
    monkeypatch.delenv("CSS_VGMSTREAM", raising=False)
    ok, reason = game_studio.game_audio_available()
    assert not ok and "vgmstream" in reason
    assert game_studio.game_audio_results() == {}
    # describing still works (names only) and records nothing about game audio
    results = game_studio.analyze_semantics(use_ai=False)
    assert results["cue"].total == 4


def test_analysis_is_read_only_cached_and_cleans_temp(game_studio, fake_game, tmp_path, monkeypatch):
    monkeypatch.setenv("CSS_VGMSTREAM", str(_fake_decoder(tmp_path)))
    before = fingerprint(fake_game)
    assert game_studio.game_audio_available() == (True, "")
    summary = game_studio.analyze_game_audio()
    assert summary["sources"] == 4 and summary["ok"] == 4 and summary["errors"] == 0
    assert fingerprint(fake_game) == before                              # the game folder is untouched
    assert not any((game_studio.paths.temp / "gameaudio").iterdir())    # no decoded audio is kept
    results = game_studio.game_audio_results()
    r = results[433831842]
    assert r.status == "ok" and r.cached and r.origin == "archive"
    assert abs(r.decoded_s - 12.0) < 0.01 and r.features["energy_index"] is not None
    assert results[558103].origin == "embedded"                         # read out of the soundbank's DATA section
    # measurements reach the evidence documents of the cues
    docs = dict(game_studio.semantic_items("cue", include_short_cues=True))
    assert "measurements" in docs["2001"] and docs["2001"]["measurements"]["energy_index"] is not None
    # a second run decodes nothing (shared cache): even a broken decoder changes nothing
    monkeypatch.setenv("CSS_VGMSTREAM", str(_broken_decoder(tmp_path)))
    again = game_studio.analyze_game_audio()
    assert again["ok"] == 4


def test_cache_is_shared_between_projects(game_studio, fake_game, tmp_path, monkeypatch):
    monkeypatch.setenv("CSS_VGMSTREAM", str(_fake_decoder(tmp_path)))
    game_studio.analyze_game_audio()
    game_studio.create_project("Second")
    game_studio.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
    game_studio.set_game_path(fake_game)
    assert all(r.cached and r.status == "ok" for r in game_studio.game_audio_results().values())


def test_decode_failures_are_reported_not_fatal(game_studio, tmp_path, monkeypatch):
    monkeypatch.setenv("CSS_VGMSTREAM", str(_broken_decoder(tmp_path)))
    summary = game_studio.analyze_game_audio()
    assert summary["errors"] == 4 and summary["ok"] == 0
    assert any("could not be decoded" in e["message"] for e in game_studio.project.recent_events())
    report = game_studio.game_audio_check()
    assert report["checked"] >= 2 and report["decoded"] == 0 and not report["passed"]
    assert (game_studio.paths.logs / "game_audio_check.json").is_file()
    # describing continues with names only
    assert game_studio.analyze_semantics(use_ai=False)["cue"].total == 4


def test_decode_check_passes_with_working_decoder(game_studio, tmp_path, monkeypatch):
    monkeypatch.setenv("CSS_VGMSTREAM", str(_fake_decoder(tmp_path)))
    report = game_studio.game_audio_check()
    assert report["passed"] and {i["stored"] for i in report["items"]} == {"archive", "embedded"}
    assert all(abs(i["decoded_s"] - 12.0) < 0.01 for i in report["items"])


def test_hash_verification():
    loc = SourceLocation(1, 1, "archive", "x.wem", 10, "0" * 40, "sha1_prefix_4")
    with pytest.raises(CompileError):
        _verify(b"RIFFdata", loc)
    import hashlib
    loc.content_hash = hashlib.sha1(b"RIFF").hexdigest()
    _verify(b"RIFFanything", loc)          # only the prefix is hashed


def test_cache_versions(tmp_path):
    cache = GameAudioCache(tmp_path / "c.sqlite3")
    cache.put_measure("k", "ok", {"a": 1}, decoded_s=2.0)
    cache.put_listening("k", "model", {"x": 1})
    assert cache.get_measure("k")["features"] == {"a": 1} and cache.get_listening("k", "model") == {"x": 1}
    assert cache.get_listening("k", "other") is None
    cache.close()


@pytest.mark.skipif(not VGMSTREAM or not Path(VGMSTREAM or "").is_file(), reason="set CSS_VGMSTREAM to vgmstream-cli")
def test_real_vgmstream_decodes_pcm_and_rejects_placeholders(game_studio, tmp_path):
    """The fake install's 'Vorbis' files are placeholders (no real Vorbis data): vgmstream must refuse them
    cleanly; its PCM file must decode to the exact length."""

    from soundtrack_studio.gameaudio.decoder import decode

    with game_studio._game_audio() as analyzer:
        loc = analyzer.locate(16721128)
        wem = tmp_path / "pcm.wem"
        wem.write_bytes(analyzer.read_wem(loc))
        decode(Path(VGMSTREAM), wem, tmp_path / "pcm.wav")
        import soundfile as sf

        info = sf.info(str(tmp_path / "pcm.wav"))
        assert info.frames == 1440000 and info.samplerate == 48000
    report = game_studio.game_audio_check()
    assert report["decoded"] == 0 and all(i.get("error") for i in report["items"])
