import shutil
import sqlite3
from pathlib import Path

import pytest
from conftest import REAL_ANALYZER_FIXTURE, fingerprint

from soundtrack_studio.analyzer_db import contract
from soundtrack_studio.analyzer_db.importer import import_database, open_snapshot, resolve_source, source_changed
from soundtrack_studio.analyzer_db.reader import AnalyzerReader
from soundtrack_studio.errors import AnalyzerDbError
from soundtrack_studio.testing.fixtures import write_analyzer_db


def copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "real" / "studio.sqlite3"
    target.parent.mkdir(parents=True)
    shutil.copyfile(REAL_ANALYZER_FIXTURE, target)
    return target


def test_real_analyzer_output_imports_and_source_is_untouched(tmp_path, paths):
    db = copy_fixture(tmp_path)
    before = fingerprint(db.parent)
    imported = import_database(db, paths)
    assert imported.report.ok, imported.report.issues
    assert imported.schema_version == 1
    assert imported.report.counts["music_objects"] == 11
    assert imported.report.selected_installation_id == 1
    # the Analyzer stores its DB in WAL mode; even a read-only SQLite open would create -wal/-shm here
    assert fingerprint(db.parent) == before
    assert paths.is_inside(imported.snapshot_path)


def test_committed_wal_content_is_included(tmp_path, paths, game):
    root, files = game
    db = write_analyzer_db(tmp_path / "w" / "studio.sqlite3", game_root=root, game_files=files)
    writer = sqlite3.connect(str(db))
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("INSERT INTO installation(id, root_path, label, first_seen, last_scanned) VALUES (2,'X:/Other','x','t','0000')")
    writer.execute("INSERT INTO scan(installation_id, started_at, finished_at, status, mode, parser_version) VALUES (2,'t','t','completed','analyze',1)")
    writer.commit()
    try:
        assert (db.parent / "studio.sqlite3-wal").stat().st_size > 0
        before = fingerprint(db.parent)
        imported = import_database(db, paths)
        assert fingerprint(db.parent) == before
        assert {i.id for i in imported.report.installations} == {1, 2}
        assert imported.manifest["included_wal"] is True
    finally:
        writer.close()


def test_snapshot_is_opened_read_only(paths, analyzer_db):
    imported = import_database(analyzer_db, paths)
    conn = open_snapshot(imported.snapshot_path)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM installation")
    conn.close()
    assert sorted(p.name for p in imported.snapshot_path.parent.iterdir()) == ["analyzer.sqlite3", "import.json"]


def test_reimport_of_same_content_reuses_snapshot(paths, analyzer_db):
    first = import_database(analyzer_db, paths)
    second = import_database(analyzer_db, paths)
    assert second.reused and second.snapshot_path == first.snapshot_path
    assert len(list(paths.databases.iterdir())) == 1
    assert not list(paths.temp.iterdir())  # temporary import folders are cleaned up


def test_analyzer_folder_can_be_dropped(tmp_path, paths, analyzer_db):
    analyzer_root = analyzer_db.parents[2]
    assert resolve_source(analyzer_root) == analyzer_db
    assert import_database(analyzer_root, paths).report.ok
    assert resolve_source(Path(str(analyzer_db) + "-wal")) == analyzer_db


def test_folder_without_database(tmp_path, paths):
    with pytest.raises(AnalyzerDbError, match="does not contain"):
        import_database(tmp_path, paths)


def test_not_a_sqlite_file(tmp_path, paths):
    bogus = tmp_path / "music.sqlite3"
    bogus.write_bytes(b"ID3" + b"\x00" * 200)
    with pytest.raises(AnalyzerDbError, match="not a SQLite database"):
        import_database(bogus, paths)


def test_missing_file(tmp_path, paths):
    with pytest.raises(AnalyzerDbError, match="does not exist"):
        import_database(tmp_path / "nope.sqlite3", paths)


def test_other_sqlite_database_is_rejected(tmp_path, paths):
    other = tmp_path / "other.db"
    conn = sqlite3.connect(str(other))
    conn.execute("CREATE TABLE songs (id INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(AnalyzerDbError, match="not created by Crimson Desert Analyzer"):
        import_database(other, paths)


@pytest.mark.parametrize("version,word", [(2, "newer"), (0, "older")])
def test_incompatible_schema_version_message(tmp_path, paths, version, word):
    db = write_analyzer_db(tmp_path / "v.sqlite3", schema_version=version)
    with pytest.raises(AnalyzerDbError) as info:
        import_database(db, paths)
    msg = info.value.message
    assert "not compatible with this version of Crimson Soundtrack Studio" in msg
    assert "Expected schema: 1" in msg and f"Found schema: {version}" in msg
    assert info.value.hint  # tells the user what to do
    assert not any(paths.databases.iterdir())


def test_missing_required_table(tmp_path, paths):
    db = write_analyzer_db(tmp_path / "m.sqlite3", wal=False)
    conn = sqlite3.connect(str(db))
    conn.execute("DROP TABLE media_source")
    conn.commit()
    conn.close()
    with pytest.raises(AnalyzerDbError, match="missing information"):
        import_database(db, paths)


def test_missing_required_column(tmp_path, paths):
    db = write_analyzer_db(tmp_path / "c.sqlite3", wal=False)
    conn = sqlite3.connect(str(db))
    conn.execute("ALTER TABLE wem DROP COLUMN duration_s")
    conn.commit()
    conn.close()
    with pytest.raises(AnalyzerDbError) as info:
        import_database(db, paths)
    assert "duration_s" in info.value.details


def test_corrupted_database(tmp_path, paths):
    db = write_analyzer_db(tmp_path / "x.sqlite3", wal=False)
    data = bytearray(db.read_bytes())
    for offset in range(4096, len(data), 7):  # keep the header, shred the pages
        data[offset] = (data[offset] * 31 + 7) & 0xFF
    db.write_bytes(bytes(data))
    with pytest.raises(AnalyzerDbError, match="damaged"):
        import_database(db, paths)


def test_truncated_database(tmp_path, paths):
    db = write_analyzer_db(tmp_path / "t.sqlite3", wal=False)
    db.write_bytes(db.read_bytes()[: 4096 * 3])
    with pytest.raises(AnalyzerDbError):
        import_database(db, paths)


def test_database_without_completed_scan(tmp_path, paths):
    db = write_analyzer_db(tmp_path / "s.sqlite3", scan_status="running")
    with pytest.raises(AnalyzerDbError, match="never finished"):
        import_database(db, paths)


def test_latest_scan_failed_uses_earlier_completed_scan(tmp_path, paths):
    db = write_analyzer_db(tmp_path / "f.sqlite3", wal=False)
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT INTO scan(installation_id, started_at, status, mode, parser_version) VALUES (1,'t','failed','analyze',1)")
    conn.commit()
    conn.close()
    imported = import_database(db, paths)
    assert imported.report.ok
    assert "latest_scan_incomplete" in {i.code for i in imported.report.warnings}


def test_database_without_music_structures_warns(tmp_path, paths):
    db = write_analyzer_db(tmp_path / "n.sqlite3", with_music=False)
    imported = import_database(db, paths)
    assert imported.report.ok
    assert "no_music_structures" in {i.code for i in imported.report.warnings}


def test_empty_database_has_no_installation(tmp_path, paths):
    db = tmp_path / "e.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.executescript(Path(contract.__file__).parents[1].joinpath("testing", "analyzer_schema_v1.sql").read_text())
    conn.execute("PRAGMA user_version=1")
    conn.commit()
    conn.close()
    with pytest.raises(AnalyzerDbError, match="does not contain a scanned"):
        import_database(db, paths)


def test_source_change_detection(paths, analyzer_db):
    imported = import_database(analyzer_db, paths)
    assert source_changed(imported.manifest) is False
    with open(analyzer_db, "ab") as handle:
        handle.write(b"\x00" * 4096)
    assert source_changed(imported.manifest) is True
    analyzer_db.unlink()
    assert source_changed(imported.manifest) is None


def test_reader_names_and_sources(paths, analyzer_db):
    imported = import_database(analyzer_db, paths)
    with AnalyzerReader(imported.snapshot_path) as reader:
        names = reader.best_names([412724365, 60970509, 999])
        assert names == {412724365: "bgm", 60970509: "Play_BGM_World"}
        kinds = {f["kind"] for f in reader.source_files(1)}
        assert kinds == {"pamt", "paz", "papgt", "loose"}
        media = reader.media(1, [433831842])
        assert media[433831842]["role"] == "music" and media[433831842]["codec"] == "Wwise Vorbis"
