import pytest
from conftest import fingerprint

from soundtrack_studio.analyzer_db import compat
from soundtrack_studio.errors import GameInstallError, LibraryError, ProjectError
from soundtrack_studio.services import Studio
from soundtrack_studio.testing.fixtures import write_test_flac


def states(status):
    return {s.number: s.state for s in status.steps}


def test_phase1_workflow(tmp_path, studio, game, analyzer_db):
    root, _files = game
    s = studio.project_status()
    assert states(s)[1] == "missing" and states(s)[2] == "missing" and states(s)[4] == "warning"  # AI optional

    assert studio.set_game_path(root).status == compat.UNVERIFIABLE
    game_before = fingerprint(root)
    db_before = fingerprint(analyzer_db.parent)
    studio.import_analyzer(analyzer_db)
    assert studio.project.last_game_check()["status"] == compat.MATCH  # re-checked after import
    model = studio.game_model()
    assert model.stats["cues"] == 4
    assert studio.game_model() is model  # memoised in-session

    music = tmp_path / "Music"
    write_test_flac(music / "a.flac", seconds=2)
    studio.set_library_path(music)
    stats = studio.scan_library()
    assert stats.analyzed == 1
    s = studio.project_status()
    assert [states(s)[i] for i in (1, 2, 3)] == ["ok"] * 3
    assert states(s)[5] == "warning"  # scanned but not described yet
    studio.analyze_semantics(use_ai=False)
    s = studio.project_status()
    assert states(s)[5] == "ok"
    assert all(states(s)[i] == "unavailable" for i in range(6, 11))
    assert not s.warnings
    assert fingerprint(root) == game_before and fingerprint(analyzer_db.parent) == db_before


def test_reopen_project_keeps_everything(tmp_path, paths, game, analyzer_db):
    s1 = Studio(paths)
    project = s1.create_project("Persist")
    s1.set_game_path(game[0])
    s1.import_analyzer(analyzer_db)
    music = tmp_path / "Music"
    write_test_flac(music / "a.flac", seconds=2)
    s1.set_library_path(music)
    s1.scan_library()
    folder = project.folder
    s1.shutdown()

    s2 = Studio(paths)
    assert s2.open_last_project().folder == folder
    status = s2.project_status()
    assert status.game_path == str(game[0].resolve())
    assert status.analyzer["schema_version"] == 1 and status.library_counts["ok"] == 1
    assert s2.game_model().stats["cues"] == 4
    s2.shutdown()


def test_missing_external_locations_are_reported(tmp_path, studio, game, analyzer_db):
    import shutil

    music = tmp_path / "Music"
    write_test_flac(music / "a.flac", seconds=1)
    studio.set_game_path(game[0])
    studio.import_analyzer(analyzer_db)
    studio.set_library_path(music)
    studio.scan_library()
    shutil.rmtree(music)
    shutil.rmtree(game[0])
    status = studio.project_status()
    assert states(status)[1] == "warning" and states(status)[3] == "warning"
    assert any("no longer available" in w for w in status.warnings)


def test_changed_source_database_is_reported(studio, analyzer_db):
    studio.import_analyzer(analyzer_db)
    with open(analyzer_db, "ab") as handle:
        handle.write(b"\x00" * 4096)
    assert any("changed since it was imported" in w for w in studio.project_status().warnings)


def test_mismatched_game_is_a_warning(studio, game, analyzer_db):
    studio.import_analyzer(analyzer_db)
    with open(game[0] / "0004" / "0.pamt", "ab") as handle:
        handle.write(b"patch")
    report = studio.set_game_path(game[0])
    assert report.status == compat.MISMATCH
    status = studio.project_status()
    assert states(status)[1] == "warning" and any("does not match" in w for w in status.warnings)


def test_input_validation(tmp_path, paths, studio):
    with pytest.raises(GameInstallError):
        studio.set_game_path(tmp_path / "nope")
    with pytest.raises(GameInstallError, match="inside the Crimson Soundtrack Studio folder"):
        studio.set_game_path(paths.output)
    with pytest.raises(LibraryError):
        studio.set_library_path(tmp_path / "nope")
    with pytest.raises(LibraryError, match="Choose a music folder"):
        studio.scan_library()
    empty = Studio(paths)
    with pytest.raises(ProjectError):
        empty.project_status()


def test_imported_snapshot_deleted(paths, studio, analyzer_db):
    import shutil

    studio.import_analyzer(analyzer_db)
    shutil.rmtree(paths.databases)
    studio._model = None
    status = studio.project_status()
    assert states(status)[2] == "warning"
    from soundtrack_studio.errors import AnalyzerDbError
    with pytest.raises(AnalyzerDbError, match="missing"):
        studio.game_model()
