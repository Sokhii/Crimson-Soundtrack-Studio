import json
import os
import shutil
import sqlite3

import pytest
from conftest import REAL_ANALYZER_FIXTURE

from soundtrack_studio.analyzer_db import compat
from soundtrack_studio.analyzer_db.importer import import_database
from soundtrack_studio.analyzer_db.reader import AnalyzerReader
from soundtrack_studio.game_model import builder
from soundtrack_studio.game_model.model import GameMusicModel
from soundtrack_studio.testing.fixtures import write_analyzer_db


def build_model(paths, db):
    imported = import_database(db, paths)
    return builder.load_or_build(paths, imported.snapshot_path, imported.sha256, imported.report.selected_installation_id)


def test_hierarchy_from_real_analyzer_output(tmp_path, paths):
    db = tmp_path / "real.sqlite3"
    shutil.copyfile(REAL_ANALYZER_FIXTURE, db)
    model = build_model(paths, db)
    assert model.stats["switches"] == 1 and model.stats["playlists"] == 2
    assert model.stats["segments"] == 4 and model.stats["tracks"] == 4
    assert model.roots == [4001]
    switch = model.nodes[4001]
    assert switch.arguments[0]["name"] == "BGM_Region"
    assert switch.child_states["3001"] == ["BGM_Region=Desert"]
    assert model.nodes[3001].children == [2001, 2002]
    cue = next(c for c in model.cues if c.segment_id == 2001)
    assert cue.path == [4001, 3001, 2001]
    assert cue.duration_ms == 180000.0 and cue.tempo_bpm == 120.0
    assert cue.event_names == ["Play_BGM_World"]
    assert cue.source_ids == [433831842]
    assert [c.segment_id for c in model.cues if c.is_transition] == [2004]
    assert model.media[558103].containers == ["embedded"]


def test_unnamed_state_is_not_invented(tmp_path, paths):
    db = tmp_path / "real.sqlite3"
    shutil.copyfile(REAL_ANALYZER_FIXTURE, db)
    model = build_model(paths, db)
    assert model.nodes[4001].child_states["3002"] == ["BGM_Region=491961918"]


def test_synthetic_fixture_matches_real_structure(paths, analyzer_db):
    model = build_model(paths, analyzer_db)
    assert model.stats["cues"] == 4 and model.roots == [4001]
    assert model.nodes[1004].source_ids == [558103]
    assert model.media[433831842].streaming == ["prefetch_streaming"]
    assert not model.warnings


def test_model_serialisation_roundtrip(paths, analyzer_db):
    model = build_model(paths, analyzer_db)
    again = GameMusicModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert again.to_dict() == model.to_dict()


def test_model_cache_is_used_for_unchanged_database(paths, analyzer_db, monkeypatch):
    imported = import_database(analyzer_db, paths)
    args = (paths, imported.snapshot_path, imported.sha256, 1)
    first = builder.load_or_build(*args)
    assert builder.cache_path(paths, imported.sha256, 1).is_file()
    monkeypatch.setattr(builder, "build", lambda *a, **k: pytest.fail("rebuilt although the database is unchanged"))
    assert builder.load_or_build(*args).stats == first.stats


def test_corrupt_model_cache_is_rebuilt(paths, analyzer_db):
    imported = import_database(analyzer_db, paths)
    cache = builder.cache_path(paths, imported.sha256, 1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("{broken", encoding="utf-8")
    assert builder.load_or_build(paths, imported.snapshot_path, imported.sha256, 1).stats["cues"] == 4


def _edit(db, sql_statements):
    conn = sqlite3.connect(str(db))
    for sql in sql_statements:
        conn.execute(sql)
    conn.commit()
    conn.close()


def test_shared_segment_cycles_and_missing_media(tmp_path, paths):
    db = write_analyzer_db(tmp_path / "e.sqlite3", wal=False)
    _edit(db, [
        # segment 2001 is also used by playlist 3002
        "INSERT INTO object_ref(bank_asset_id, from_object_id, to_id, kind, confidence) VALUES (1,3002,2001,'child','parsed')",
        # a (corrupt) cycle: the switch claims to be a child of a segment
        "INSERT INTO object_ref(bank_asset_id, from_object_id, to_id, kind, confidence) VALUES (1,2003,4001,'child','parsed')",
        # a track whose media has no WEM record
        "UPDATE wwise_object SET fields_json=json_set(fields_json, '$.sources[0].source_id', 777) WHERE object_id=1003",
        "INSERT INTO object_ref(bank_asset_id, from_object_id, to_id, kind, confidence) VALUES (1,1003,777,'track_source','parsed')",
        "UPDATE wwise_object SET parse_status='partial' WHERE object_id=1002",
    ])
    model = build_model(paths, db)
    cue = next(c for c in model.cues if c.segment_id == 2001)
    assert cue.parent_count == 2
    assert model.media[777].found is False
    assert any("no WEM record" in w for w in model.warnings)
    assert any("partially decoded" in w for w in model.warnings)
    assert len(model.cues) == 4  # cycle did not explode the walk


def test_music_outside_interactive_hierarchy_is_reported(tmp_path, paths):
    db = write_analyzer_db(tmp_path / "o.sqlite3", wal=False)
    _edit(db, ["UPDATE classification SET role='music' WHERE entity_key=16721128"])
    model = build_model(paths, db)
    assert model.stats["music_outside_hierarchy"] == 1
    assert any("not part of the interactive music hierarchy" in w for w in model.warnings)


# ------------------------------------------------------------ install check
def recorded(paths, analyzer_db):
    imported = import_database(analyzer_db, paths)
    with AnalyzerReader(imported.snapshot_path) as reader:
        return reader.source_files(1), reader.installation(1)["root_path"]


def test_install_match(paths, analyzer_db, game):
    files, root = recorded(paths, analyzer_db)
    report = compat.check_installation(game[0], files, root)
    assert report.status == compat.MATCH and report.checked_files == 6 and not report.notes


def test_install_copied_elsewhere_is_probable_match(tmp_path, paths, analyzer_db, game):
    files, root = recorded(paths, analyzer_db)
    moved = tmp_path / "Moved" / "Crimson Desert"
    shutil.copytree(game[0], moved)  # copytree preserves mtimes; simulate a copy tool that does not
    for p in moved.rglob("*"):
        if p.is_file():
            os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 5_000_000_000))
    report = compat.check_installation(moved, files, root)
    assert report.status == compat.PROBABLE_MATCH and report.usable
    assert report.notes  # different path than the one the Analyzer scanned


def test_install_updated_game_is_mismatch(paths, analyzer_db, game):
    files, root = recorded(paths, analyzer_db)
    with open(game[0] / "0004" / "0.pamt", "ab") as handle:
        handle.write(b"update")
    (game[0] / "0000" / "0.paz").unlink()
    report = compat.check_installation(game[0], files, root)
    assert report.status == compat.MISMATCH and not report.usable
    assert report.size_mismatch == ["0004/0.pamt"] and report.missing == ["0000/0.paz"]
    assert "updated" in report.summary()


def test_not_a_game_folder(tmp_path, paths, analyzer_db):
    files, root = recorded(paths, analyzer_db)
    assert compat.check_installation(tmp_path, files, root).status == compat.NOT_GAME


def test_nothing_recorded_is_unverifiable(game):
    assert compat.check_installation(game[0], [], "").status == compat.UNVERIFIABLE
