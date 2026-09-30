"""Checks against a real Analyzer database from a real Crimson Desert installation (local only).

    CSS_REAL_ANALYZER_DB="D:/Crimson Desert Analyzer/data/database/studio.sqlite3" \\
    CSS_REAL_GAME_DIR="D:/SteamLibrary/steamapps/common/Crimson Desert" \\
    python -m pytest tests/test_real_analyzer_db.py -m analyzer_db -s

Game data is never committed; these tests skip when the variables are not set.
"""

import os
import time
from pathlib import Path

import pytest
from conftest import fingerprint

from soundtrack_studio.analyzer_db import compat
from soundtrack_studio.services import Studio

REAL_DB = os.environ.get("CSS_REAL_ANALYZER_DB")
REAL_GAME = os.environ.get("CSS_REAL_GAME_DIR")

pytestmark = [pytest.mark.analyzer_db,
              pytest.mark.skipif(not REAL_DB, reason="set CSS_REAL_ANALYZER_DB to a real Analyzer database")]


def test_real_database(paths):
    db = Path(REAL_DB)
    before = {k: v[:2] for k, v in fingerprint(db.parent).items()}  # size + mtime (no need to hash GBs)
    studio = Studio(paths)
    studio.create_project("Real")
    started = time.monotonic()
    imported = studio.import_analyzer(db)
    print(f"\nimport: {time.monotonic() - started:.1f}s  schema={imported.schema_version} counts={imported.report.counts}")
    for issue in imported.report.issues:
        print(f"  {issue.severity}: {issue.message}")
    started = time.monotonic()
    model = studio.game_model()
    print(f"model: {time.monotonic() - started:.1f}s  {model.stats}")
    for w in model.warnings:
        print(f"  warning: {w}")
    assert imported.report.ok
    assert model.stats["cues"] > 0 and model.stats["tracks"] > 0
    started = time.monotonic()
    studio._model = None
    studio.game_model()
    print(f"model from cache: {time.monotonic() - started:.2f}s")
    if REAL_GAME:
        report = studio.set_game_path(Path(REAL_GAME))
        print(f"install check: {report.status} ({report.checked_files} files)")
        assert report.status in (compat.MATCH, compat.PROBABLE_MATCH, compat.MISMATCH)
    studio.shutdown()
    assert {k: v[:2] for k, v in fingerprint(db.parent).items()} == before
