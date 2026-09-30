from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from soundtrack_studio.app_paths import AppPaths  # noqa: E402
from soundtrack_studio.testing.fixtures import make_fake_game, write_analyzer_db  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
# Produced by Crimson Desert Analyzer 0.1.0's own pipeline (--selftest on its synthetic install).
REAL_ANALYZER_FIXTURE = FIXTURES / "analyzer_selftest_v1.sqlite3"


@pytest.fixture
def paths(tmp_path) -> AppPaths:
    return AppPaths(tmp_path / "Crimson Soundtrack Studio").ensure()


@pytest.fixture
def game(tmp_path):
    root = tmp_path / "Games" / "Crimson Desert"
    files = make_fake_game(root)
    return root, files


@pytest.fixture
def analyzer_db(tmp_path, game) -> Path:
    root, files = game
    return write_analyzer_db(tmp_path / "Analyzer" / "data" / "database" / "studio.sqlite3", game_root=root, game_files=files)


@pytest.fixture
def studio(paths):
    from soundtrack_studio.services import Studio

    s = Studio(paths)
    s.create_project("Test Project")
    yield s
    s.shutdown()


def fingerprint(folder: Path) -> dict:
    """Name, size, mtime and bytes of every file in a folder (detects any modification or new side file)."""

    out = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            st = p.stat()
            out[p.relative_to(folder).as_posix()] = (st.st_size, st.st_mtime_ns, p.read_bytes())
    return out
