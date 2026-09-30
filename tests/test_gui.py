"""GUI tests (offscreen). They drive the real widgets through their public slots."""

import time

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from soundtrack_studio.config import Settings  # noqa: E402
from soundtrack_studio.testing.fixtures import write_test_flac  # noqa: E402
from soundtrack_studio.ui import main_window  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait(app, window, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        QThreadPool.globalInstance().waitForDone(20)
        app.processEvents()
        if window.job is None:
            return
    raise AssertionError("background job did not finish")


@pytest.fixture
def window(app, paths, monkeypatch):
    errors = []
    monkeypatch.setattr(main_window, "show_error", lambda parent, exc, log_path=None: errors.append(exc))
    w = main_window.MainWindow(paths, Settings.load(paths))
    w.errors = errors
    w.studio.create_project("GUI")
    w.refresh(reload_game_data=True)
    yield w
    w.close()


def test_full_phase1_flow_through_the_gui(app, window, tmp_path, game, analyzer_db):
    home = window.pages["home"]
    window.run_job("game", lambda r, c: window.studio.set_game_path(game[0]), on_done=lambda _r: window.refresh())
    wait(app, window)
    home.import_db(str(analyzer_db.parents[2]))  # dropping the whole Analyzer folder
    wait(app, window)
    assert not window.errors
    assert "Schema 1" in home.db_info.text()
    assert "matches" in home.game_status.text()

    window.nav.setCurrentRow(1)
    wait(app, window)
    game_page = window.pages["game"]
    assert game_page.model is not None and game_page.cues.rowCount() == 4
    assert game_page.tree.topLevelItemCount() == 1
    game_page.filter.setText("Desert")
    visible = [r for r in range(4) if not game_page.cues.isRowHidden(r)]
    assert len(visible) == 2
    game_page.tree.setCurrentItem(game_page.tree.topLevelItem(0))
    assert "BGM_Region" in game_page.details.toPlainText()

    music = tmp_path / "Music"
    write_test_flac(music / "x" / "Track 音.flac", seconds=3, tags={"TITLE": "Track 音"})
    window.studio.set_library_path(music)
    window.scan_library()
    wait(app, window)
    library = window.pages["library"]
    assert library.model.rowCount() == 1
    library.table.selectRow(0)
    assert "Track 音" in library.details.toPlainText()
    assert "Measurements" in library.details.toPlainText()


def test_bad_database_shows_friendly_error(app, window, tmp_path):
    bogus = tmp_path / "bogus.sqlite3"
    bogus.write_bytes(b"x" * 200)
    window.pages["home"].import_db(str(bogus))
    wait(app, window)
    assert len(window.errors) == 1
    assert window.errors[0].message == "The selected file is not a SQLite database."
    assert window.job is None


def test_all_pages_render_without_project(app, paths, monkeypatch):
    monkeypatch.setattr(main_window, "show_error", lambda *a, **k: None)
    w = main_window.MainWindow(paths, Settings.load(paths))
    for row in range(w.nav.count()):
        w.nav.setCurrentRow(row)
        app.processEvents()
    assert w.pages["home"].title.text() == "No project open"
    w.close()
