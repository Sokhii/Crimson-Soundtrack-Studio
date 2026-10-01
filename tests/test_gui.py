"""GUI tests (offscreen). They drive the real widgets through their public slots."""

import json
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


def test_ai_page_and_describe_music(app, window, tmp_path, analyzer_db):
    from soundtrack_studio.ai.runtime import ScriptedBackend

    good = ('{"mood": ["dark"], "emotion": [], "atmosphere": ["vast"], "instrumentation": [], "style": ["orchestral"],'
            ' "themes": [], "energy": 40, "darkness": 80, "tension": 50, "valence": 20, "vocal_presence": null,'
            ' "confidence": 70, "summary": "Dark and vast."}')
    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"GGUF" + b"\0" * 100)
    studio = window.studio
    studio.select_model(studio.add_custom_model(model_file).id)
    studio.backend_factory = lambda model: ScriptedBackend([good], model_id=model.id)
    studio.import_analyzer(analyzer_db)
    music = tmp_path / "Music"
    write_test_flac(music / "a.flac", seconds=2, tags={"TITLE": "Night"})
    studio.set_library_path(music)
    studio.scan_library()
    window.nav.setCurrentRow(3)  # AI Model page
    app.processEvents()
    page = window.pages["ai"]
    assert "Recommended tier" in page.hw_label.text() and "m" in page.active_label.text()
    window.describe_music()
    wait(app, window)
    assert not window.errors
    assert all(p.source == "llm" for p in studio.profiles("track").values())
    window.nav.setCurrentRow(2)
    library = window.pages["library"]
    library.table.selectRow(0)
    assert "Dark and vast." in library.details.toPlainText() and library.edit_btn.isEnabled()


def test_matching_page_review_flow(app, window, tmp_path, game, analyzer_db):
    studio = window.studio
    studio.import_analyzer(analyzer_db)
    music = tmp_path / "Music"
    for name, title, tone in (("a.flac", "Dark Requiem", 110), ("b.flac", "Village Dawn", 440)):
        write_test_flac(music / name, seconds=45, bpm=None, tone_hz=tone, tags={"TITLE": title})
    studio.set_library_path(music)
    studio.scan_library()
    window.nav.setCurrentRow(4)  # Matching
    page = window.pages["matching"]
    page.find_matches()
    wait(app, window)
    assert not window.errors
    assert page.table.rowCount() == 3 and all(r["status"] == "proposed" for r in page.rows)
    page.table.selectRow(0)
    page.show_row(0)
    assert "Replacement:" in page.details.toPlainText() and "Length:" in page.details.toPlainText()
    page.accept_current()
    assert page.rows[0]["status"] == "accepted"
    page.table.selectRow(1)
    page.show_row(1)
    page.keep_current()
    statuses = sorted(r["status"] for r in page.rows)
    assert statuses == ["accepted", "keep original", "proposed"]
    page.status_filter.setCurrentText("Accepted / chosen")
    visible = [i for i in range(page.table.rowCount()) if not page.table.isRowHidden(i)]
    assert len(visible) == 1
    page.include_short.setChecked(True)
    assert page.table.rowCount() == 4


def test_build_page_builds_a_mod(app, window, tmp_path, monkeypatch):
    from soundtrack_studio.testing.fixtures import ANALYZER_FAKE_INSTALL_DB, extract_analyzer_fake_install

    messages = []
    monkeypatch.setattr(window, "info", lambda title, text: messages.append((title, text)))
    studio = window.studio
    game = extract_analyzer_fake_install(tmp_path / "Game")
    studio.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
    studio.set_game_path(game)
    write_test_flac(tmp_path / "Music" / "a.flac", seconds=60, tags={"TITLE": "Theme"})
    studio.set_library_path(tmp_path / "Music")
    studio.scan_library()
    from soundtrack_studio.testing.fake_wwise import install_fake_wwise

    monkeypatch.delenv("CSS_WWISE_CONSOLE", raising=False)
    monkeypatch.delenv("WWISEROOT", raising=False)
    monkeypatch.setattr("soundtrack_studio.compiler.wwise._candidates", lambda: iter(()))
    window.nav.setCurrentRow(5)  # Build
    page = window.pages["build"]
    assert not page.build_btn.isEnabled()  # nothing confirmed yet
    track = studio.library_tracks()[0]["id"]
    studio.match_store().choose("2001", track)
    page.refresh()
    # the Vorbis format (default) needs Wwise; without it the page says so and PCM still works
    assert page.encoder_box.currentData() == "wwise_vorbis" and "not installed" in page.wwise_label.text()
    page.find_wwise()                                       # says where it looked instead of doing nothing
    assert messages[-1][0] == "Find Wwise" and "not found" in messages[-1][1] and "Choose" in messages[-1][1]
    messages.clear()
    assert not page.build_btn.isEnabled() and "Install Wwise" in page.summary.text()
    page.encoder_box.setCurrentIndex(page.encoder_box.findData("pcm"))
    assert studio.settings.build_encoder == "pcm" and page.build_btn.isEnabled()
    page.encoder_box.setCurrentIndex(page.encoder_box.findData("wwise_vorbis"))
    monkeypatch.setenv("CSS_WWISE_CONSOLE", str(install_fake_wwise(tmp_path / "wwise")))
    page.refresh()
    assert "Found" in page.wwise_label.text() and page.test_wwise_btn.isEnabled()
    page.find_wwise()
    assert messages[-1][0] == "Find Wwise" and "Found Wwise" in messages[-1][1]
    messages.clear()
    assert page.build_btn.isEnabled() and "1</b> confirmed" in page.summary.text()
    page.test_wwise()
    wait(app, window, timeout=60)
    assert messages and messages[-1][0] == "Test Wwise" and "Wwise works" in messages[-1][1]
    messages.clear()
    page.name.setText("GUI Mod")
    page.make_zip.setChecked(False)
    page.build()
    wait(app, window, timeout=60)
    assert not window.errors, window.errors
    assert messages and messages[0][0] == "Mod built"
    assert (studio.paths.output / "GUI Mod" / "manifest.json").is_file()
    report = json.loads((studio.paths.output / "GUI Mod.build-report.json").read_text())
    assert report["codec"] == "vorbis"
    assert page.history.rowCount() == 1 and page.history.item(0, 1).text() == "completed"


def test_listening_model_panel_and_game_audio_buttons(app, window, tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from test_game_audio import _fake_decoder

    from soundtrack_studio.testing.fake_listening import install_fake_listening_model
    from soundtrack_studio.testing.fixtures import ANALYZER_FAKE_INSTALL_DB, extract_analyzer_fake_install

    studio = window.studio
    ai = window.pages["ai"]
    ai.refresh()
    assert ai.listen_card.isVisible() or not ai.isVisible()
    assert "Not downloaded" in ai.listen_status.text() and not ai.listen_toggle.isEnabled()
    install_fake_listening_model(studio)
    ai.refresh()
    assert ai.listen_toggle.isEnabled() and ai.listen_toggle.text() == "Turn on"
    ai._listen_toggle()
    assert studio.active_listening_model() is not None and ai.listen_toggle.text() == "Turn off"
    monkeypatch.setattr(window, "info", lambda title, text: window.__dict__.setdefault("infos", []).append(text))
    ai._listen_test()
    wait(app, window)
    assert not window.errors and "listened to test sounds correctly" in window.infos[-1]

    game = extract_analyzer_fake_install(tmp_path / "Crimson Desert")
    studio.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
    studio.set_game_path(game)
    monkeypatch.setenv("CSS_VGMSTREAM", str(_fake_decoder(tmp_path)))
    window.refresh(reload_game_data=True)
    wait(app, window)
    page = window.pages["game"]
    page.set_model(studio.game_model())
    assert page.audio_btn.isEnabled() and "0 of 4" in page.audio_status.text()
    window.test_game_audio()
    wait(app, window)
    assert "Decoding works" in window.infos[-1]
    window.analyze_game_audio()
    wait(app, window)
    page.set_model(studio.game_model())
    assert "4 of 4" in page.audio_status.text() and "4 listened to" in page.audio_status.text()
    page._current = ("node", studio.game_model().cues[0].segment_id)
    page.show_current()
    html = page.details.toHtml()
    assert "Measured from the game audio" in html and "Heard by the listening model" in html


def test_redescribe_buttons_and_dialog(app, window):
    from soundtrack_studio.ui.main_window import RedescribeDialog

    assert window.pages["ai"].redescribe_btn.text().startswith("Re-describe")
    assert window.pages["matching"].redescribe_btn.text().startswith("Re-describe")
    dialog = RedescribeDialog(window)
    assert set(dialog.chosen()) == {"cue", "track"}
    dialog.game.setChecked(False)
    assert dialog.chosen() == ("track",)
    dialog.library.setChecked(False)
    assert dialog.chosen() == ()


def test_matching_page_approve_menu_controls_which_levels_are_approved(app, window):
    studio = window.studio
    page = window.pages["matching"]
    assert page.accept_all_btn.text() == "Approve" and page.approve_levels() == ["high"]
    page.approve_actions["medium"].setChecked(True)
    assert studio.settings.approve_levels == ["high", "medium"]
    page.approve_actions["high"].setChecked(False)
    assert studio.settings.approve_levels == ["medium"]
    assert "medium" in page.accept_all_btn.toolTip()


def test_matching_page_max_uses_setting_is_saved_and_follows_reuse(app, window):
    studio = window.studio
    page = window.pages["matching"]
    assert page.max_uses.value() == 0 and page.max_uses.specialValueText() == "Automatic"
    page.max_uses.setValue(3)
    assert studio.settings.max_uses_per_track == 3
    page.allow_reuse.setChecked(False)
    assert not page.max_uses.isEnabled()
    page.allow_reuse.setChecked(True)
    assert page.max_uses.isEnabled()
    page.max_uses.setValue(0)
    assert studio.settings.max_uses_per_track == 0


def test_matching_page_toggle_switches_between_standout_and_legacy(app, window, tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from soundtrack_studio.testing.fake_listening import install_fake_listening_model

    studio = window.studio
    page = window.pages["matching"]
    page.refresh()
    assert page.standout.isChecked() and not page.standout.isEnabled()          # no listening model: legacy only
    install_fake_listening_model(studio)
    studio.select_listening_model("clap-larger-music-speech")
    page.refresh()
    assert page.standout.isEnabled() and page.standout.isChecked()
    page.standout.setChecked(False)
    assert studio.settings.matching_mode == "legacy"
    page.standout.setChecked(True)
    assert studio.settings.matching_mode == "standout"
