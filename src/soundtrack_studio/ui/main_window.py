"""Main window: navigation, project handling, background jobs and error display."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

from PySide6.QtCore import QByteArray, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QHBoxLayout, QInputDialog, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton, QStackedWidget,
                               QVBoxLayout, QWidget)

from .. import APP_DISPLAY_NAME, __version__
from ..app_paths import AppPaths
from ..config import Settings
from ..library.scanner import ScanProgress
from ..services import Studio
from . import workers
from .game_page import GameDataPage
from .home_page import HomePage
from .library_page import LibraryPage
from .widgets import STYLE, Dialogs, esc, show_error

log = logging.getLogger(__name__)


class PlaceholderPage(QWidget):
    def __init__(self, title: str, text: str) -> None:
        super().__init__()
        self.setObjectName("page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        head = QLabel(title)
        head.setObjectName("pageTitle")
        layout.addWidget(head)
        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(body)
        layout.addStretch(1)

    def refresh(self) -> None:
        pass


class ProjectDialog(QDialog):
    """Start screen: open an existing project or create a new one."""

    def __init__(self, studio: Studio, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.studio = studio
        self.chosen: Optional[Path] = None
        self.new_name: Optional[str] = None
        self.setWindowTitle("Projects")
        self.resize(520, 380)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Open a project</b> or create a new one. Projects are stored in the "
                                "<i>projects</i> folder next to the program."))
        self.list = QListWidget()
        for info in studio.list_projects():
            label = info.name + (f"   (last changed {info.modified_at[:16].replace('T', ' ')})" if info.modified_at else "")
            if info.error:
                label += f"   [{info.error}]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(info.folder))
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)
        self.list.itemDoubleClicked.connect(lambda _i: self.open_selected())
        layout.addWidget(self.list, 1)
        row = QHBoxLayout()
        new_btn = QPushButton("New project…")
        new_btn.clicked.connect(self.create)
        open_btn = QPushButton("Open")
        open_btn.setDefault(True)
        open_btn.setEnabled(self.list.count() > 0)
        open_btn.clicked.connect(self.open_selected)
        row.addWidget(new_btn)
        row.addStretch(1)
        row.addWidget(open_btn)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        layout.addLayout(row)

    def open_selected(self) -> None:
        item = self.list.currentItem()
        if item:
            self.chosen = Path(item.data(Qt.ItemDataRole.UserRole))
            self.accept()

    def create(self) -> None:
        name, ok = QInputDialog.getText(self, "New project", "Project name:", text="My Soundtrack")
        if ok and name.strip():
            self.new_name = name.strip()
            self.accept()


class MainWindow(QMainWindow):
    def __init__(self, paths: AppPaths, settings: Settings) -> None:
        super().__init__()
        self.paths = paths
        self.studio = Studio(paths, settings)
        self.dialogs = Dialogs(native=settings.use_native_dialogs)
        self.job: Optional[workers.Job] = None
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.resize(1280, 820)
        if settings.window_geometry:
            try:
                self.restoreGeometry(QByteArray.fromHex(settings.window_geometry.encode()))
            except Exception:  # noqa: BLE001 - bad saved geometry is harmless
                pass

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(190)
        self.stack = QStackedWidget()
        layout.addWidget(self.nav)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.pages: Dict[str, QWidget] = {}
        self._add_page("home", "Home", HomePage(self))
        self._add_page("game", "Game Data", GameDataPage(self))
        self._add_page("library", "Music Library", LibraryPage(self))
        self._add_page("matching", "Matching", PlaceholderPage(
            "Matching",
            "Thematic matching arrives in a later version.<br><br>It will propose which of your tracks fits each "
            "game cue by mood, atmosphere, energy, instrumentation and style (not by gameplay category), explain "
            "why, and let you accept, reject or override every proposal. Your decisions always win over the AI."), False)
        self._add_page("build", "Build", PlaceholderPage(
            "Build",
            "Building the mod arrives in a later version.<br><br>The Studio will create a separate mod package for "
            "a mod manager such as DMM. Your Crimson Desert installation is never modified."), False)
        self.nav.currentRowChanged.connect(self._page_changed)
        self.nav.setCurrentRow(0)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self.cancel_job)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_btn)
        self._menus()

    # ---------------------------------------------------------------- setup
    def _add_page(self, key: str, title: str, page: QWidget, enabled: bool = True) -> None:
        self.pages[key] = page
        self.stack.addWidget(page)
        item = QListWidgetItem(title if enabled else f"{title}  (later)")
        if not enabled:
            item.setForeground(Qt.GlobalColor.gray)
        self.nav.addItem(item)

    def _menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, slot in (("&Projects…", self.choose_project), ("Open program &folder", self.open_app_folder),
                           ("Open &log folder", lambda: self._open_folder(self.paths.logs))):
            action = QAction(text, self)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("E&xit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("&About", self)
        about.triggered.connect(self.about)
        help_menu.addAction(about)

    def _page_changed(self, row: int) -> None:
        self.stack.setCurrentIndex(row)
        key = list(self.pages)[row]
        if key == "game":
            self.load_game_model()
        elif hasattr(self.pages[key], "refresh"):
            self.pages[key].refresh()

    # -------------------------------------------------------------- projects
    def start(self) -> None:
        if self.studio.open_last_project() is None:
            self.choose_project()
        self.refresh(reload_game_data=True)

    def choose_project(self) -> None:
        if self.job is not None:
            return
        dialog = ProjectDialog(self.studio, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            if dialog.new_name:
                self.studio.create_project(dialog.new_name)
            elif dialog.chosen:
                self.studio.open_project(dialog.chosen)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.error(exc)
        self.nav.setCurrentRow(0)
        self.refresh(reload_game_data=True)

    # ------------------------------------------------------------------ jobs
    def run_job(self, title: str, fn: Callable, on_done: Optional[Callable] = None) -> None:
        if self.job is not None:
            QMessageBox.information(self, APP_DISPLAY_NAME, "Please wait until the current task has finished.")
            return
        job = workers.Job(fn)
        self.job = job
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.statusBar().showMessage(title + "…")
        job.signals.progress.connect(self._job_progress)

        def finished(result) -> None:
            self._job_ended()
            self.statusBar().showMessage(title + " - done.", 5000)
            if on_done:
                on_done(result)

        def failed(exc) -> None:
            self._job_ended()
            self.statusBar().showMessage(title + " - stopped.", 5000)
            self.error(exc)
            self.refresh()

        job.signals.done.connect(finished)
        job.signals.error.connect(failed)
        workers.start(job)

    def _job_progress(self, text: str, done: int, total: int) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
            self.statusBar().showMessage(f"{text} ({done}/{total})")
        else:
            self.progress.setRange(0, 0)
            self.statusBar().showMessage(text + "…")

    def _job_ended(self) -> None:
        self.job = None
        self.progress.setVisible(False)
        self.cancel_btn.setVisible(False)

    def cancel_job(self) -> None:
        if self.job is not None:
            self.job.cancel()
            self.statusBar().showMessage("Cancelling…")

    def scan_library(self) -> None:
        def work(report, cancelled):
            def progress(p: ScanProgress) -> None:
                report(p.phase, p.done, p.total)
            return self.studio.scan_library(progress, cancelled)

        def done(stats) -> None:
            self.refresh()
            msg = (f"Library scan finished: {stats.files_found} files, {stats.analyzed} analysed, "
                   f"{stats.unchanged} unchanged, {stats.errors} unreadable.")
            self.statusBar().showMessage(msg, 10000)

        self.run_job("Scanning music library", work, on_done=done)

    def load_game_model(self) -> None:
        page: GameDataPage = self.pages["game"]
        if self.job is not None:  # reloaded when the running task finishes
            return
        if self.studio.project is None or self.studio.project.active_analyzer() is None:
            page.set_model(None)
            return
        if page.model is not None and self.studio._model is page.model:
            return
        self.run_job("Reading game music structures",
                     lambda report, cancelled: self.studio.game_model(lambda text: report(text)),
                     on_done=page.set_model)

    # --------------------------------------------------------------- refresh
    def refresh(self, reload_game_data: bool = False) -> None:
        project = self.studio.project
        self.setWindowTitle(f"{project.name} - {APP_DISPLAY_NAME}" if project else APP_DISPLAY_NAME)
        self.pages["home"].refresh()
        self.pages["library"].refresh()
        if reload_game_data:
            self.pages["game"].set_model(None)
            if self.stack.currentWidget() is self.pages["game"]:
                self.load_game_model()

    def error(self, exc: BaseException) -> None:
        show_error(self, exc, self.paths.logs / "studio.log")

    # ----------------------------------------------------------------- misc
    def _open_folder(self, folder: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def open_app_folder(self) -> None:
        self._open_folder(self.paths.root)

    def about(self) -> None:
        QMessageBox.about(self, APP_DISPLAY_NAME, (
            f"<b>{APP_DISPLAY_NAME}</b> {esc(__version__)}<br><br>Replace Crimson Desert's music with your own "
            "legally obtained music, as a separate mod.<br><br>This program is portable: all of its settings, "
            f"projects, models, caches and logs are stored in<br><i>{esc(self.paths.root)}</i><br><br>"
            "Open-source software (MIT licence). No music or game files are included."))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.job is not None:
            self.job.cancel()
            workers.QThreadPool.globalInstance().waitForDone(5000)
        self.studio.settings.window_geometry = bytes(self.saveGeometry().toHex()).decode()
        self.studio.shutdown()
        super().closeEvent(event)


def run_gui(paths: AppPaths, settings: Settings, smoke: bool = False) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CrimsonSoundtrackStudio")
    app.setApplicationDisplayName(APP_DISPLAY_NAME)
    app.setStyleSheet(STYLE)
    window = MainWindow(paths, settings)
    window.show()
    if smoke:
        # CI: open the window, visit every page, close. No project dialog (it would block).
        window.studio.open_last_project()
        window.refresh(reload_game_data=True)

        def visit(i: int = 0) -> None:
            if i < window.nav.count():
                window.nav.setCurrentRow(i)
                QTimer.singleShot(200, lambda: visit(i + 1))
            else:
                window.close()
        QTimer.singleShot(300, visit)
    else:
        QTimer.singleShot(0, window.start)
    code = app.exec()
    log.info("GUI closed (exit code %s)", code)
    return code
