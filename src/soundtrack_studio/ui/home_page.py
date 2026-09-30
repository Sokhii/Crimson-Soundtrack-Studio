"""Home / Project page: the primary workflow in one place."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from ..analyzer_db.importer import ANALYZER_DB_RELATIVE
from .widgets import STATE_ICON, Card, DropZone, esc


class HomePage(QWidget):
    def __init__(self, host) -> None:
        super().__init__()
        self.host = host
        self.setObjectName("page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        inner = QWidget()
        inner.setObjectName("page")
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(14)

        head = QHBoxLayout()
        self.title = QLabel("No project open")
        self.title.setObjectName("pageTitle")
        head.addWidget(self.title, 1)
        switch = QPushButton("Projects…")
        switch.clicked.connect(host.choose_project)
        head.addWidget(switch)
        layout.addLayout(head)

        columns = QHBoxLayout()
        columns.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(14)
        right = QVBoxLayout()
        right.setSpacing(14)
        columns.addLayout(left, 3)
        columns.addLayout(right, 2)
        layout.addLayout(columns)

        # 1. game
        self.game_card = Card("1 · Crimson Desert installation")
        self.game_path = QLabel()
        self.game_path.setWordWrap(True)
        self.game_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.game_status = QLabel()
        self.game_status.setWordWrap(True)
        row = QHBoxLayout()
        self.game_btn = QPushButton("Choose game folder…")
        self.game_btn.clicked.connect(self.choose_game)
        self.recheck_btn = QPushButton("Check again")
        self.recheck_btn.clicked.connect(self.recheck_game)
        row.addWidget(self.game_btn)
        row.addWidget(self.recheck_btn)
        row.addStretch(1)
        for w in (self.game_path, self.game_status):
            self.game_card.body.addWidget(w)
        self.game_card.body.addLayout(row)
        left.addWidget(self.game_card)

        # 2. analyzer
        self.db_card = Card("2 · Analyzer database")
        self.drop = DropZone("Drop the Crimson Desert Analyzer database here (the file or the whole Analyzer folder)\n"
                             "or click to choose it.")
        self.drop.dropped.connect(self.import_db)
        self.drop.clicked.connect(self.browse_db)
        self.db_info = QLabel()
        self.db_info.setWordWrap(True)
        self.db_info.setTextFormat(Qt.TextFormat.RichText)
        self.db_card.body.addWidget(self.drop)
        self.db_card.body.addWidget(self.db_info)
        left.addWidget(self.db_card)

        # 3. music
        self.lib_card = Card("3 · Music library")
        self.lib_path = QLabel()
        self.lib_path.setWordWrap(True)
        self.lib_info = QLabel()
        self.lib_info.setWordWrap(True)
        row = QHBoxLayout()
        self.lib_btn = QPushButton("Choose music folder…")
        self.lib_btn.clicked.connect(self.choose_library)
        self.scan_btn = QPushButton("Scan library")
        self.scan_btn.clicked.connect(host.scan_library)
        row.addWidget(self.lib_btn)
        row.addWidget(self.scan_btn)
        row.addStretch(1)
        for w in (self.lib_path, self.lib_info):
            self.lib_card.body.addWidget(w)
        self.lib_card.body.addLayout(row)
        note = QLabel("Supported now: FLAC. Your files are only read, never changed.")
        note.setObjectName("muted")
        self.lib_card.body.addWidget(note)
        left.addWidget(self.lib_card)

        # 4. AI
        ai = Card("4 · AI model")
        combo = QComboBox()
        combo.addItems(["Low", "Medium (recommended)", "High", "Custom…"])
        combo.setCurrentIndex(1)
        combo.setEnabled(False)
        ai.body.addWidget(combo)
        label = QLabel("The built-in local AI (no internet or extra software needed) arrives in the next phase. "
                       "Models will be stored in this program's own 'models' folder.")
        label.setWordWrap(True)
        label.setObjectName("muted")
        ai.body.addWidget(label)
        left.addWidget(ai)
        left.addStretch(1)

        # right column: workflow + warnings + build
        steps = Card("Progress")
        self.steps = QLabel()
        self.steps.setTextFormat(Qt.TextFormat.RichText)
        self.steps.setWordWrap(True)
        steps.body.addWidget(self.steps)
        right.addWidget(steps)
        warn = Card("Warnings")
        self.warnings = QLabel()
        self.warnings.setTextFormat(Qt.TextFormat.RichText)
        self.warnings.setWordWrap(True)
        warn.body.addWidget(self.warnings)
        right.addWidget(warn)
        build = Card("Last build")
        self.build = QLabel("No builds yet. Building the mod arrives in a later version.")
        self.build.setWordWrap(True)
        self.build.setObjectName("muted")
        build.body.addWidget(self.build)
        right.addWidget(build)
        right.addStretch(1)

    # ---------------------------------------------------------------- actions
    def choose_game(self) -> None:
        path = self.host.dialogs.folder(self, "Select your Crimson Desert folder")
        if path:
            self.host.run_job("Checking the Crimson Desert installation",
                              lambda report, cancelled: self.host.studio.set_game_path(Path(path)),
                              on_done=lambda _r: self.host.refresh())

    def recheck_game(self) -> None:
        self.host.run_job("Checking the Crimson Desert installation",
                          lambda report, cancelled: self.host.studio.check_game(), on_done=lambda _r: self.host.refresh())

    def browse_db(self) -> None:
        path = self.host.dialogs.open_file(
            self, "Select the Crimson Desert Analyzer database",
            "Analyzer database (*.sqlite3 *.sqlite *.db);;All files (*)")
        if path:
            self.import_db(path)

    def import_db(self, path: str) -> None:
        self.host.run_job(
            "Importing the Analyzer database",
            lambda report, cancelled: self.host.studio.import_analyzer(Path(path), report, cancelled),
            on_done=lambda _r: self.host.refresh(reload_game_data=True))

    def choose_library(self) -> None:
        path = self.host.dialogs.folder(self, "Select the folder containing your music")
        if path:
            try:
                self.host.studio.set_library_path(Path(path))
            except Exception as exc:  # noqa: BLE001 - shown to the user
                self.host.error(exc)
                return
            self.host.refresh()
            self.host.scan_library()

    # ---------------------------------------------------------------- refresh
    def refresh(self) -> None:
        studio = self.host.studio
        has_project = studio.project is not None
        for w in (self.game_card, self.db_card, self.lib_card):
            w.setEnabled(has_project)
        if not has_project:
            self.title.setText("No project open")
            self.steps.setText("Create or open a project to begin.")
            self.warnings.setText("")
            return
        status = studio.project_status()
        self.title.setText(status.name)
        self.game_path.setText(status.game_path or "No folder selected.")
        self.recheck_btn.setEnabled(bool(status.game_path))
        check = status.game_check
        if check:
            details = check["details"]
            extra = ""
            if details.get("missing") or details.get("size_mismatch"):
                extra = f" ({len(details.get('missing', []))} files missing, {len(details.get('size_mismatch', []))} changed)"
            self.game_status.setText(details.get("summary", check["status"]) + extra)
        else:
            self.game_status.setText("")
        if status.analyzer:
            a = status.analyzer
            c = a.get("counts", {})
            self.db_info.setText(
                f"<b>Schema {esc(a['schema_version'])}</b> · imported {esc(a['imported_at'])}<br>"
                f"{c.get('banks', 0)} soundbanks · {c.get('music_objects', 0)} music structures · "
                f"{c.get('music_media', 0)} music audio files<br>"
                f"<span style='color:gray'>Source: {esc(a['source_path'])}<br>The original file is never changed; "
                f"the Studio works on its own copy.</span>")
        else:
            self.db_info.setText(f"<span style='color:gray'>Usually found at "
                                 f"<i>{esc(ANALYZER_DB_RELATIVE.as_posix())}</i> inside the Analyzer folder.</span>")
        self.lib_path.setText(status.library_path or "No folder selected.")
        counts = status.library_counts
        self.scan_btn.setEnabled(bool(status.library_path))
        if counts.get("total"):
            self.lib_info.setText(f"{counts.get('ok', 0)} tracks analysed · {counts.get('error', 0)} unreadable · "
                                  f"{counts.get('missing', 0)} missing · {counts.get('duplicates', 0)} duplicates")
        else:
            self.lib_info.setText("")
        lines = []
        for step in status.steps:
            icon, color = STATE_ICON[step.state]
            muted = step.state == "unavailable"
            title = f"<span style='color:gray'>{esc(step.title)}</span>" if muted else esc(step.title)
            detail = f"<br><span style='color:gray;font-size:11px'>&nbsp;&nbsp;&nbsp;&nbsp;{esc(step.detail)}</span>" \
                if step.detail and not muted else ""
            lines.append(f"<div style='margin:3px 0'><span style='color:{color};font-weight:bold'>{icon}</span>"
                         f"&nbsp; {step.number}. {title}{detail}</div>")
        self.steps.setText("".join(lines))
        self.warnings.setText("".join(f"<div style='margin:3px 0'>⚠ {esc(w)}</div>" for w in status.warnings)
                              or "<span style='color:gray'>No warnings.</span>")
