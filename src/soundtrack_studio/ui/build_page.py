"""Build page: package settings, confirmed replacements, build + validation, history."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..app_paths import is_file, os_path
from ..compiler.build import BuildSettings
from ..compiler.validate import REPORT_NAME
from .widgets import Card, esc, fmt_bytes

LAYOUT_LABELS = {"crimson_browser": "Mod manager package (manifest.json + files/) — DMM / CDUMM",
                 "package_folders": "Package folders only (0004/…) — alternative layout"}


class BuildPage(QWidget):
    def __init__(self, host) -> None:
        super().__init__()
        self.host = host
        self.setObjectName("page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        title = QLabel("Build")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        intro = QLabel("Builds a separate mod package in this program's 'output' folder. Your Crimson Desert "
                       "installation and your music files are only read, never changed. Install the result with "
                       "your mod manager.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        top = QHBoxLayout()
        settings_card = Card("Package")
        form = QFormLayout()
        self.name = QLineEdit()
        self.author = QLineEdit()
        self.version = QLineEdit()
        self.description = QLineEdit()
        self.layout_box = QComboBox()
        for key, label in LAYOUT_LABELS.items():
            self.layout_box.addItem(label, key)
        self.normalize = QCheckBox("Even out loudness between tracks")
        self.target = QDoubleSpinBox()
        self.target.setRange(-30.0, -8.0)
        self.target.setSuffix(" dBFS RMS")
        self.make_zip = QCheckBox("Also create a .zip file")
        for label, widget in (("Mod name", self.name), ("Author", self.author), ("Version", self.version),
                              ("Description", self.description), ("Layout", self.layout_box), ("", self.normalize),
                              ("Loudness target", self.target), ("", self.make_zip)):
            form.addRow(label, widget)
        settings_card.body.addLayout(form)
        top.addWidget(settings_card, 3)
        summary_card = Card("Ready to build")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        summary_card.body.addWidget(self.summary)
        self.build_btn = QPushButton("Build mod")
        self.build_btn.setMinimumHeight(36)
        self.build_btn.clicked.connect(self.build)
        summary_card.body.addWidget(self.build_btn)
        top.addWidget(summary_card, 2)
        layout.addLayout(top)

        history_label = QLabel("Build history")
        history_label.setObjectName("cardTitle")
        layout.addWidget(history_label)
        self.history = QTableWidget(0, 5)
        self.history.setHorizontalHeaderLabels(["Finished", "Status", "Replacements", "Size", "Output"])
        self.history.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.history.verticalHeader().setVisible(False)
        layout.addWidget(self.history, 1)
        row = QHBoxLayout()
        self.open_btn = QPushButton("Open output folder")
        self.open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.host.paths.output))))
        self.report_btn = QPushButton("Show build report")
        self.report_btn.clicked.connect(self.show_report)
        row.addWidget(self.open_btn)
        row.addWidget(self.report_btn)
        row.addStretch(1)
        layout.addLayout(row)
        self.builds = []

    def refresh(self) -> None:
        studio = self.host.studio
        if studio.project is None:
            self.summary.setText("Create or open a project first.")
            self.build_btn.setEnabled(False)
            return
        s = studio.build_settings()
        self.name.setText(s.mod_name)
        self.author.setText(s.author)
        self.version.setText(s.version)
        self.description.setText(s.description)
        self.layout_box.setCurrentIndex(max(0, self.layout_box.findData(s.layout)))
        self.normalize.setChecked(s.normalize)
        self.target.setValue(s.target_rms_dbfs)
        self.make_zip.setChecked(s.make_zip)
        mapping = studio.match_store().final_mapping()
        counts = studio.match_store().status_counts()
        status = studio.project_status()
        problems = []
        if not status.game_path:
            problems.append("Select your Crimson Desert folder (Home page).")
        elif status.game_check and status.game_check["status"] not in ("match", "probable_match"):
            problems.append("The game folder does not match the Analyzer database (Home page).")
        if not mapping:
            problems.append("Confirm at least one replacement on the Matching page.")
        text = [f"<b>{len(mapping)}</b> confirmed replacements"]
        if counts.get("proposed"):
            text.append(f"<br><span style='color:gray'>{counts['proposed']} proposals are not reviewed and will not "
                        "be built.</span>")
        text.append("<br><span style='color:gray'>Audio is stored as uncompressed PCM (about 11.5 MB per stereo "
                    "minute) so no proprietary encoder is needed.</span>")
        if problems:
            text.append("<br><br>" + "<br>".join(f"⚠ {esc(p)}" for p in problems))
        self.summary.setText("".join(text))
        self.build_btn.setEnabled(not problems)
        self.builds = studio.builds()
        self.history.setRowCount(len(self.builds))
        for i, b in enumerate(self.builds):
            out = b["zip"] or b["output_dir"]
            values = [(b["finished_at"] or b["started_at"])[:16].replace("T", " "), b["status"] +
                      (f": {b['error']}" if b.get("error") else ""), str(b["summary"].get("cues", "")),
                      fmt_bytes(b["summary"].get("size_bytes")), str(out or "")]
            for col, value in enumerate(values):
                self.history.setItem(i, col, QTableWidgetItem(value))
        self.history.resizeColumnsToContents()
        self.history.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

    def _settings(self) -> BuildSettings:
        base = self.host.studio.build_settings()
        return replace(base, mod_name=self.name.text().strip() or base.mod_name, author=self.author.text().strip(),
                       version=self.version.text().strip() or "1.0.0", description=self.description.text().strip(),
                       layout=self.layout_box.currentData(), normalize=self.normalize.isChecked(),
                       target_rms_dbfs=self.target.value(), make_zip=self.make_zip.isChecked())

    def build(self) -> None:
        settings = self._settings()

        def done(result) -> None:
            self.refresh()
            self.host.refresh()
            notes = "\n".join(f"• {w}" for w in result.warnings[:8])
            self.host.info("Mod built", f"The mod was built and validated:\n{result.output_dir}"
                           + (f"\n{result.zip_path}" if result.zip_path else "")
                           + (f"\n\nNotes:\n{notes}" if notes else "")
                           + "\n\nInstall it with your mod manager (DMM or CDUMM).")

        self.host.run_job("Building the mod", lambda report, cancelled: self.host.studio.build_mod(
            settings, report, cancelled), on_done=done, on_fail=self.refresh)

    def show_report(self) -> None:
        row = self.history.currentRow()
        build = self.builds[row] if 0 <= row < len(self.builds) else (self.builds[0] if self.builds else None)
        if build and build["output_dir"] and Path(os_path(build["output_dir"])).is_dir():
            report = Path(build["output_dir"]) / REPORT_NAME
            if not is_file(report):                       # mods built before the report was renamed
                report = Path(build["output_dir"]) / "build_report.json"
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(report if is_file(report) else build["output_dir"])))
