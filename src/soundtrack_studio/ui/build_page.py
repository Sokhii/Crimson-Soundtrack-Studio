"""Build page: package settings, confirmed replacements, build + validation, history."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from ..app_paths import is_file, os_path
from ..compiler.build import BuildSettings
from ..compiler.validate import OLD_REPORT_NAMES, report_file_name
from .widgets import Card, esc, fmt_bytes

ENCODER_LABELS = {"wwise_vorbis": "Wwise Vorbis — same as the working Nexus music mods (needs Wwise)",
                  "pcm": "Uncompressed PCM — no extra software (not proven to play in game)"}
WWISE_STEPS = ("Wwise is made by Audiokinetic and is free for non-commercial use. It cannot be shipped with the "
               "Studio, so it is installed once with Audiokinetic's own installer:\n\n"
               "1. On the page that opens, download the Audiokinetic Launcher (WwiseLauncher.exe) and run it.\n"
               "2. Sign in (a free Audiokinetic account is needed) and open the 'Wwise' tab.\n"
               "3. Install Wwise {version}.x (it matches the game's soundbanks); the default 'Authoring' package "
               "is enough.\n"
               "4. Come back here and press 'Find Wwise', then 'Test Wwise'.\n\n"
               "Wwise installs into Program Files and keeps its own settings in your user profile; the Studio only "
               "runs it and keeps all converted files in its own folder.")
LOUDNESS_LABELS = {"match": "Match the original music (automatic)",
                   "fixed": "Fixed loudness (manual)",
                   "off": "Keep each track's own level"}
LAYOUT_LABELS = {"crimson_browser": "Mod manager package (manifest.json + files/) — DMM / CDUMM",
                 "package_folders": "Package folders only (0004/…) — alternative layout"}


def loudness_text(summary) -> str:
    if not summary or not summary.get("cues") or summary.get("mode") == "off":
        return ""
    text = (f"\n\nLoudness: {summary['reached_target']} of {summary['cues']} replacements reached their target"
            + (" (the original's loudness)" if summary["mode"] == "match" else "") + ".")
    if summary.get("below_target"):
        text += (f" {summary['below_target']} stay up to {summary['most_below_db']:g} dB quieter so their peaks do "
                 "not distort.")
    if summary.get("raised_to_floor"):
        text += (f" {summary['raised_to_floor']} originals measured very quiet; those were raised to "
                 f"{summary['floor_lufs']:g} LUFS.")
    if summary.get("estimated"):
        text += (f"\n{summary['estimated']} originals only had an older level measurement, so their loudness was "
                 "estimated; 'Analyse game audio' (or Describe music) measures them properly once.")
    return text


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
        self.loudness_box = QComboBox()
        for key, label in LOUDNESS_LABELS.items():
            self.loudness_box.addItem(label, key)
        self.loudness_box.setToolTip(
            "Automatic: each replacement is made as loud as the game's original music for that cue (measured by "
            "'Analyse game audio'); cues whose original was not measured use the target below.\n"
            "Fixed: every replacement is made as loud as the target below.\n"
            "Own level: tracks keep their loudness.\n\n"
            "In every mode a track is never raised so far that its peaks would distort (true peak at most -1 dBTP); "
            "such a track stays a little quieter and the build report says by how much.")
        self.loudness_box.currentIndexChanged.connect(self._loudness_changed)
        self.target = QDoubleSpinBox()
        self.target.setRange(-30.0, -8.0)
        self.target.setDecimals(1)
        self.target.setSuffix(" LUFS")
        self.floor = QDoubleSpinBox()
        self.floor.setRange(0.0, 20.0)
        self.floor.setDecimals(1)
        self.floor.setSingleStep(1.0)
        self.floor.setSuffix(" dB below the target")
        self.floor.setToolTip("Automatic mode: a cue is never made quieter than this far below the loudness target, "
                              "even if the original it replaces measured quieter (for example one quiet layer of a "
                              "layered piece). Peaks are still never pushed into distortion.")
        self.make_zip = QCheckBox("Also create a .zip file")
        self.encoder_box = QComboBox()
        for key, label in ENCODER_LABELS.items():
            self.encoder_box.addItem(label, key)
        self.encoder_box.currentIndexChanged.connect(self._encoder_changed)
        for label, widget in (("Mod name", self.name), ("Author", self.author), ("Version", self.version),
                              ("Description", self.description), ("Layout", self.layout_box),
                              ("Audio format", self.encoder_box), ("Loudness", self.loudness_box),
                              ("Loudness target", self.target), ("Never quieter than", self.floor),
                              ("", self.make_zip)):
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

        wwise_card = Card("Dependencies: Wwise (for the Vorbis format)")
        self.wwise_label = QLabel()
        self.wwise_label.setWordWrap(True)
        self.wwise_label.setTextFormat(Qt.TextFormat.RichText)
        wwise_card.body.addWidget(self.wwise_label)
        wrow = QHBoxLayout()
        self.get_wwise_btn = QPushButton("Get Wwise…")
        self.get_wwise_btn.clicked.connect(self.get_wwise)
        self.find_wwise_btn = QPushButton("Find Wwise")
        self.find_wwise_btn.clicked.connect(self.find_wwise)
        self.choose_wwise_btn = QPushButton("Choose WwiseConsole.exe…")
        self.choose_wwise_btn.clicked.connect(self.choose_wwise)
        self.test_wwise_btn = QPushButton("Test Wwise")
        self.test_wwise_btn.clicked.connect(self.test_wwise)
        for b in (self.get_wwise_btn, self.find_wwise_btn, self.choose_wwise_btn, self.test_wwise_btn):
            wrow.addWidget(b)
        wrow.addStretch(1)
        wwise_card.body.addLayout(wrow)
        layout.addWidget(wwise_card)

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
        self.wwise = {}

    def refresh(self) -> None:
        studio = self.host.studio
        self._refresh_wwise()
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
        self.loudness_box.setCurrentIndex(max(0, self.loudness_box.findData(s.loudness_mode)))
        self.target.setValue(s.target_lufs)
        self.floor.setValue(s.match_floor_db)
        self._loudness_changed()
        self.make_zip.setChecked(s.make_zip)
        self.encoder_box.blockSignals(True)
        self.encoder_box.setCurrentIndex(max(0, self.encoder_box.findData(s.encoder)))
        self.encoder_box.blockSignals(False)
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
        if s.encoder == "wwise_vorbis" and not self.wwise.get("found"):
            problems.append("Install Wwise for the Vorbis format (below), or choose the PCM format.")
        text = [f"<b>{len(mapping)}</b> confirmed replacements"]
        if counts.get("proposed"):
            text.append(f"<br><span style='color:gray'>{counts['proposed']} proposals are not reviewed and will not "
                        "be built.</span>")
        if s.encoder == "wwise_vorbis":
            text.append("<br><span style='color:gray'>Audio is converted to Wwise Vorbis by your Wwise installation, "
                        "the format the game and the working Nexus music mods use (about 1 MB per stereo "
                        "minute).</span>")
        else:
            text.append("<br><span style='color:gray'>Audio is stored as uncompressed PCM (about 11.5 MB per stereo "
                        "minute). This format has not been confirmed to play in the game.</span>")
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
                       layout=self.layout_box.currentData(), loudness_mode=self.loudness_box.currentData(),
                       target_lufs=self.target.value(), match_floor_db=self.floor.value(),
                       make_zip=self.make_zip.isChecked(),
                       encoder=self.encoder_box.currentData())

    # ------------------------------------------------------------------ Wwise
    def _refresh_wwise(self) -> None:
        status = self.wwise = self.host.studio.wwise_status()
        if status["found"]:
            version = status["version"] or "unknown version"
            note = "" if status["matches_game"] else (f"<br><span style='color:#c98a00'>The game's soundbanks were "
                                                      f"made with Wwise {status['recommended']}; another version may "
                                                      "also work but is less certain.</span>")
            self.wwise_label.setText(f"<span style='color:#2e9d52'>✓ Found</span> {esc(version)}<br>"
                                     f"<span style='color:gray'>{esc(status['console'])}</span>{note}")
        else:
            self.wwise_label.setText("Wwise is <b>not installed</b>. It is needed for the Vorbis format, which every "
                                     "working Crimson Desert music mod uses. Press <b>Get Wwise…</b> for the "
                                     f"installer (Wwise {status['recommended']} recommended).")
        self.test_wwise_btn.setEnabled(status["found"])

    def _loudness_changed(self, _index: int = 0) -> None:
        mode = self.loudness_box.currentData()
        self.target.setEnabled(mode != "off")
        self.floor.setEnabled(mode == "match")
        self.target.setToolTip("The loudness every replacement gets." if mode == "fixed" else
                               "Used only for cues whose original music was not measured." if mode == "match" else
                               "Not used: tracks keep their own level.")

    def _encoder_changed(self, _index: int) -> None:
        studio = self.host.studio
        studio.settings.build_encoder = self.encoder_box.currentData()
        studio.settings.save(studio.paths)
        if studio.project is not None:
            self.refresh()

    def get_wwise(self) -> None:
        status = self.host.studio.wwise_status()
        QDesktopServices.openUrl(QUrl(status["download_page"]))
        self.host.info("Get Wwise", WWISE_STEPS.format(version=status["recommended"]))
        self.refresh()

    def find_wwise(self) -> None:
        self.refresh()
        status = self.wwise
        if status["found"]:
            self.host.info("Find Wwise", f"Found Wwise {status['version'] or ''}:\n{status['console']}")
            return
        self.host.info("Find Wwise", "Wwise was not found in the usual places:\n\n" + "\n".join(
            f"• {p}" for p in status["searched"][:12]) + "\n\nIf it is installed somewhere else, press 'Choose "
            "WwiseConsole.exe…' and pick  <your Wwise folder>\\Authoring\\x64\\Release\\bin\\WwiseConsole.exe  "
            "(the Audiokinetic Launcher shows the install folder on the Wwise version's page).")

    def choose_wwise(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose WwiseConsole.exe", "", "WwiseConsole (WwiseConsole.exe);;"
                                              "Programs (*.exe)")
        if path:
            try:
                self.host.studio.set_wwise_console(path)
            except Exception as exc:  # noqa: BLE001 - shown to the user
                self.host.info("Wwise", getattr(exc, "message", str(exc)))
            self.refresh()

    def test_wwise(self) -> None:
        def done(result) -> None:
            if result["ok"]:
                self.host.info("Test Wwise", f"Wwise works: a 2-second test tone became a {result['bytes']:,} byte "
                               f"Wwise Vorbis file ({result['channels']} channels, {result['sample_rate']} Hz).")
            else:
                self.host.info("Test Wwise", "Wwise ran, but the result was not as expected. Details: "
                               "logs\\wwise_check.json and logs\\wwise_console.log.")

        self.host.run_job("Testing Wwise", lambda report, cancelled: self.host.studio.test_wwise(), on_done=done)

    def build(self) -> None:
        settings = self._settings()

        def done(result) -> None:
            self.refresh()
            self.host.refresh()
            notes = "\n".join(f"• {w}" for w in result.warnings[:8])
            self.host.info("Mod built", f"The mod was built and validated:\n{result.output_dir}"
                           + (f"\n{result.zip_path}" if result.zip_path else "")
                           + loudness_text(result.report.get("loudness"))
                           + (f"\n\nNotes:\n{notes}" if notes else "")
                           + "\n\nInstall it with your mod manager (DMM or CDUMM).")

        self.host.run_job("Building the mod", lambda report, cancelled: self.host.studio.build_mod(
            settings, report, cancelled), on_done=done, on_fail=self.refresh)

    def show_report(self) -> None:
        row = self.history.currentRow()
        build = self.builds[row] if 0 <= row < len(self.builds) else (self.builds[0] if self.builds else None)
        if build and build["output_dir"] and Path(os_path(build["output_dir"])).is_dir():
            folder = Path(build["output_dir"])
            candidates = [folder.parent / report_file_name(folder.name)] + [folder / n for n in OLD_REPORT_NAMES]
            report = next((c for c in candidates if is_file(c)), None)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(report or build["output_dir"])))
