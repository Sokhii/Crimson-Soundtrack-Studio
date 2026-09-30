"""Matching page: proposals with explanations, confidence and the user's overrides."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser,
                               QVBoxLayout, QWidget)

from ..matching.engine import MatchSettings, confidence_label
from .semantic_widgets import profile_html
from .widgets import esc, fmt_duration

STATUS_COLOR = {"accepted": "#2e9d52", "chosen": "#2e9d52", "proposed": "#1f6fb2", "rejected": "#b3261e",
                "keep original": "#777777", "unmatched": "#c98a00"}
FIT_LABELS = {"auto": "Automatic", "trim": "Trim (fade out)", "loop": "Loop to fill", "pad": "Play once, then silence"}


class TrackPicker(QDialog):
    def __init__(self, rows: List[Dict[str, Any]], cue_label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Choose a track for {cue_label}")
        self.resize(820, 560)
        self.rows = rows
        self.chosen: Optional[int] = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("All usable tracks, best thematic fit first. Your choice always wins over the AI."))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter…")
        layout.addWidget(self.filter)
        self.table = QTableWidget(len(rows), 4)
        self.table.setHorizontalHeaderLabels(["Track", "Length", "Fit", "Why"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for i, r in enumerate(rows):
            cand = r["candidate"]
            values = [r["track"].label, fmt_duration(r["track"].duration_s),
                      f"{cand.score:.0%}" if cand else "too short", "; ".join(cand.reasons[:2]) if cand else ""]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, r["track"].id)
                self.table.setItem(i, col, item)
        self.table.resizeColumnToContents(0)
        self.table.doubleClicked.connect(lambda _i: self.accept_choice())
        self.filter.textChanged.connect(self._filter)
        layout.addWidget(self.table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept_choice)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _filter(self, text: str) -> None:
        q = text.lower()
        for i in range(self.table.rowCount()):
            self.table.setRowHidden(i, bool(q) and q not in self.table.item(i, 0).text().lower())

    def accept_choice(self) -> None:
        item = self.table.item(self.table.currentRow(), 0)
        if item:
            self.chosen = item.data(Qt.ItemDataRole.UserRole)
            self.accept()


class MatchingPage(QWidget):
    def __init__(self, host) -> None:
        super().__init__()
        self.host = host
        self.setObjectName("page")
        self.rows: List[Dict[str, Any]] = []
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.cue_profiles = {}
        self.track_profiles = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        title = QLabel("Matching")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        intro = QLabel("The Studio proposes a replacement for each piece of game music by comparing musical character "
                       "(mood, atmosphere, energy, instrumentation, style), not gameplay categories. Nothing is built "
                       "until you accept or choose a track. Your decisions are never changed by re-running matching.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        bar = QHBoxLayout()
        self.describe_btn = QPushButton("1. Describe music")
        self.describe_btn.clicked.connect(host.describe_music)
        self.match_btn = QPushButton("2. Find matches")
        self.match_btn.clicked.connect(self.find_matches)
        self.use_ai = QCheckBox("Use local AI to judge the best candidates")
        self.allow_reuse = QCheckBox("Allow a track for several cues")
        self.allow_reuse.setChecked(True)
        self.include_short = QCheckBox("Include short/transition cues")
        self.include_short.toggled.connect(lambda _v: self.refresh())
        self.accept_all_btn = QPushButton("Accept all high-confidence")
        self.accept_all_btn.clicked.connect(self.accept_all)
        for w in (self.describe_btn, self.match_btn, self.use_ai, self.allow_reuse, self.include_short):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.accept_all_btn)
        layout.addLayout(bar)
        filters = QHBoxLayout()
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All", "Not reviewed", "Accepted / chosen", "Rejected", "Keep original", "No match"])
        self.status_filter.currentIndexChanged.connect(lambda _i: self.apply_filter())
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by cue, state, event or track…")
        self.search.textChanged.connect(lambda _t: self.apply_filter())
        self.summary = QLabel()
        filters.addWidget(self.status_filter)
        filters.addWidget(self.search, 1)
        filters.addWidget(self.summary)
        layout.addLayout(filters)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Game cue", "Length", "Replacement", "Confidence", "Status"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.currentCellChanged.connect(lambda r, *_: self.show_row(r))
        split.addWidget(self.table)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        self.details = QTextBrowser()
        side_layout.addWidget(self.details, 1)
        actions = QHBoxLayout()
        self.accept_btn = QPushButton("Accept")
        self.accept_btn.clicked.connect(self.accept_current)
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.clicked.connect(self.reject_current)
        self.choose_btn = QPushButton("Choose another…")
        self.choose_btn.clicked.connect(self.choose_current)
        self.keep_btn = QPushButton("Keep original")
        self.keep_btn.clicked.connect(self.keep_current)
        self.clear_btn = QPushButton("Clear decision")
        self.clear_btn.clicked.connect(self.clear_current)
        for b in (self.accept_btn, self.reject_btn, self.choose_btn, self.keep_btn, self.clear_btn):
            actions.addWidget(b)
        side_layout.addLayout(actions)
        fit_row = QHBoxLayout()
        fit_row.addWidget(QLabel("Fit:"))
        self.fit_mode = QComboBox()
        for key, label in FIT_LABELS.items():
            self.fit_mode.addItem(label, key)
        fit_row.addWidget(self.fit_mode)
        fit_row.addWidget(QLabel("Start at (s):"))
        self.start_offset = QDoubleSpinBox()
        self.start_offset.setRange(0, 36000)
        self.start_offset.setDecimals(1)
        fit_row.addWidget(self.start_offset)
        self.fit_btn = QPushButton("Apply")
        self.fit_btn.clicked.connect(self.apply_fit)
        fit_row.addWidget(self.fit_btn)
        fit_row.addStretch(1)
        side_layout.addLayout(fit_row)
        split.addWidget(side)
        split.setSizes([700, 520])
        layout.addWidget(split, 1)
        self.current_key: Optional[str] = None

    # ---------------------------------------------------------------- data
    def refresh(self) -> None:
        studio = self.host.studio
        has_data = studio.project is not None and studio.project.active_analyzer() is not None
        for w in (self.describe_btn, self.match_btn, self.accept_all_btn):
            w.setEnabled(has_data)
        self.use_ai.setEnabled(studio.active_model() is not None)
        if not self.use_ai.isEnabled():
            self.use_ai.setChecked(False)
        if not has_data:
            self.rows = []
            self.table.setRowCount(0)
            self.summary.setText("Import an Analyzer database and scan your music first.")
            return
        try:
            self.rows = studio.matching_rows(include_short=self.include_short.isChecked())
        except Exception as exc:  # noqa: BLE001 - e.g. missing snapshot; shown once
            self.host.error(exc)
            self.rows = []
        self.tracks = {t["id"]: t for t in studio.library_tracks()}
        self.cue_profiles = studio.profiles("cue")
        self.track_profiles = studio.profiles("track")
        self.table.setRowCount(len(self.rows))
        for i, row in enumerate(self.rows):
            cue = row["cue"]
            track_id = self._shown_track(row)
            props = row["proposals"]
            conf = ""
            if props and track_id == props[0].track_id and row["status"] in ("proposed", "accepted"):
                conf = f"{confidence_label(props[0].confidence)} ({props[0].confidence:.0%})"
            context = " › ".join(cue.path_labels[:-1][-2:])
            if cue.states:
                context += f"  [{', '.join(cue.states[:2])}]"
            values = [f"{cue.label}   {context}", fmt_duration(cue.duration_ms / 1000 if cue.duration_ms else None),
                      self._track_label(track_id) if track_id else (row["skipped_reason"] or "—"), conf, row["status"]]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, row["key"])
                if col == 4:
                    item.setData(Qt.ItemDataRole.ForegroundRole, _qcolor(STATUS_COLOR.get(row["status"], "#000")))
                self.table.setItem(i, col, item)
        counts: Dict[str, int] = {}
        for row in self.rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        self.summary.setText(" · ".join(f"{v} {k}" for k, v in sorted(counts.items())))
        self.apply_filter()
        self._restore_selection()

    def _shown_track(self, row) -> Optional[int]:
        decision = row["decision"]
        if decision and decision.action in ("accept", "manual"):
            return decision.track_id
        if decision and decision.action == "keep_original":
            return None
        return row["proposals"][0].track_id if row["proposals"] else None

    def _track_label(self, track_id: Optional[int]) -> str:
        t = self.tracks.get(track_id)
        if not t:
            return f"(track {track_id} no longer in the library)"
        return " – ".join(x for x in (t.get("artist"), t.get("title") or t["rel_path"].rsplit("/", 1)[-1]) if x)

    def apply_filter(self) -> None:
        wanted = self.status_filter.currentText()
        groups = {"Not reviewed": ("proposed",), "Accepted / chosen": ("accepted", "chosen"), "Rejected": ("rejected",),
                  "Keep original": ("keep original",), "No match": ("unmatched",)}
        q = self.search.text().strip().lower()
        for i, row in enumerate(self.rows):
            visible = wanted == "All" or row["status"] in groups.get(wanted, ())
            if visible and q:
                cue = row["cue"]
                hay = " ".join([*cue.path_labels, *cue.states, *cue.event_names, self.table.item(i, 2).text()]).lower()
                visible = q in hay or q in row["key"]
            self.table.setRowHidden(i, not visible)

    def _restore_selection(self) -> None:
        if self.current_key is None:
            return
        for i, row in enumerate(self.rows):
            if row["key"] == self.current_key:
                self.table.selectRow(i)
                self.show_row(i)
                return

    # -------------------------------------------------------------- details
    def show_row(self, index: int) -> None:
        if not (0 <= index < len(self.rows)):
            return
        row = self.rows[index]
        self.current_key = row["key"]
        cue = row["cue"]
        decision = row["decision"]
        props = row["proposals"]
        shown = self._shown_track(row)
        out = [f"<h3>{esc(cue.label)}</h3><p style='color:gray'>{esc(' › '.join(cue.path_labels))}</p>"]
        facts = [f"{fmt_duration(cue.duration_ms / 1000 if cue.duration_ms else None)}"]
        if cue.tempo_bpm:
            facts.append(f"{cue.tempo_bpm:g} BPM")
        if cue.states:
            facts.append("plays when " + ", ".join(cue.states))
        if cue.is_transition:
            facts.append("transition segment")
        out.append(f"<p>{esc(' · '.join(facts))}</p>")
        status = row["status"]
        out.append(f"<p><b>Status:</b> <span style='color:{STATUS_COLOR.get(status, '#000')}'>{esc(status)}</span>")
        if decision and decision.rejected:
            out.append(f" · rejected: {esc(', '.join(self._track_label(t) for t in decision.rejected))}")
        out.append("</p>")
        out.append(profile_html(self.cue_profiles.get(row["key"]), "Original game music (inferred character)"))
        if shown:
            prop = next((p for p in props if p.track_id == shown), None)
            out.append(f"<h3>Replacement: {esc(self._track_label(shown))}</h3>")
            if decision and decision.action == "manual":
                out.append("<p>Chosen by you.</p>")
            if prop:
                out.append(f"<p>Thematic fit {prop.score:.0%} · confidence <b>{confidence_label(prop.confidence)}</b> "
                           f"({prop.confidence:.0%})</p>")
                if prop.ai_reason:
                    out.append(f"<p><b>Local AI:</b> {esc(prop.ai_reason)} ({prop.ai_fit:.0%} fit)</p>")
                out.append("<ul>" + "".join(f"<li>{esc(r)}</li>" for r in prop.reasons) + "</ul>")
                if prop.warnings:
                    out.append("<p style='color:#c98a00'>" + "<br>".join(f"⚠ {esc(w)}" for w in prop.warnings) + "</p>")
            out.append(profile_html(self.track_profiles.get(str(shown)), "Replacement character"))
        elif row["skipped_reason"]:
            out.append(f"<p style='color:#c98a00'>{esc(row['skipped_reason'])}</p>")
        others = [p for p in props if p.track_id != shown]
        if others:
            out.append("<h4>Other candidates</h4><table>")
            for p in others:
                out.append(f"<tr><td>{esc(self._track_label(p.track_id))}</td><td style='padding-left:8px'>"
                           f"{p.score:.0%}</td><td style='padding-left:8px;color:gray'>{esc('; '.join(p.reasons[:1]))}</td></tr>")
            out.append("</table>")
        self.details.setHtml("".join(out))
        has_track = shown is not None
        self.accept_btn.setEnabled(has_track and not (decision and decision.action in ("accept", "manual")))
        self.reject_btn.setEnabled(has_track)
        self.clear_btn.setEnabled(decision is not None)
        confirmed = decision is not None and decision.action in ("accept", "manual")
        for w in (self.fit_mode, self.start_offset, self.fit_btn):
            w.setEnabled(confirmed)
        if decision:
            self.fit_mode.setCurrentIndex(max(0, self.fit_mode.findData(decision.fit_mode)))
            self.start_offset.setValue(decision.start_offset_s)

    # ------------------------------------------------------------- actions
    def _row(self) -> Optional[Dict[str, Any]]:
        return next((r for r in self.rows if r["key"] == self.current_key), None)

    def _after(self) -> None:
        self.refresh()
        self.host.pages["home"].refresh()

    def accept_current(self) -> None:
        row = self._row()
        track = self._shown_track(row) if row else None
        if track:
            self.host.studio.match_store().accept(row["key"], track)
            self._after()

    def reject_current(self) -> None:
        row = self._row()
        track = self._shown_track(row) if row else None
        if track:
            self.host.studio.match_store().reject(row["key"], track)
            self._after()

    def choose_current(self) -> None:
        row = self._row()
        if not row:
            return
        picker = TrackPicker(self.host.studio.rank_tracks_for_cue(row["key"]), row["cue"].label, self)
        if picker.exec() and picker.chosen is not None:
            self.host.studio.match_store().choose(row["key"], picker.chosen)
            self._after()

    def keep_current(self) -> None:
        if self.current_key:
            self.host.studio.match_store().keep_original(self.current_key)
            self._after()

    def clear_current(self) -> None:
        if self.current_key:
            self.host.studio.match_store().clear(self.current_key)
            self._after()

    def apply_fit(self) -> None:
        if self.current_key:
            try:
                self.host.studio.match_store().set_fit(self.current_key, self.fit_mode.currentData(),
                                                       self.start_offset.value())
            except ValueError as exc:
                self.host.info("Fit", str(exc))
            self._after()

    def accept_all(self) -> None:
        if self.host.confirm("Accept proposals", "Accept every proposal with high confidence? You can still change "
                             "each one afterwards."):
            count = self.host.studio.match_store().accept_all(min_confidence=0.6)
            self._after()
            self.host.info("Accept proposals", f"{count} proposals accepted.")

    def find_matches(self) -> None:
        settings = MatchSettings(include_short_cues=self.include_short.isChecked(),
                                 allow_reuse=self.allow_reuse.isChecked(), use_ai=self.use_ai.isChecked())

        def done(stats) -> None:
            self._after()
            self.host.statusBar().showMessage(
                f"Matching finished: {stats['proposed']} cues have a proposal, {stats['unmatched']} have none.", 15000)

        self.host.run_job("Finding matches", lambda report, cancelled: self.host.studio.find_matches(
            settings, report, cancelled), on_done=done)


def _qcolor(text: str):
    from PySide6.QtGui import QBrush, QColor

    return QBrush(QColor(text))
