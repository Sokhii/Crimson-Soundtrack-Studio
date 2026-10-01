"""Game Data page: read-only browser of the music structures from the Analyzer database."""

from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from ..game_model.model import KIND_LABEL, GameMusicModel, MusicCue, MusicNode
from .semantic_widgets import ProfileEditor, profile_html
from .widgets import esc, fmt_duration

ROLE_ID = Qt.ItemDataRole.UserRole


class NumItem(QTableWidgetItem):
    """Table item that sorts by a numeric key."""

    def __init__(self, text: str, key: float) -> None:
        super().__init__(text)
        self.key = key

    def __lt__(self, other) -> bool:
        return self.key < getattr(other, "key", 0)


HEARD_GROUPS = (("instrumentation", "Instruments"), ("style", "Style"), ("mood", "Mood"), ("atmosphere", "Atmosphere"),
                ("emotion", "Emotion"), ("rhythm", "Rhythm"), ("texture", "Texture"))


def _heard_text(summary: dict) -> str:
    """One line per group: the standout words with their 0-100 scores (higher = clearer standout)."""

    lines = []
    if summary.get("vocals"):
        where = f", singing heard in {summary['vocals_excerpts']} excerpts" if summary.get("vocals_excerpts") else ""
        score = f" {summary['vocals_score']}/100" if summary.get("vocals_score") is not None else ""
        lines.append(f"Vocals: {summary['vocals']}{score}{where}")
    for key, label in HEARD_GROUPS:
        tags = summary.get(key) or {}
        if tags:
            lines.append(f"{label}: " + ", ".join(f"{t} {s}" for t, s in tags.items()))
    if summary.get("calibrated") is False:
        lines.append("(scores are provisional: too little music analysed yet to compare against)")
    return "\n".join(lines)


class GameDataPage(QWidget):
    def __init__(self, host) -> None:
        super().__init__()
        self.host = host
        self.model: Optional[GameMusicModel] = None
        self.cue_by_segment: dict = {}
        self.setObjectName("page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        title = QLabel("Game Data")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        self.summary = QLabel("Import an Analyzer database on the Home page to browse the game's music.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        audio_bar = QHBoxLayout()
        self.audio_status = QLabel("")
        self.audio_status.setWordWrap(True)
        self.audio_status.setToolTip("The game's music is decoded read-only into the program's temp folder, measured "
                                     "(and listened to, when a listening model is on) and deleted again. Only the "
                                     "measurements are kept. The game folder is never changed.")
        audio_bar.addWidget(self.audio_status, 1)
        self.audio_btn = QPushButton("Analyse game audio")
        self.audio_btn.setToolTip("Decode the game's music (read-only) and measure it, so game cues are described "
                                  "from their actual sound, not only from their names.")
        self.audio_btn.clicked.connect(lambda: self.host.analyze_game_audio())
        self.check_btn = QPushButton("Test decoding")
        self.check_btn.setToolTip("Decode a few game music files and report whether it works; nothing is kept.")
        self.check_btn.clicked.connect(lambda: self.host.test_game_audio())
        audio_bar.addWidget(self.audio_btn)
        audio_bar.addWidget(self.check_btn)
        layout.addLayout(audio_bar)
        bar = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by name, ID, state or event…")
        self.filter.textChanged.connect(self.apply_filter)
        bar.addWidget(self.filter, 1)
        self.technical = QCheckBox("Show technical details")
        self.technical.toggled.connect(self.show_current)
        bar.addWidget(self.technical)
        layout.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.tabs = QTabWidget()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Structure", "Type", "Duration", "Tempo", "Details"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.tree.setColumnWidth(0, 300)
        self.tree.currentItemChanged.connect(lambda cur, _prev: self._select(("node", cur.data(0, ROLE_ID)) if cur else None))
        self.tabs.addTab(self.tree, "Music structure")
        self.cues = QTableWidget(0, 7)
        self.cues.setHorizontalHeaderLabels(["Cue (segment)", "Context", "Duration", "Tempo", "Tracks", "Audio", "Playback"])
        self.cues.setSortingEnabled(True)
        self.cues.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.cues.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.cues.horizontalHeader().setStretchLastSection(True)
        self.cues.currentCellChanged.connect(lambda r, *_: self._select(("node", self._row_id(self.cues, r))))
        self.tabs.addTab(self.cues, "Cues")
        self.banks = QTableWidget(0, 5)
        self.banks.setHorizontalHeaderLabels(["Bank", "ID", "Music structures", "Objects", "Embedded audio"])
        self.banks.setSortingEnabled(True)
        self.banks.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.banks.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.banks.horizontalHeader().setStretchLastSection(True)
        self.banks.currentCellChanged.connect(lambda r, *_: self._select(("bank", self._row_id(self.banks, r))))
        self.tabs.addTab(self.banks, "Soundbanks")
        split.addWidget(self.tabs)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        self.details = QTextBrowser()
        self.details.setOpenLinks(False)
        side_layout.addWidget(self.details, 1)
        self.edit_btn = QPushButton("Edit character…")
        self.edit_btn.setEnabled(False)
        self.edit_btn.clicked.connect(self.edit_profile)
        side_layout.addWidget(self.edit_btn)
        split.addWidget(side)
        self.profiles = {}
        self.audio_results = {}
        split.setSizes([640, 420])
        layout.addWidget(split, 1)
        self._current = None

    # ------------------------------------------------------------- loading
    def set_model(self, model: Optional[GameMusicModel]) -> None:
        self.model = model
        self.cue_by_segment = {c.segment_id: c for c in model.cues} if model else {}
        self.profiles = self.host.studio.profiles("cue") if model is not None and self.host.studio.project else {}
        self.refresh_audio_status(model)
        self.tree.clear()
        self.cues.setRowCount(0)
        self.banks.setRowCount(0)
        self.details.clear()
        if model is None:
            self.summary.setText("Import an Analyzer database on the Home page to browse the game's music.")
            return
        s = model.stats
        text = (f"{s['cues']} music cues (segments) in {s['playlists']} playlists and {s['switches']} switches, "
                f"{s['tracks']} tracks using {s['media']} audio files, across {s['music_banks']} of {s['banks']} soundbanks.")
        if model.warnings:
            text += "<br>" + "<br>".join(f"⚠ {esc(w)}" for w in model.warnings)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setText(text)
        self._fill_tree(model)
        self._fill_cues(model)
        self._fill_banks(model)
        self.apply_filter(self.filter.text())

    def refresh_audio_status(self, model: Optional[GameMusicModel]) -> None:
        studio = self.host.studio
        self.audio_results = {}
        if model is None or studio.project is None:
            self.audio_status.setText("")
            self.audio_btn.setEnabled(False)
            self.check_btn.setEnabled(False)
            return
        ok, reason = studio.game_audio_available()
        self.audio_btn.setEnabled(ok)
        self.check_btn.setEnabled(ok)
        if not ok:
            self.audio_status.setText(f"Game audio: not analysed. {reason}")
            return
        try:
            self.audio_results = studio.game_audio_results()
        except Exception as exc:  # noqa: BLE001 - status line only
            self.audio_status.setText(f"Game audio: status unavailable ({getattr(exc, 'message', exc)})")
            return
        done = sum(1 for r in self.audio_results.values() if r.status == "ok")
        failed = sum(1 for r in self.audio_results.values() if r.status == "error")
        heard = sum(1 for r in self.audio_results.values() if r.listening)
        total = len(self.audio_results)
        text = f"Game audio: {done} of {total} music files decoded and measured"
        if failed:
            text += f", {failed} could not be decoded"
        if heard:
            text += f", {heard} listened to"
        self.audio_status.setText(text + ".")

    def _fill_tree(self, model: GameMusicModel) -> None:
        self.tree.setUpdatesEnabled(False)
        transitions = {t for n in model.nodes.values() for t in n.transition_segments}

        def add(parent, oid: int, trail: frozenset, state: str = "") -> None:
            node = model.nodes.get(oid)
            if node is None:
                return
            if oid in transitions:
                state = ", ".join(x for x in (state, "transition") if x)
            item = QTreeWidgetItem([node.label, KIND_LABEL.get(node.kind, node.kind), self._node_duration(node),
                                    f"{node.tempo_bpm:g}" if node.tempo_bpm else "", self._node_summary(node, state)])
            item.setData(0, ROLE_ID, oid)
            if isinstance(parent, QTreeWidget):
                parent.addTopLevelItem(item)
            else:
                parent.addChild(item)
            if oid in trail:  # guard against cyclic data
                return
            for child in node.children:
                add(item, child, trail | {oid}, ", ".join(node.child_states.get(str(child), [])))

        for root in model.roots:
            add(self.tree, root, frozenset())
        if self.tree.topLevelItemCount() <= 3:
            self.tree.expandToDepth(1)
        self.tree.setUpdatesEnabled(True)

    def _fill_cues(self, model: GameMusicModel) -> None:
        self.cues.setSortingEnabled(False)
        self.cues.setRowCount(len(model.cues))
        for row, cue in enumerate(model.cues):
            context = " › ".join(cue.path_labels[:-1])
            if cue.states:
                context += f"  [{', '.join(cue.states)}]"
            if cue.is_transition:
                context = "(transition) " + context
            items = [
                QTableWidgetItem(cue.label), QTableWidgetItem(context),
                NumItem(fmt_duration(cue.duration_ms / 1000 if cue.duration_ms else None), cue.duration_ms or 0),
                NumItem(f"{cue.tempo_bpm:g}" if cue.tempo_bpm else "", cue.tempo_bpm or 0),
                NumItem(str(len(cue.track_ids)), len(cue.track_ids)),
                QTableWidgetItem(self._audio_summary(cue)), QTableWidgetItem(", ".join(cue.streaming)),
            ]
            items[0].setData(ROLE_ID, cue.segment_id)
            for col, item in enumerate(items):
                self.cues.setItem(row, col, item)
        self.cues.setSortingEnabled(True)
        self.cues.resizeColumnsToContents()

    def _fill_banks(self, model: GameMusicModel) -> None:
        banks = sorted(model.banks.values(), key=lambda b: (-b.music_objects, b.path))
        self.banks.setSortingEnabled(False)
        self.banks.setRowCount(len(banks))
        for row, b in enumerate(banks):
            items = [QTableWidgetItem(b.name or b.path), NumItem(str(b.bank_id), b.bank_id),
                     NumItem(str(b.music_objects), b.music_objects), NumItem(str(b.object_count), b.object_count),
                     NumItem(str(b.media_count), b.media_count)]
            items[0].setData(ROLE_ID, b.bank_id)
            for col, item in enumerate(items):
                self.banks.setItem(row, col, item)
        self.banks.setSortingEnabled(True)
        self.banks.resizeColumnsToContents()

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _row_id(table: QTableWidget, row: int):
        item = table.item(row, 0) if row >= 0 else None
        return item.data(ROLE_ID) if item else None

    @staticmethod
    def _node_duration(node: MusicNode) -> str:
        if node.duration_ms:
            return fmt_duration(node.duration_ms / 1000)
        if node.clips and node.clips[0].source_duration_ms:
            return fmt_duration(node.clips[0].source_duration_ms / 1000)
        return ""

    @staticmethod
    def _node_summary(node: MusicNode, state: str) -> str:
        parts = [state] if state else []
        if node.kind == "switch" and node.arguments:
            parts.append("by " + ", ".join(a.get("name") or str(a.get("group_id")) for a in node.arguments))
        if node.kind == "track":
            parts.append(f"{len(node.source_ids)} audio file(s)")
        if node.kind == "segment" and node.markers:
            parts.append(f"{len(node.markers)} markers")
        return " · ".join(parts)

    def _audio_summary(self, cue: MusicCue) -> str:
        parts = [f"{len(cue.source_ids)} file(s)"]
        if cue.codecs:
            parts.append("/".join(cue.codecs))
        if cue.channels:
            parts.append("/".join(f"{c}ch" for c in cue.channels))
        return " · ".join(parts)

    # ----------------------------------------------------------- filtering
    def apply_filter(self, text: str) -> None:
        if self.model is None:
            return
        q = text.strip().lower()
        for row in range(self.cues.rowCount()):
            hay = " ".join(self.cues.item(row, c).text().lower() for c in range(self.cues.columnCount()) if self.cues.item(row, c))
            sid = self.cues.item(row, 0).data(ROLE_ID)
            cue_events = ""
            if q:
                cue = self.cue_by_segment.get(sid)
                cue_events = " ".join(cue.event_names + [str(s) for s in cue.source_ids]).lower() if cue else ""
            self.cues.setRowHidden(row, bool(q) and q not in f"{hay} {sid} {cue_events}")
        for row in range(self.banks.rowCount()):
            hay = " ".join(self.banks.item(row, c).text().lower() for c in range(2))
            self.banks.setRowHidden(row, bool(q) and q not in hay)

        def visit(item: QTreeWidgetItem) -> bool:
            own = not q or q in f"{item.text(0)} {item.text(4)} {item.data(0, ROLE_ID)}".lower()
            child_match = False
            for i in range(item.childCount()):
                child_match = visit(item.child(i)) or child_match
            item.setHidden(not (own or child_match))
            return own or child_match

        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i))

    # -------------------------------------------------------------- details
    def _select(self, key) -> None:
        self._current = key
        self.show_current()

    def show_current(self) -> None:
        key = self._current
        if not key or self.model is None or key[1] is None:
            return
        kind, ident = key
        if kind == "bank":
            b = self.model.banks.get(ident)
            if b:
                self.details.setHtml(f"<h3>{esc(b.name or 'Unnamed bank')}</h3><p>ID {b.bank_id} · Wwise bank version "
                                     f"{b.version}<br>Path: {esc(b.path)}<br>{b.music_objects} music structures · "
                                     f"{b.object_count} objects · {b.media_count} embedded audio files</p>")
            return
        node = self.model.nodes.get(ident)
        if node is None:
            return
        self.edit_btn.setEnabled(str(ident) in self.profiles)
        html = self._node_html(node)
        if node.kind == "segment":
            html += self._audio_html(node)
            html += profile_html(self.profiles.get(str(ident)), "Musical character (inferred)")
        self.details.setHtml(html)

    def _audio_html(self, node: MusicNode) -> str:
        cue = self.cue_by_segment.get(node.object_id)
        if cue is None or not self.audio_results:
            return ""
        measured, heard = self.host.studio.cue_audio(cue, self.audio_results)
        if not measured:
            errors = [self.audio_results[s].error for s in self.host.studio.cue_sources(cue)
                      if s in self.audio_results and self.audio_results[s].status == "error"]
            return (f"<p style='color:#c98a00'>⚠ The audio of this cue could not be decoded: {esc(errors[0])}</p>"
                    if errors else "")
        parts = []
        for label, key, fmt in (("energy", "energy_index", "{:.2f}"), ("brightness", "brightness_index", "{:.2f}"),
                                ("level", "rms_dbfs", "{:.0f} dBFS"), ("tempo", "tempo_bpm", "{:.0f} BPM (estimate)")):
            if measured.get(key) is not None:
                parts.append(f"{label} {fmt.format(measured[key])}")
        out = "<h4>Measured from the game audio</h4><p>" + esc(" · ".join(parts) or "no measurements") + "</p>"
        summary = self.host.studio.heard_summary(heard, game_only=True)
        if summary:
            out += "<h4>Heard by the listening model</h4><p>" + esc(_heard_text(summary)).replace("\n", "<br>") + "</p>"
        return out

    def _node_html(self, node: MusicNode) -> str:
        m = self.model
        assert m is not None
        out = [f"<h3>{esc(node.label)}</h3>",
               f"<p><b>{esc(KIND_LABEL.get(node.kind))}</b> · ID {node.object_id}"
               + (f" · banks: {esc(', '.join(m.banks[b].name or str(b) if b in m.banks else str(b) for b in node.banks))}" if node.banks else "")
               + "</p>"]
        if node.parse_status not in ("parsed", "shallow"):
            out.append(f"<p style='color:#c98a00'>⚠ Only partially decoded by the Analyzer ({esc(node.parse_status)}).</p>")
        cue = self.cue_by_segment.get(node.object_id)
        if cue:
            out.append(f"<p>Context: {esc(' › '.join(cue.path_labels))}</p>")
            if cue.states:
                out.append(f"<p>Plays when: {esc(', '.join(cue.states))}</p>")
            if cue.event_names:
                out.append(f"<p>Started by events: {esc(', '.join(cue.event_names))}</p>")
            if cue.is_transition:
                out.append("<p>Used as a <b>transition</b> segment between other music.</p>")
            if cue.parent_count > 1:
                out.append(f"<p>Reused by {cue.parent_count} containers.</p>")
        if node.duration_ms:
            out.append(f"<p>Duration: {fmt_duration(node.duration_ms / 1000)}")
            if node.tempo_bpm:
                out.append(f" · {node.tempo_bpm:g} BPM {esc(node.time_signature or '')}")
            out.append("</p>")
        if node.markers:
            out.append("<p>Markers: " + esc(", ".join(f"{mk.get('name') or mk.get('id')} @ {fmt_duration((mk.get('position_ms') or 0) / 1000)}"
                                                     for mk in node.markers)) + "</p>")
        if node.kind == "switch":
            groups = ", ".join(a.get("name") or str(a.get("group_id")) for a in node.arguments)
            out.append(f"<p>Chooses music by: {esc(groups or 'unknown')}</p><ul>")
            for child in node.children:
                label = m.nodes[child].label if child in m.nodes else str(child)
                out.append(f"<li>{esc(label)}: {esc(', '.join(node.child_states.get(str(child), [])) or 'default')}</li>")
            out.append("</ul>")
        if node.kind == "playlist" and node.playlist:
            out.append(f"<p>Playlist with {len(node.playlist)} entries.</p>")
        media = m.media_for_node(node.object_id)
        if media:
            out.append("<h4>Audio files</h4><table cellspacing=4>"
                       "<tr><th align=left>ID</th><th align=left>Length</th><th align=left>Format</th><th align=left>Playback</th></tr>")
            for info in media[:200]:
                fmt = " · ".join(x for x in [info.codec, f"{info.channels}ch" if info.channels else "",
                                             f"{info.sample_rate // 1000 if info.sample_rate else '?'} kHz"] if x)
                if not info.found:
                    fmt = "<span style='color:#c98a00'>not found in the Analyzer database</span>"
                else:
                    fmt = esc(fmt)
                out.append(f"<tr><td>{info.source_id}</td><td>{fmt_duration(info.duration_s)}</td><td>{fmt}</td>"
                           f"<td>{esc(', '.join(info.streaming + info.containers))}</td></tr>")
            out.append("</table>")
        if self.technical.isChecked():
            out.append("<h4>Technical details (from the Analyzer)</h4><pre style='font-size:11px'>"
                       + esc(json.dumps(node.fields, indent=2, ensure_ascii=False)[:20000]) + "</pre>")
        return "".join(out)


    def edit_profile(self) -> None:
        key = self._current
        if not key or key[0] != "node":
            return
        effective = self.profiles.get(str(key[1]))
        if effective is None:
            return
        node = self.model.nodes[key[1]]
        dialog = ProfileEditor(effective.llm or effective.rules, effective.profile, node.label, self)
        if dialog.exec():
            store = self.host.studio.semantic_store()
            if dialog.reset_requested:
                store.clear_override("cue", str(key[1]))
            else:
                store.set_override("cue", str(key[1]), dialog.values())
            self.profiles = self.host.studio.profiles("cue")
            self.show_current()
