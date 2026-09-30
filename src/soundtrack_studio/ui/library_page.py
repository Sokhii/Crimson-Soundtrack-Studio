"""Music Library page: scanned tracks with their deterministic metadata and measurements."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QSplitter, QTableView,
                               QTextBrowser, QVBoxLayout, QWidget)

from ..library.features import METHODS
from .semantic_widgets import ProfileEditor, profile_html
from .widgets import esc, fmt_bytes, fmt_duration

COLUMNS = ["Title", "Status", "Artist", "Album", "Length", "Format", "Tempo (est.)", "Energy", "Brightness", "File"]


def _title(t: Dict[str, Any]) -> str:
    return t.get("title") or t["rel_path"].rsplit("/", 1)[-1]


def _format(t: Dict[str, Any]) -> str:
    if not t.get("sample_rate"):
        return t.get("codec") or ""
    bits = f"/{t['bit_depth']}-bit" if t.get("bit_depth") else ""
    return f"{t.get('codec') or ''} {t['sample_rate'] / 1000:g} kHz{bits} · {t.get('channels')}ch"


class TrackModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[Dict[str, Any]] = []

    def set_rows(self, rows: List[Dict[str, Any]]) -> None:
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        t = self.rows[index.row()]
        f = t.get("features") or {}
        col = index.column()
        if role == Qt.ItemDataRole.UserRole:  # sort key
            return [_title(t).lower(), t["status"], (t.get("artist") or "").lower(), (t.get("album") or "").lower(),
                    t.get("duration_s") or 0, t.get("sample_rate") or 0, f.get("tempo_bpm") or 0,
                    f.get("energy_index") or 0, f.get("brightness_index") or 0, t["rel_path"].lower()][col]
        if role == Qt.ItemDataRole.DisplayRole:
            status = {"ok": "OK", "error": "⚠ unreadable", "missing": "missing"}.get(t["status"], t["status"])
            if t["status"] == "ok" and t.get("duplicate_of"):
                status = "duplicate"
            return [_title(t), status, t.get("artist") or "", t.get("album") or "", fmt_duration(t.get("duration_s")),
                    _format(t), f"{f['tempo_bpm']:g}" if f.get("tempo_bpm") else "",
                    f"{f['energy_index']:.2f}" if f.get("energy_index") is not None else "",
                    f"{f['brightness_index']:.2f}" if f.get("brightness_index") is not None else "",
                    t["rel_path"]][col]
        if role == Qt.ItemDataRole.ToolTipRole and t.get("error"):
            return t["error"]
        return None


class LibraryPage(QWidget):
    def __init__(self, host) -> None:
        super().__init__()
        self.host = host
        self.setObjectName("page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        title = QLabel("Music Library")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        bar = QHBoxLayout()
        self.path = QLabel("No music folder selected.")
        bar.addWidget(self.path, 1)
        choose = QPushButton("Choose folder…")
        choose.clicked.connect(lambda: host.pages["home"].choose_library())
        bar.addWidget(choose)
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(host.scan_library)
        bar.addWidget(self.scan_btn)
        layout.addLayout(bar)
        self.summary = QLabel("")
        layout.addWidget(self.summary)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by title, artist, album, genre or file name…")
        layout.addWidget(self.filter)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.model = TrackModel()
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)
        self.filter.textChanged.connect(self.proxy.setFilterFixedString)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.selectionModel().currentRowChanged.connect(self.show_track)
        split.addWidget(self.table)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        self.details = QTextBrowser()
        side_layout.addWidget(self.details, 1)
        self.edit_btn = QPushButton("Edit character…")
        self.edit_btn.setEnabled(False)
        self.edit_btn.clicked.connect(self.edit_profile)
        side_layout.addWidget(self.edit_btn)
        split.addWidget(side)
        split.setSizes([760, 380])
        self.profiles = {}
        self.heard = {}
        self.current_track = None
        layout.addWidget(split, 1)

    def refresh(self) -> None:
        studio = self.host.studio
        if studio.project is None:
            self.model.set_rows([])
            self.path.setText("No project open.")
            self.scan_btn.setEnabled(False)
            return
        path = studio.project.get("music_library_path") or ""
        self.path.setText(path or "No music folder selected.")
        self.scan_btn.setEnabled(bool(path))
        rows = studio.library_tracks()
        self.profiles = studio.profiles("track")
        try:
            self.heard = studio.track_listening()
        except Exception:  # noqa: BLE001 - optional information
            self.heard = {}
        self.model.set_rows(rows)
        self.table.resizeColumnsToContents()
        counts = studio.library_counts()
        self.summary.setText(f"{counts.get('ok', 0)} tracks · {counts.get('error', 0)} unreadable · "
                             f"{counts.get('missing', 0)} missing · {counts.get('duplicates', 0)} duplicates"
                             if rows else "Scan a music folder to see your tracks here.")

    def show_track(self, current: QModelIndex, _previous: Optional[QModelIndex] = None) -> None:
        if not current.isValid():
            return
        t = self.model.rows[self.proxy.mapToSource(current).row()]
        self.current_track = t
        self.edit_btn.setEnabled(str(t["id"]) in self.profiles)
        f = t.get("features") or {}
        out = [f"<h3>{esc(_title(t))}</h3>", f"<p style='color:gray'>{esc(t['root'])}/{esc(t['rel_path'])}</p>"]
        if t["status"] == "error":
            out.append(f"<p style='color:#b3261e'>This file could not be read: {esc(t.get('error'))}</p>")
        elif t["status"] == "missing":
            out.append("<p style='color:#c98a00'>This file was not found during the last scan.</p>")
        if t.get("duplicate_of"):
            out.append("<p>Same audio as another file in the library (duplicate).</p>")
        tags = [("Artist", t.get("artist")), ("Album", t.get("album")), ("Album artist", t.get("album_artist")),
                ("Composer", t.get("composer")), ("Genre", t.get("genre")), ("Year", t.get("year")),
                ("Track", t.get("track_number")), ("Disc", t.get("disc_number"))]
        out.append("<h4>File</h4><table>")
        for label, value in [("Length", fmt_duration(t.get("duration_s"))), ("Format", _format(t)),
                             ("Size", fmt_bytes(t.get("size")))] + tags:
            if value not in (None, ""):
                out.append(f"<tr><td style='color:gray;padding-right:10px'>{label}</td><td>{esc(value)}</td></tr>")
        out.append("</table>")
        if t["status"] == "ok":
            out.append(profile_html(self.profiles.get(str(t["id"]))))
            summary = self.host.studio.heard_summary(self.heard.get(t["id"])) if self.heard.get(t["id"]) else None
            if summary:
                from .game_page import _heard_text

                out.append("<h4>Heard by the listening model</h4><p>" + esc(_heard_text(summary)) + "</p>"
                           "<p style='font-size:11px;color:gray'>From listening to "
                           f"{self.heard[t['id']].get('excerpts', 0)} ten-second excerpts of the audio.</p>")
        if f:
            rows = [
                ("Tempo", f"{f['tempo_bpm']:g} BPM (confidence {f.get('tempo_confidence', 0):.2f})" if f.get("tempo_bpm")
                 else "no steady beat detected", "tempo_bpm"),
                ("Energy", _num(f.get("energy_index")), "energy_index"),
                ("Brightness", _num(f.get("brightness_index")), "brightness_index"),
                ("Average level", _unit(f.get("rms_dbfs"), "dBFS"), "rms_dbfs"),
                ("Peak level", _unit(f.get("peak_dbfs"), "dBFS"), "peak_dbfs"),
                ("Crest factor", _unit(f.get("crest_db"), "dB"), "crest_db"),
                ("Level spread", _unit(f.get("level_spread_db"), "dB"), "level_spread_db"),
                ("Spectral centroid", _unit(f.get("spectral_centroid_hz"), "Hz"), "spectral_centroid_hz"),
                ("Stereo width", _num(f.get("stereo_width")), "stereo_width"),
                ("Onsets / s", _num(f.get("onset_rate")), "onset_rate"),
            ]
            out.append("<h4>Measurements</h4><table>")
            for label, value, key in rows:
                how = METHODS.get(key, "")
                out.append(f"<tr><td style='color:gray;padding-right:10px'>{label}</td><td>{esc(value)}</td>"
                           f"<td style='color:gray;padding-left:10px;font-size:11px'>{esc(how)}</td></tr>")
            out.append("</table>")
            band = f.get("band_energy") or {}
            if band:
                out.append(f"<p style='font-size:11px;color:gray'>Energy split: low {band.get('low', 0):.0%} · "
                           f"mid {band.get('mid', 0):.0%} · high {band.get('high', 0):.0%}</p>")
            if f.get("notes"):
                out.append("<p style='font-size:11px;color:gray'>" + "<br>".join(esc(n) for n in f["notes"]) + "</p>")
            out.append("<p style='font-size:11px;color:gray'>These are measured facts from the audio signal. "
                       "'Estimate' and 'heuristic' values are approximate. Musical key is not guessed; vocals are only judged by the optional listening model.</p>")
        self.details.setHtml("".join(out))


    def edit_profile(self) -> None:
        t = self.current_track
        effective = self.profiles.get(str(t["id"])) if t else None
        if effective is None:
            return
        base = effective.llm or effective.rules
        dialog = ProfileEditor(base, effective.profile, _title(t), self)
        if dialog.exec():
            store = self.host.studio.semantic_store()
            if dialog.reset_requested:
                store.clear_override("track", str(t["id"]))
            else:
                store.set_override("track", str(t["id"]), dialog.values())
            self.profiles = self.host.studio.profiles("track")
            self.show_track(self.table.currentIndex())


def _num(value) -> str:
    return "" if value is None else f"{value:.2f}"


def _unit(value, unit: str) -> str:
    return "" if value is None else f"{value:g} {unit}"

