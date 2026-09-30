"""Small shared widgets and helpers for the GUI (presentation only)."""

from __future__ import annotations

import html
import logging
import traceback
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFileDialog, QFrame, QLabel, QMessageBox, QSizePolicy, QVBoxLayout, QWidget)

from ..errors import OperationCancelled, StudioError

log = logging.getLogger(__name__)

STATE_ICON = {"ok": ("✔", "#2e9d52"), "warning": ("⚠", "#c98a00"), "missing": ("○", "#888888"),
              "unavailable": ("–", "#aaaaaa")}


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}" if hours else f"{minutes}:{sec:02d}"


def fmt_bytes(size: Optional[int]) -> str:
    if size is None:
        return ""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return str(size)


class Card(QFrame):
    """A titled panel used on the Home page."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        header = QLabel(title)
        header.setObjectName("cardTitle")
        layout.addWidget(header)
        self.body = QVBoxLayout()
        self.body.setSpacing(6)
        layout.addLayout(self.body)


class DropZone(QLabel):
    """Accepts a dropped file/folder or a click to browse."""

    dropped = Signal(str)
    clicked = Signal()

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("hover", True)
            self.style().polish(self)

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self.setProperty("hover", False)
        self.style().polish(self)

    def dropEvent(self, event) -> None:  # noqa: N802
        self.setProperty("hover", False)
        self.style().polish(self)
        urls = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.dropped.emit(urls[0])

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class Dialogs:
    """File dialogs. The Qt dialog is the portable default: the native Windows dialog
    records recently used folders in the registry."""

    def __init__(self, native: bool = False) -> None:
        self.native = native

    def _options(self) -> QFileDialog.Option:
        return QFileDialog.Option(0) if self.native else QFileDialog.Option.DontUseNativeDialog

    def folder(self, parent: QWidget, title: str, start: str = "") -> Optional[str]:
        path = QFileDialog.getExistingDirectory(parent, title, start, self._options() | QFileDialog.Option.ShowDirsOnly)
        return path or None

    def open_file(self, parent: QWidget, title: str, filters: str, start: str = "") -> Optional[str]:
        path, _ = QFileDialog.getOpenFileName(parent, title, start, filters, options=self._options())
        return path or None


def show_error(parent: Optional[QWidget], exc: BaseException, log_path: Optional[Path] = None) -> None:
    if isinstance(exc, OperationCancelled):
        return
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    if isinstance(exc, StudioError):
        box.setWindowTitle(exc.title)
        box.setText(exc.message)
        if exc.hint:
            box.setInformativeText(exc.hint)
        details = exc.details
        log.warning("%s: %s | %s", exc.title, exc.message, exc.details)
    else:
        box.setWindowTitle("Unexpected error")
        box.setText("Something unexpected went wrong. The operation was stopped; your files were not changed.")
        box.setInformativeText("Technical details were written to the log file." +
                               (f"\n{log_path}" if log_path else ""))
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log.error("Unexpected error in GUI operation:\n%s", details)
    if details:
        box.setDetailedText(details)
    box.exec()


def bullet_html(items: List[str]) -> str:
    return "".join(f"<div>• {esc(i)}</div>" for i in items)


STYLE = """
QWidget#page { background: palette(base); }
QLabel#pageTitle { font-size: 20px; font-weight: 600; }
QLabel#cardTitle { font-size: 14px; font-weight: 600; }
QFrame#card { border: 1px solid palette(mid); border-radius: 6px; background: palette(window); }
QLabel#dropZone { border: 2px dashed palette(mid); border-radius: 6px; padding: 10px; color: palette(text); }
QLabel#dropZone[hover="true"] { border-color: #b3261e; background: rgba(179,38,30,0.08); }
QLabel#muted { color: palette(dark); }
QListWidget#nav { font-size: 14px; }
QListWidget#nav::item { padding: 10px 8px; }
QListWidget#nav::item:selected { background: #7a1f1a; color: white; }
"""
