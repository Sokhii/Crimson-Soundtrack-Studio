"""Display and editing of semantic profiles (the user's edits always win)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QSlider, QVBoxLayout, QWidget)

from ..semantic.profile import CATEGORIES, SCALES, SemanticProfile
from ..semantic.store import EffectiveProfile
from .widgets import esc

SOURCE_TEXT = {"rules": "rule-based estimate", "llm": "local AI", "user": "your edits"}


def profile_html(effective: Optional[EffectiveProfile], title: str = "Character") -> str:
    if effective is None:
        return (f"<h4>{esc(title)}</h4><p style='color:gray'>Not described yet. Use <i>Describe music</i> on the "
                "Matching or AI Model page.</p>")
    p = effective.profile
    source = SOURCE_TEXT.get(p.source, p.source)
    if p.source == "llm" and p.model_id:
        source += f" ({esc(p.model_id)})"
    out = [f"<h4>{esc(title)} <span style='color:gray;font-weight:normal;font-size:11px'>· {source} · confidence "
           f"{p.confidence:.0%}</span></h4>"]
    if p.summary:
        out.append(f"<p><i>{esc(p.summary)}</i></p>")
    out.append("<table>")
    for key in CATEGORIES:
        values = getattr(p, key)
        if values:
            out.append(f"<tr><td style='color:gray;padding-right:8px'>{key.title()}</td><td>{esc(', '.join(values))}</td></tr>")
    if p.themes:
        out.append(f"<tr><td style='color:gray;padding-right:8px'>Themes</td><td>{esc(', '.join(p.themes))}</td></tr>")
    for key in SCALES:
        value = getattr(p, key)
        if value is not None:
            bar = "█" * int(round(value * 10)) + "░" * (10 - int(round(value * 10)))
            out.append(f"<tr><td style='color:gray;padding-right:8px'>{key.title()}</td>"
                       f"<td><span style='font-family:monospace'>{bar}</span> {value:.2f}</td></tr>")
    if p.vocal_presence is not None:
        out.append(f"<tr><td style='color:gray;padding-right:8px'>Vocals</td><td>{'yes' if p.vocal_presence else 'no'}</td></tr>")
    out.append("</table>")
    if effective.llm_error:
        out.append(f"<p style='color:#c98a00;font-size:11px'>The local AI could not describe this: "
                   f"{esc(effective.llm_error[:160])}</p>")
    if effective.evidence and p.source == "rules":
        out.append("<p style='color:gray;font-size:11px'>Evidence: " + esc("; ".join(effective.evidence[:6])) + "</p>")
    return "".join(out)


class ProfileEditor(QDialog):
    """Edit a profile. Only fields that differ from the underlying (AI/rules) profile become the override."""

    def __init__(self, base: SemanticProfile, current: SemanticProfile, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.base = base
        self.setWindowTitle(f"Edit character - {title}")
        self.resize(760, 620)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Your edits take precedence over the AI and rule-based descriptions. "
                                "Use 'Reset' to go back to the automatic description."))
        grid = QGridLayout()
        self.lists: Dict[str, QListWidget] = {}
        for i, (key, vocab) in enumerate(CATEGORIES.items()):
            box = QGroupBox(key.title())
            box_layout = QVBoxLayout(box)
            widget = QListWidget()
            for word in vocab:
                item = QListWidgetItem(word)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if word in getattr(current, key) else Qt.CheckState.Unchecked)
                widget.addItem(item)
            box_layout.addWidget(widget)
            self.lists[key] = widget
            grid.addWidget(box, 0, i)
        layout.addLayout(grid, 1)
        form = QFormLayout()
        self.sliders: Dict[str, QSlider] = {}
        for key in SCALES:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            value = getattr(current, key)
            slider.setValue(int(round((value if value is not None else 0.5) * 100)))
            self.sliders[key] = slider
            form.addRow(key.title(), slider)
        self.vocals = QComboBox()
        self.vocals.addItems(["unknown", "instrumental", "has vocals"])
        self.vocals.setCurrentIndex(0 if current.vocal_presence is None else (2 if current.vocal_presence else 1))
        form.addRow("Vocals", self.vocals)
        self.themes = QLineEdit(", ".join(current.themes))
        form.addRow("Themes (comma separated)", self.themes)
        self.summary = QLineEdit(current.summary)
        form.addRow("Summary", self.summary)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
                                   | QDialogButtonBox.StandardButton.Reset)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.reset_requested = False
        buttons.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self._reset)
        layout.addWidget(buttons)

    def _reset(self) -> None:
        self.reset_requested = True
        self.accept()

    def values(self) -> Dict[str, Any]:
        edited: Dict[str, Any] = {}
        for key, widget in self.lists.items():
            chosen = [widget.item(i).text() for i in range(widget.count())
                      if widget.item(i).checkState() == Qt.CheckState.Checked]
            if chosen != getattr(self.base, key):
                edited[key] = chosen
        for key, slider in self.sliders.items():
            value = slider.value() / 100
            base_value = getattr(self.base, key)
            if base_value is None or abs(base_value - value) > 0.005:
                edited[key] = value
        vocals = {0: None, 1: False, 2: True}[self.vocals.currentIndex()]
        if vocals != self.base.vocal_presence:
            edited["vocal_presence"] = vocals
        themes = [t.strip() for t in self.themes.text().split(",") if t.strip()]
        if themes != self.base.themes:
            edited["themes"] = themes
        if self.summary.text().strip() != self.base.summary:
            edited["summary"] = self.summary.text().strip()
        return edited
