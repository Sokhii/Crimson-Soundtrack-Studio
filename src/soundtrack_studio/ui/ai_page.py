"""AI Model page: model tiers, downloads, verification, selection and music description."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QRadioButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..ai.hardware import recommend_tier
from .widgets import Card, esc

FIT_TEXT = {"gpu": "fits in GPU memory", "cpu": "runs on CPU (slower)", "no": "too large for this PC",
            "unknown": "unknown"}
STATUS_TEXT = {"verified": "downloaded, verified", "available": "available", "not_downloaded": "not downloaded",
               "partial": "partly downloaded (resume)", "missing": "file missing", "downloading": "downloading…"}


class AIPage(QWidget):
    def __init__(self, host) -> None:
        super().__init__()
        self.host = host
        self.setObjectName("page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        title = QLabel("AI Model")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        intro = QLabel("The Studio describes the character of your music and the game's music with a local AI model "
                       "that runs on this computer (no internet needed after the download, nothing to install). "
                       "Without a model, simpler rule-based descriptions are used. Models are stored in this "
                       "program's own 'models' folder.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        top = QHBoxLayout()
        self.hw_card = Card("This computer")
        self.hw_label = QLabel()
        self.hw_label.setWordWrap(True)
        self.hw_label.setTextFormat(Qt.TextFormat.RichText)
        self.hw_card.body.addWidget(self.hw_label)
        top.addWidget(self.hw_card, 1)
        self.active_card = Card("Selected model")
        self.active_label = QLabel()
        self.active_label.setWordWrap(True)
        self.active_label.setTextFormat(Qt.TextFormat.RichText)
        self.active_card.body.addWidget(self.active_label)
        row = QHBoxLayout()
        self.none_btn = QPushButton("Use no model (rule-based only)")
        self.none_btn.clicked.connect(lambda: self._select(""))
        self.describe_btn = QPushButton("Describe music now")
        self.describe_btn.clicked.connect(host.describe_music)
        self.redescribe_btn = QPushButton("Re-describe…")
        self.redescribe_btn.setToolTip("Have the AI write the descriptions again from scratch, ignoring saved answers "
                                       "(the game's music, your library, or both).")
        self.redescribe_btn.clicked.connect(host.redescribe_music)
        row.addWidget(self.none_btn)
        row.addWidget(self.describe_btn)
        row.addWidget(self.redescribe_btn)
        row.addStretch(1)
        self.active_card.body.addLayout(row)
        top.addWidget(self.active_card, 1)
        layout.addLayout(top)

        tiers = QHBoxLayout()
        self.tier_group = QButtonGroup(self)
        for i, key in enumerate(("low", "medium", "high", "custom")):
            button = QRadioButton(key.title())
            button.setProperty("tier", key)
            self.tier_group.addButton(button, i)
            tiers.addWidget(button)
        tiers.addStretch(1)
        self.tier_info = QLabel()
        self.tier_info.setWordWrap(True)
        layout.addLayout(tiers)
        layout.addWidget(self.tier_info)
        self.tier_group.buttonClicked.connect(lambda _b: self.fill_table())

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Model", "Download", "Memory needed", "On this PC", "Status", "Licence"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self._download)
        self.use_btn = QPushButton("Use this model")
        self.use_btn.clicked.connect(lambda: self._select(self._current_id()))
        self.test_btn = QPushButton("Test")
        self.test_btn.clicked.connect(self._test)
        self.verify_btn = QPushButton("Verify file")
        self.verify_btn.clicked.connect(self._verify)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete)
        self.custom_btn = QPushButton("Add custom GGUF model…")
        self.custom_btn.clicked.connect(self._add_custom)
        for b in (self.download_btn, self.use_btn, self.test_btn, self.verify_btn, self.delete_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.custom_btn)
        layout.addLayout(buttons)
        self.rows = []

        # ------------------------------------------------ optional listening model, beside the description models
        self.listen_card = Card("Listening model (optional)")
        self.listen_intro = QLabel(
            "The description models above read tags, names and measurements; they cannot hear the music. A listening "
            "model <b>listens to the audio itself</b> - your tracks and the game's music - and recognises instruments, "
            "vocals, mood and style, and how much two pieces sound alike. Its findings are given to the description "
            "model (or the rule-based descriptions) and to matching. It works alongside them and replaces nothing.")
        self.listen_intro.setWordWrap(True)
        self.listen_intro.setTextFormat(Qt.TextFormat.RichText)
        self.listen_card.body.addWidget(self.listen_intro)
        self.listen_status = QLabel()
        self.listen_status.setWordWrap(True)
        self.listen_status.setTextFormat(Qt.TextFormat.RichText)
        self.listen_card.body.addWidget(self.listen_status)
        lrow = QHBoxLayout()
        self.listen_download = QPushButton("Download")
        self.listen_download.clicked.connect(self._listen_download)
        self.listen_toggle = QPushButton("Turn on")
        self.listen_toggle.clicked.connect(self._listen_toggle)
        self.listen_test = QPushButton("Test")
        self.listen_test.clicked.connect(self._listen_test)
        self.listen_delete = QPushButton("Delete")
        self.listen_delete.clicked.connect(self._listen_delete)
        self.listen_device = QComboBox()
        self.listen_device.addItem("Graphics card when available", "auto")
        self.listen_device.addItem("CPU only", "cpu")
        self.listen_device.setToolTip("Where the listening model runs. The graphics card (DirectX 12: AMD, NVIDIA, "
                                      "Intel) is much faster; the CPU always works.")
        self.listen_device.currentIndexChanged.connect(self._listen_device_changed)
        for b in (self.listen_download, self.listen_toggle, self.listen_test, self.listen_delete):
            lrow.addWidget(b)
        lrow.addStretch(1)
        lrow.addWidget(QLabel("Runs on:"))
        lrow.addWidget(self.listen_device)
        self.listen_card.body.addLayout(lrow)
        layout.addWidget(self.listen_card)

    # ---------------------------------------------------------------- data
    def refresh(self) -> None:
        studio = self.host.studio
        info = studio.hardware()
        gpu = ", ".join(f"{esc(g.name)} ({g.vram_gb:g} GB)" if g.vram_gb else esc(g.name) for g in info.gpus) or "not detected"
        runtime = studio.runtime_path()
        recommended = recommend_tier(info)
        self.hw_label.setText(
            f"Memory: {info.ram_total_gb or '?'} GB · CPU threads: {info.cpu_threads}<br>Graphics: {gpu}<br>"
            f"Recommended tier: <b>{recommended.title()}</b><br>"
            + ("AI runtime: ready" if runtime else
               "<span style='color:#b3261e'>AI runtime (llama.cpp) not found in the 'runtime\\llama' folder.</span>"))
        if self.tier_group.checkedButton() is None:
            for b in self.tier_group.buttons():
                b.setChecked(b.property("tier") == recommended)
        model = studio.active_model()
        if model:
            self.active_label.setText(f"<b>{esc(model.display_name)}</b> ({model.tier.title()} tier)<br>"
                                      "Used for describing music and explaining matches.")
        elif studio.settings.ai_model_id:
            self.active_label.setText("The selected model is not downloaded yet.")
        else:
            self.active_label.setText("None: rule-based descriptions from tags, names and measurements.")
        self.describe_btn.setEnabled(studio.project is not None)
        self.redescribe_btn.setEnabled(studio.project is not None and model is not None)
        self.fill_table()
        self.fill_listening()

    def _listening_entry(self):
        entries = self.host.studio.listening_models()
        return entries[0] if entries else None

    def fill_listening(self) -> None:
        e = self._listening_entry()
        if e is None:
            self.listen_card.setVisible(False)
            return
        m = e["model"]
        if e["installed"]:
            state = "<b>On</b>: used when describing music." if e["active"] else "Downloaded, currently <b>off</b>."
            test = e.get("last_test") or {}
            if test:
                state += f" Last test: {'OK' if test.get('ok') else 'failed'} on {esc(str(test.get('provider', '')))}."
        elif e["partial"]:
            state = "Partly downloaded (resume)."
        else:
            state = f"Not downloaded (about {m.approximate_size_mb} MB)."
        self.listen_status.setText(f"<b>{esc(m.display_name)}</b> · licence {esc(m.license)} · {state}")
        self.listen_download.setVisible(not e["installed"])
        self.listen_download.setText("Resume download" if e["partial"] else "Download")
        self.listen_toggle.setEnabled(e["installed"])
        self.listen_toggle.setText("Turn off" if e["active"] else "Turn on")
        self.listen_test.setEnabled(e["installed"])
        self.listen_delete.setEnabled(e["installed"] or e["partial"])
        device = self.host.studio.settings.listening_device
        self.listen_device.blockSignals(True)
        self.listen_device.setCurrentIndex(max(0, self.listen_device.findData(device)))
        self.listen_device.blockSignals(False)

    def _listen_download(self) -> None:
        e = self._listening_entry()
        if e is None:
            return
        m = e["model"]
        text = (f"Download {m.display_name} (about {m.approximate_size_mb} MB) from Hugging Face into the program's "
                f"models folder?\n\nLicence: {m.license}\n{m.license_url}")
        if not self.host.confirm("Download listening model", text):
            return
        self.host.run_job(f"Downloading {m.display_name}",
                          lambda report, cancelled: self.host.studio.download_listening_model(m.id, report, cancelled),
                          on_done=lambda _r: self.refresh(), on_fail=lambda: self.refresh())

    def _listen_toggle(self) -> None:
        e = self._listening_entry()
        if e is None:
            return
        self.host.studio.select_listening_model("" if e["active"] else e["model"].id)
        self.refresh()
        self.host.refresh()

    def _listen_test(self) -> None:
        e = self._listening_entry()
        if e is None:
            return

        def done(result) -> None:
            self.refresh()
            self.host.info("Listening model test",
                           ("The listening model loaded and listened to test sounds correctly." if result["ok"] else
                            "The listening model loaded, but its output was not usable.")
                           + f"\n\nRuns on: {result['provider']}\nTime: {result['seconds']} s")

        self.host.run_job("Loading and testing the listening model",
                          lambda report, cancelled: (report("Loading the listening model", 0, 0),
                                                     self.host.studio.test_listening_model(e["model"].id))[1],
                          on_done=done, on_fail=lambda: self.refresh())

    def _listen_delete(self) -> None:
        e = self._listening_entry()
        if e and self.host.confirm("Delete listening model", f"Delete {e['model'].display_name} from the models "
                                   "folder? What it heard so far stays cached; you can download it again later."):
            self.host.studio.delete_listening_model(e["model"].id)
            self.refresh()
            self.host.refresh()

    def _listen_device_changed(self) -> None:
        self.host.studio.settings.listening_device = self.listen_device.currentData()
        self.host.studio.release_listening_model()
        self.host.studio.settings.save(self.host.studio.paths)

    def _tier(self) -> str:
        button = self.tier_group.checkedButton()
        return button.property("tier") if button else "medium"

    def fill_table(self) -> None:
        studio = self.host.studio
        tier = self._tier()
        catalog = studio.catalog()
        info = catalog.tiers.get(tier)
        if info:
            ram = f" · {info.recommended_ram_gb:g} GB system RAM" if info.recommended_ram_gb else ""
            self.tier_info.setText(f"{info.best_for} Recommended: {info.recommended_vram}{ram}.")
        self.rows = [r for r in studio.model_rows() if r["model"].tier == tier]
        self.table.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows):
            m = r["model"]
            name = m.display_name + ("  ✔ selected" if r["selected"] else "")
            if catalog.tiers.get(tier) and catalog.tiers[tier].default_model == m.id:
                name += "  (default)"
            status = STATUS_TEXT.get(r["status"], r["status"])
            if r["inference_ok"] is True:
                status += ", tested"
            values = [name, f"{m.approximate_size_gb:g} GB" if m.approximate_size_gb else "",
                      f"~{m.approximate_ram_gb:g} GB RAM / {m.recommended_vram_gb:g} GB VRAM" if m.recommended_vram_gb
                      else "", FIT_TEXT.get(r["fits"], r["fits"]), status, m.license]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, m.id)
                if col == 5 and m.license_url:
                    item.setToolTip(m.license_url)
                self.table.setItem(i, col, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._update_buttons()

    def _current(self):
        row = self.table.currentRow()
        return self.rows[row] if 0 <= row < len(self.rows) else None

    def _current_id(self) -> str:
        r = self._current()
        return r["model"].id if r else ""

    def _update_buttons(self) -> None:
        r = self._current()
        present = bool(r and r["present"])
        custom = bool(r and r["model"].tier == "custom")
        self.download_btn.setEnabled(bool(r) and not present and not custom)
        self.download_btn.setText("Resume download" if r and r["status"] == "partial" else "Download")
        self.use_btn.setEnabled(present)
        self.test_btn.setEnabled(present and self.host.studio.runtime_path() is not None)
        self.verify_btn.setEnabled(present)
        self.delete_btn.setEnabled(bool(r) and (present or custom or r["status"] == "partial"))
        self.delete_btn.setText("Remove from list" if custom else "Delete")

    # ------------------------------------------------------------- actions
    def _download(self) -> None:
        r = self._current()
        if not r:
            return
        m = r["model"]
        text = (f"Download {m.display_name} (about {m.approximate_size_gb:g} GB) from Hugging Face into the "
                f"program's models folder?\n\nLicence: {m.license}\n{m.license_url}")
        if not self.host.confirm("Download model", text):
            return
        self.host.run_job(f"Downloading {m.display_name}",
                          lambda report, cancelled: self.host.studio.download_model(m.id, report, cancelled),
                          on_done=lambda _r: self.refresh(), on_fail=lambda: self.refresh())

    def _verify(self) -> None:
        model_id = self._current_id()
        self.host.run_job("Verifying the model file",
                          lambda report, cancelled: self.host.studio.verify_model(model_id, cancelled),
                          on_done=lambda _r: self.refresh())

    def _delete(self) -> None:
        r = self._current()
        if not r:
            return
        custom = r["model"].tier == "custom"
        text = ("Remove this custom model from the list? The file itself is not deleted." if custom else
                f"Delete {r['model'].display_name} from the models folder? You can download it again later.")
        if self.host.confirm("Remove model" if custom else "Delete model", text):
            try:
                self.host.studio.delete_model(r["model"].id)
            except Exception as exc:  # noqa: BLE001 - shown to the user
                self.host.error(exc)
            self.refresh()
            self.host.refresh()

    def _select(self, model_id: str) -> None:
        try:
            self.host.studio.select_model(model_id)
        except Exception as exc:  # noqa: BLE001
            self.host.error(exc)
        self.refresh()
        self.host.refresh()

    def _test(self) -> None:
        model_id = self._current_id()

        def done(result) -> None:
            self.refresh()
            self.host.info("Model test", ("The model loaded and answered correctly." if result["ok"] else
                                          "The model loaded, but its answer was not usable. It may still work for "
                                          "descriptions; a different model is recommended.")
                           + f"\n\nTime: {result['seconds']} s")

        self.host.run_job("Loading and testing the model",
                          lambda report, cancelled: (report("Loading the model (this can take a minute)", 0, 0),
                                                     self.host.studio.test_model(model_id))[1],
                          on_done=done, on_fail=lambda: self.refresh())

    def _add_custom(self) -> None:
        path = self.host.dialogs.open_file(self, "Select a GGUF model file", "GGUF models (*.gguf);;All files (*)")
        if not path:
            return
        try:
            model = self.host.studio.add_custom_model(Path(path))
        except Exception as exc:  # noqa: BLE001
            self.host.error(exc)
            return
        for b in self.tier_group.buttons():
            b.setChecked(b.property("tier") == "custom")
        self._select(model.id)
