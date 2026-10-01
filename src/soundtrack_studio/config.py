"""Application settings stored as JSON in ``data/config/settings.json``.

No registry, no QSettings. Unknown keys from newer versions are preserved so
an older build does not destroy a newer build's settings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List

from .app_paths import AppPaths

log = logging.getLogger(__name__)

SETTINGS_VERSION = 1


@dataclass
class Settings:
    settings_version: int = SETTINGS_VERSION
    last_project: str = ""                    # stored path (app:projects/<slug>)
    recent_projects: List[str] = field(default_factory=list)
    # Native Windows file dialogs record recently used folders in the registry
    # (ComDlg32 MRU lists). The Qt dialog does not, so it is the portable default.
    use_native_dialogs: bool = False
    log_level: str = "INFO"
    ai_model_id: str = ""                     # selected local model ("" = rule-based descriptions only)
    ai_gpu_layers: str = "auto"               # "auto", "0" (CPU only) or a layer count
    ai_threads: int = 0                       # 0 = llama.cpp default
    llama_server_path: str = ""               # empty = bundled runtime/llama/
    analyze_game_audio: bool = True           # decode the game's music read-only (temp/) and measure it
    vgmstream_path: str = ""                  # empty = bundled runtime/vgmstream/
    listening_model_id: str = ""              # optional listening model (CLAP); "" = off
    listening_device: str = "auto"            # auto (GPU when available) | cpu
    matching_mode: str = "standout"           # "standout" = compare standout scores | "legacy" = tag matching
    approve_levels: List[str] = field(default_factory=lambda: ["high"])   # confidence levels the Approve button accepts
    build_encoder: str = "wwise_vorbis"       # audio format of built mods: wwise_vorbis (needs Wwise) | pcm
    wwise_console_path: str = ""              # WwiseConsole.exe; empty = find it (WWISEROOT, Program Files)
    wwise_conversion: str = "Vorbis Quality High"   # Wwise conversion setting used for the music
    max_uses_per_track: int = 0               # most cues one track may be proposed for (0 = automatic)
    window_geometry: str = ""                 # hex-encoded QByteArray
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, paths: AppPaths) -> "Settings":
        file = paths.settings_file
        if not file.is_file():
            return cls()
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("settings root is not an object")
        except (OSError, ValueError) as exc:
            backup = file.with_name("settings.corrupt.json")
            log.warning("settings file unreadable (%s); starting with defaults, old file kept as %s", exc, backup.name)
            try:
                file.replace(backup)
            except OSError:
                pass
            return cls()
        known = {f.name for f in fields(cls)}
        settings = cls(**{k: v for k, v in data.items() if k in known and k != "extra"})
        settings.extra = dict(data.get("extra") or {})
        for key, value in data.items():
            if key not in known:
                settings.extra.setdefault(f"_unknown.{key}", value)
        return settings

    def save(self, paths: AppPaths) -> None:
        paths.config.mkdir(parents=True, exist_ok=True)
        tmp = paths.settings_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(paths.settings_file)

    def remember_project(self, stored_path: str, limit: int = 10) -> None:
        self.last_project = stored_path
        self.recent_projects = [stored_path] + [p for p in self.recent_projects if p != stored_path]
        del self.recent_projects[limit:]
