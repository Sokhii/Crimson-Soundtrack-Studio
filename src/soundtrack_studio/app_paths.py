"""Executable-relative portable directory layout.

All persistent data created by the Studio lives below the *application root*:
the directory that contains ``CrimsonSoundtrackStudio.exe`` in a frozen build.
The root is derived from the executable's real location - never from the
current working directory - so launching through a shortcut, a mod manager or
a command prompt somewhere else behaves the same.

Nothing is written to %APPDATA%, %LOCALAPPDATA%, %PROGRAMDATA%, the registry,
Documents or any other user/system-managed location.

Paths the Studio records (in settings and projects) are stored with
:meth:`AppPaths.to_stored`: locations inside the root are stored *relative*
(``app:projects/foo``) so moving or copying the whole folder keeps them valid;
user-selected external locations (game folder, music folder) stay absolute.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ENV_HOME_OVERRIDE = "CRIMSON_STUDIO_HOME"
STORED_APP_PREFIX = "app:"
MODEL_TIERS = ("low", "medium", "high", "custom")


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def detect_app_root() -> Path:
    """Return the portable application root.

    * ``CRIMSON_STUDIO_HOME`` set: that folder (tests and developer overrides).
    * Frozen (PyInstaller) build: the folder holding the executable.
    * Source checkout: ``<repo>/dev_home`` so development runs never litter the
      repository root (the folder is git-ignored).
    """

    override = os.environ.get(ENV_HOME_OVERRIDE)
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2] / "dev_home"


@dataclass(frozen=True)
class AppPaths:
    root: Path

    # ------------------------------------------------------------ layout
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def config(self) -> Path:
        return self.data / "config"

    @property
    def settings_file(self) -> Path:
        return self.config / "settings.json"

    @property
    def cache(self) -> Path:
        return self.data / "cache"

    @property
    def databases(self) -> Path:
        """Imported (snapshotted) Analyzer databases, one folder per content hash."""

        return self.data / "databases"

    @property
    def models(self) -> Path:
        return self.root / "models"

    def model_tier_dir(self, tier: str) -> Path:
        if tier not in MODEL_TIERS:
            raise ValueError(f"unknown model tier {tier!r}")
        return self.models / tier

    @property
    def projects(self) -> Path:
        return self.root / "projects"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def temp(self) -> Path:
        return self.root / "temp"

    @property
    def runtime(self) -> Path:
        """Bundled, read-only runtime components (llama.cpp in a later phase). Not created at startup."""

        return self.root / "runtime"

    def all_dirs(self) -> List[Path]:
        return [
            self.data, self.config, self.cache, self.databases,
            self.models, *(self.model_tier_dir(t) for t in MODEL_TIERS),
            self.projects, self.output, self.logs, self.temp,
        ]

    def ensure(self) -> "AppPaths":
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)
        return self

    # ----------------------------------------------------------- helpers
    def is_inside(self, path: Path | str) -> bool:
        try:
            Path(path).resolve().relative_to(self.root.resolve())
            return True
        except (ValueError, OSError):
            return False

    def to_stored(self, path: Path | str | None) -> str:
        """Serialise a path for settings/project files (relative when inside the root)."""

        if path is None or str(path) == "":
            return ""
        p = Path(path)
        try:
            rel = p.resolve().relative_to(self.root.resolve())
            return STORED_APP_PREFIX + rel.as_posix()
        except (ValueError, OSError):
            return str(p)

    def from_stored(self, stored: Optional[str]) -> Optional[Path]:
        if not stored:
            return None
        if stored.startswith(STORED_APP_PREFIX):
            return self.root / Path(stored[len(STORED_APP_PREFIX):])
        return Path(stored)

    def clean_temp(self, max_age_hours: float = 24.0) -> int:
        """Remove leftovers in ``temp/`` older than ``max_age_hours`` (age-based so a second
        running instance keeps its current files). Returns the number of entries removed."""

        import shutil
        import time

        if not self.temp.is_dir():
            return 0
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        for entry in self.temp.iterdir():
            try:
                if entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
                    removed += 1
            except OSError:
                pass
        return removed

    def check_writable(self) -> Optional[str]:
        """Return a user-facing message when the portable folder cannot be written."""

        probe = self.temp / ".write_probe"
        try:
            self.temp.mkdir(parents=True, exist_ok=True)
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            return (
                f"Crimson Soundtrack Studio cannot write to its own folder:\n{self.root}\n\n"
                "The Studio is portable and keeps all of its settings, projects, models and logs next to "
                "the program. Move the whole folder somewhere you can write to (for example a folder on a "
                f"games drive) instead of Program Files.\n\nTechnical detail: {exc}"
            )
        return None


def os_path(path: Path | str, *, windows: Optional[bool] = None, force: bool = False) -> str:
    """Return a path string safe for the OS file APIs.

    On Windows, absolute paths near or beyond MAX_PATH (260) get the ``\\\\?\\``
    extended-length prefix so deeply nested music folders still open. ``force``
    prefixes short paths too - used for the base of directory walks, whose
    children can grow past the limit. Other platforms return the path unchanged.
    """

    windows = (os.name == "nt") if windows is None else windows
    text = str(path)
    if not windows or text.startswith("\\\\?\\") or (len(text) < 240 and not force):
        return text
    text = os.path.abspath(text) if os.name == "nt" else text
    text = text.replace("/", "\\")
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text[2:]
    return "\\\\?\\" + text


def long_path(path: Path | str) -> Path:
    """``path`` that Windows opens even beyond 260 characters (extended-length prefix); unchanged elsewhere."""

    return Path(os_path(path))


def is_file(path: Path | str) -> bool:
    """``Path.is_file()`` for user files: also true for paths beyond Windows' 260-character limit."""

    try:
        return Path(os_path(path)).is_file()
    except OSError:
        return False


_current: Optional[AppPaths] = None


def get_paths() -> AppPaths:
    global _current
    if _current is None:
        _current = AppPaths(detect_app_root())
    return _current


def set_paths(paths: AppPaths) -> None:
    global _current
    _current = paths
