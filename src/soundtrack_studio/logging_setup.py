"""Rotating file logs under ``logs/``.

Logs contain what is needed to debug (versions, project, Analyzer DB schema,
counts, errors) but user-identifying details are redacted: the user's home
folder and account name are replaced with placeholders, and music tags are
never logged.
"""

from __future__ import annotations

import faulthandler
import getpass
import logging
import logging.handlers
import platform
import sys
from pathlib import Path
from typing import List, Tuple

from . import __version__
from .app_paths import AppPaths

LOG_FILE = "studio.log"
_fault_file = None


class RedactingFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.replacements: List[Tuple[str, str]] = []
        home = str(Path.home())
        if len(home) > 3:
            self.replacements += [(home, "<home>"), (home.replace("\\", "/"), "<home>")]
        try:
            user = getpass.getuser()
        except Exception:  # pragma: no cover - platform specific
            user = ""
        if user and len(user) >= 3:
            for sep in ("\\", "/"):
                self.replacements.append((f"{sep}{user}{sep}", f"{sep}<user>{sep}"))

    def redact(self, text: str) -> str:
        for old, new in self.replacements:
            text = text.replace(old, new)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        record.msg = self.redact(message)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self.redact(record.exc_text)
        return True


def setup_logging(paths: AppPaths, level: str = "INFO", console: bool = False) -> Path:
    global _fault_file
    paths.logs.mkdir(parents=True, exist_ok=True)
    log_path = paths.logs / LOG_FILE
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    redactor = RedactingFilter()
    file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(redactor)
    root.addHandler(file_handler)
    if console and sys.stderr is not None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        stream.addFilter(redactor)
        root.addHandler(stream)
    if _fault_file is None:
        try:
            _fault_file = open(paths.logs / "crash.log", "a", encoding="utf-8")
            faulthandler.enable(_fault_file)
        except OSError:
            _fault_file = None
    logging.getLogger("soundtrack_studio").info(
        "Crimson Soundtrack Studio %s | Python %s | %s | frozen=%s | root=%s",
        __version__, platform.python_version(), platform.platform(), bool(getattr(sys, "frozen", False)),
        redactor.redact(str(paths.root)),
    )
    return log_path


def shutdown_logging() -> None:
    global _fault_file
    logging.shutdown()
    if _fault_file is not None:
        try:
            faulthandler.disable()
            _fault_file.close()
        except Exception:
            pass
        _fault_file = None

