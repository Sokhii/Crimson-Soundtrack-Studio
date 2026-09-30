"""vgmstream wrapper: decodes one Wwise ``.wem`` file into a temporary PCM ``.wav``.

vgmstream (ISC licence, https://github.com/vgmstream/vgmstream) is bundled in
``runtime/vgmstream/``. It only reads the input file and writes the output file
it is given; both live in the application's ``temp/`` folder.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from ..app_paths import AppPaths, os_path
from ..errors import StudioError


class DecodeError(StudioError):
    title = "Game audio could not be decoded"


def executable_name() -> str:
    return "vgmstream-cli.exe" if os.name == "nt" else "vgmstream-cli"


def find_vgmstream(paths: AppPaths, override: str = "") -> Optional[Path]:
    for candidate in (override, os.environ.get("CSS_VGMSTREAM", "")):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    exe = executable_name()
    for candidate in (paths.runtime / "vgmstream" / exe, paths.runtime / exe):
        if candidate.is_file():
            return candidate
    return next(iter(sorted(paths.runtime.glob(f"**/{exe}"))), None) if paths.runtime.is_dir() else None


def _run(exe: Path, args: list, timeout: float) -> subprocess.CompletedProcess:
    creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    try:
        return subprocess.run([os_path(exe)] + args, capture_output=True, timeout=timeout,
                              cwd=os_path(exe.parent), creationflags=creationflags)
    except subprocess.TimeoutExpired as exc:
        raise DecodeError("Decoding took too long and was stopped.", details=" ".join(str(a) for a in args)) from exc
    except OSError as exc:
        raise DecodeError("The audio decoder (vgmstream) could not be started.",
                          hint="Reinstall the portable build; the decoder lives in the 'runtime\\vgmstream' folder.",
                          details=f"{exe}: {exc}") from exc


def version(exe: Path) -> str:
    result = _run(exe, ["-V"], 20)
    try:
        return str(json.loads(result.stdout.decode("utf-8", "replace")).get("version", "")) or "unknown"
    except ValueError:
        text = (result.stdout or result.stderr).decode("utf-8", "replace")
        return text.strip().splitlines()[0] if text.strip() else "unknown"


def decode(exe: Path, wem: Path, wav: Path, timeout: float = 300) -> None:
    """Decode ``wem`` once, ignoring loop points (``-i``), into ``wav`` (16-bit PCM)."""

    wav.unlink(missing_ok=True)
    result = _run(exe, ["-i", "-o", os_path(wav), os_path(wem)], timeout)
    if result.returncode != 0 or not wav.is_file() or wav.stat().st_size <= 44:
        message = (result.stderr or result.stdout).decode("utf-8", "replace").strip()
        wav.unlink(missing_ok=True)
        raise DecodeError("vgmstream could not decode this game audio file.",
                          details=f"{wem.name}: exit {result.returncode}: {message[:400]}")
