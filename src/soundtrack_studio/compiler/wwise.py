"""Wwise Vorbis encoding through the user's own Wwise installation.

Every community music mod that is known to play in the game (Crimson Tamriel V1/V2, Way To Valhalla) ships Wwise
Vorbis ``.wem`` files made by Wwise and keeps the soundbanks' Vorbis codec and prefetch layout (see
docs/research/modding_format.md). Wwise is proprietary (Audiokinetic; free for non-commercial use) and cannot be
redistributed, so the Studio neither bundles nor reimplements it: the user installs it with the Audiokinetic
Launcher and the Studio runs its command-line tool, ``WwiseConsole.exe``:

    WwiseConsole.exe create-new-project <project.wproj>
    WwiseConsole.exe convert-external-source <project.wproj> --source-file <list.wsources> --output <folder>

The conversion project lives in ``data/wwise/`` and every file it converts and writes is inside the application
folder. Wwise itself is installed outside it by its own installer and may keep settings in the user's profile;
that is the one part of the workflow the Studio cannot keep portable.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from xml.sax.saxutils import quoteattr

import numpy as np

from ..app_paths import AppPaths, os_path
from .archive import CompileError
from . import wem

log = logging.getLogger(__name__)

DOWNLOAD_PAGE = "https://www.audiokinetic.com/en/download/"
RECOMMENDED_VERSION = "2023.1"          # the game's soundbanks are version 150, written by Wwise 2023.1
DEFAULT_CONVERSION = "Vorbis Quality High"
CONSOLE_REL = Path("Authoring") / "x64" / "Release" / "bin" / "WwiseConsole.exe"
PROJECT_NAME = "CssConvert"


class WwiseError(CompileError):
    title = "Wwise problem"


def _candidates() -> Iterable[Path]:
    root = os.environ.get("WWISEROOT", "")
    if root:
        yield Path(root) / CONSOLE_REL
    for base in (os.environ.get("ProgramFiles(x86)", ""), os.environ.get("ProgramFiles", ""),
                 r"C:\Program Files (x86)", r"C:\Program Files"):
        if not base:
            continue
        folder = Path(base) / "Audiokinetic"
        try:
            installs = sorted(folder.glob("Wwise*"), reverse=True) if folder.is_dir() else []
        except OSError:
            installs = []
        installs.sort(key=lambda p: RECOMMENDED_VERSION not in p.name)   # the game's version first, then newest
        for install in installs:
            yield install / CONSOLE_REL
    found = shutil.which("WwiseConsole") or shutil.which("WwiseConsole.exe")
    if found:
        yield Path(found)


def find_console(override: str = "") -> Optional[Path]:
    """WwiseConsole.exe: the chosen file, ``CSS_WWISE_CONSOLE`` (tests), ``%WWISEROOT%``, Program Files, PATH."""

    for candidate in (override, os.environ.get("CSS_WWISE_CONSOLE", "")):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    for candidate in _candidates():
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def install_name(console: Path) -> str:
    """'Wwise2023.1.4.8496' from ...\\Audiokinetic\\Wwise2023.1.4.8496\\Authoring\\...; '' when unknown."""

    for part in reversed(console.parts):
        if part.lower().startswith("wwise") and any(ch.isdigit() for ch in part):
            return part
    return ""


def _plain(path: Path) -> str:
    """A normal absolute path for Wwise (the build folder may carry the \\\\?\\ long-path prefix, which Wwise's
    own path handling is not known to accept); the build's Wwise paths stay well under 260 characters."""

    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        return "\\\\" + text[8:]
    return text[4:] if text.startswith("\\\\?\\") else text


class WwiseEncoder:
    def __init__(self, paths: AppPaths, console: Path, conversion: str = DEFAULT_CONVERSION,
                 timeout_per_file: float = 120.0) -> None:
        self.paths = paths
        self.console = Path(console)
        self.conversion = conversion or DEFAULT_CONVERSION
        self.timeout_per_file = timeout_per_file
        self.home = paths.data / "wwise"
        self.project = self.home / PROJECT_NAME / f"{PROJECT_NAME}.wproj"
        self.log_file = paths.logs / "wwise_console.log"

    # ------------------------------------------------------------ process
    def _run(self, args: List[str], timeout: float) -> str:
        cmd = [os_path(self.console)] + args
        if self.console.suffix.lower() == ".py":                  # stand-in used by the tests
            cmd = [sys.executable] + cmd
        creationflags = 0x08000000 if os.name == "nt" else 0      # CREATE_NO_WINDOW
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=creationflags,
                                    cwd=os_path(self.home), stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            raise WwiseError("Wwise took too long and was stopped.", details=" ".join(args)) from exc
        except OSError as exc:
            raise WwiseError("Wwise (WwiseConsole.exe) could not be started.",
                             hint="Check the Wwise location on the Build page.", details=f"{self.console}: {exc}") from exc
        output = ((result.stdout or b"") + b"\n" + (result.stderr or b"")).decode("utf-8", "replace")
        with open(self.log_file, "a", encoding="utf-8") as handle:
            handle.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args[:2])} "
                         f"(exit {result.returncode}, {time.monotonic() - started:.1f}s)\n{output[-20000:]}\n")
        if result.returncode != 0:
            raise WwiseError("Wwise reported an error.", hint="Details are in logs\\wwise_console.log.",
                             details=output.strip()[-1500:])
        return output

    def ensure_project(self) -> Path:
        if not self.project.is_file():
            self.home.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(self.project.parent, ignore_errors=True)
            self._run(["create-new-project", _plain(self.project)], timeout=600)
            if not self.project.is_file():
                raise WwiseError("Wwise did not create its conversion project.", details=str(self.project))
        return self.project

    # --------------------------------------------------------- conversion
    def convert(self, wavs: List[Path], work: Path) -> Dict[str, bytes]:
        """Convert ``wavs`` (one folder) to Wwise Vorbis; {wav stem: .wem bytes}. Every output is checked."""

        if not wavs:
            return {}
        self.ensure_project()
        root = wavs[0].parent
        if any(w.parent != root for w in wavs):
            raise ValueError("all files of one conversion must be in the same folder")
        out = work / "wem"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        listing = work / "list.wsources"
        lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                 f"<ExternalSourcesList SchemaVersion=\"1\" Root={quoteattr(_plain(root))}>"]
        lines += [f"  <Source Path={quoteattr(w.name)} Conversion={quoteattr(self.conversion)}/>" for w in wavs]
        lines.append("</ExternalSourcesList>")
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._run(["convert-external-source", _plain(self.project), "--source-file", _plain(listing),
                   "--output", _plain(out)], timeout=300 + self.timeout_per_file * len(wavs))
        produced = {p.stem: p for p in out.rglob("*.wem")}     # Wwise writes into <output>\Windows\
        results: Dict[str, bytes] = {}
        for w in wavs:
            path = produced.get(w.stem)
            if path is None:
                raise WwiseError("Wwise did not produce a converted file.", hint="See logs\\wwise_console.log.",
                                 details=w.name)
            data = path.read_bytes()
            info = wem.read_wem_info(data)
            if not info.is_vorbis:
                raise WwiseError("Wwise converted the music, but not to Vorbis.",
                                 hint=f"The conversion setting '{self.conversion}' must be a Vorbis setting.",
                                 details=f"{w.name}: format 0x{info.format_tag:04X}")
            results[w.stem] = data
        shutil.rmtree(out, ignore_errors=True)
        return results

    def self_test(self, work: Path) -> Dict[str, object]:
        """Convert two seconds of a test tone; used by the Build page's 'Test Wwise'."""

        work.mkdir(parents=True, exist_ok=True)
        try:
            wav_dir = work / "wav"
            wav_dir.mkdir(exist_ok=True)
            t = np.arange(96000, dtype=np.float32) / 48000
            tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            wem.write_wav(wav_dir / "test_tone.wav", np.stack([tone, tone], axis=1), 48000)
            data = self.convert([wav_dir / "test_tone.wav"], work)["test_tone"]
            info = wem.read_wem_info(data)
            return {"ok": info.is_vorbis and abs((info.samples or 0) - 96000) <= 2048 and info.channels == 2, "bytes": len(data),
                    "samples": info.samples, "channels": info.channels, "sample_rate": info.sample_rate,
                    "console": str(self.console), "version": install_name(self.console)}
        finally:
            shutil.rmtree(work, ignore_errors=True)
