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
import re
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
CONVERSION_NAME = "CSS Vorbis"          # a conversion setting the Studio defines itself in its Wwise project
CONVERSION_ID = "{7A1C5E52-3B8D-4C61-9F0A-2D5B6E9C1A47}"
CONVERSION_PLUGIN_ID = "{C3E81F46-5D27-49B0-8A6E-0F4D92B7A5C1}"
CONSOLE_RELS = tuple(Path("Authoring") / arch / "Release" / "bin" / "WwiseConsole.exe" for arch in ("x64", "Win32"))
CONSOLE_REL = CONSOLE_RELS[0]
PROJECT_NAME = "CssConvert"
INSTALL_PARENTS = ("Audiokinetic",)     # ...\Audiokinetic\Wwise2023.1.x or ...\Audiokinetic\Wwise 2023.1.x


class WwiseError(CompileError):
    title = "Wwise problem"


def search_roots() -> List[Path]:
    """Folders that can contain the Wwise installs: the Launcher's default is C:\\Audiokinetic, older ones used
    Program Files; other drives are checked too because the install folder is a choice in the Launcher."""

    roots: List[Path] = []
    for base in (os.environ.get("ProgramFiles(x86)", ""), os.environ.get("ProgramFiles", ""),
                 r"C:\Program Files (x86)", r"C:\Program Files"):
        if base:
            roots.append(Path(base) / "Audiokinetic")
    for letter in "CDEFGHIJ":
        roots.append(Path(f"{letter}:\\") / "Audiokinetic")
        roots.append(Path(f"{letter}:\\") / "Wwise")
    seen, out = set(), []
    for r in roots:
        if str(r).lower() not in seen:
            seen.add(str(r).lower())
            out.append(r)
    return out


def _candidates() -> Iterable[Path]:
    root = os.environ.get("WWISEROOT", "")
    if root:
        for rel in CONSOLE_RELS:
            yield Path(root) / rel
    for folder in search_roots():
        try:
            installs = sorted(folder.glob("Wwise*"), reverse=True) if folder.is_dir() else []
        except OSError:
            installs = []
        installs.sort(key=lambda p: RECOMMENDED_VERSION not in p.name)   # the game's version first, then newest
        for install in installs:
            for rel in CONSOLE_RELS:
                yield install / rel
        if folder.name == "Wwise":                                       # a folder that is itself an install
            for rel in CONSOLE_RELS:
                yield folder / rel
    found = shutil.which("WwiseConsole") or shutil.which("WwiseConsole.exe")
    if found:
        yield Path(found)


def searched_places() -> List[str]:
    """Where find_console() looks, for the message shown when nothing is found."""

    places = ([os.environ["WWISEROOT"]] if os.environ.get("WWISEROOT") else []) + [str(r) for r in search_roots()]
    return places + ["the PATH"]


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


# What Wwise itself saved for a project whose default conversion was set to Vorbis in the user's test (Wwise 2023.1.19,
# project schema 119): the Windows plug-in is "Vorbis" (company 0, plug-in 4), everything else at its defaults.
_FALLBACK_WORK_UNIT = """<?xml version="1.0" encoding="utf-8"?>
<WwiseDocument Type="WorkUnit" ID="{289DFBFC-2CF4-4F01-972E-9BA48FE725AE}" SchemaVersion="119">
\t<Conversions>
\t\t<WorkUnit Name="Default Work Unit" ID="{289DFBFC-2CF4-4F01-972E-9BA48FE725AE}" PersistMode="Standalone">
\t\t\t<ChildrenList>
\t\t\t\t<Conversion Name="Default Conversion Settings" ID="{6D1B890C-9826-4384-BF07-C15223E9FB56}">
\t\t\t\t\t<PropertyList>
\t\t\t\t\t\t<Property Name="Channels" Type="int32">
\t\t\t\t\t\t\t<ValueList>
\t\t\t\t\t\t\t\t<Value Platform="Windows">4</Value>
\t\t\t\t\t\t\t</ValueList>
\t\t\t\t\t\t</Property>
\t\t\t\t\t\t<Property Name="LRMix" Type="Real64">
\t\t\t\t\t\t\t<ValueList>
\t\t\t\t\t\t\t\t<Value Platform="Windows">0</Value>
\t\t\t\t\t\t\t</ValueList>
\t\t\t\t\t\t</Property>
\t\t\t\t\t\t<Property Name="MaxSampleRate" Type="int32">
\t\t\t\t\t\t\t<ValueList>
\t\t\t\t\t\t\t\t<Value Platform="Windows">0</Value>
\t\t\t\t\t\t\t</ValueList>
\t\t\t\t\t\t</Property>
\t\t\t\t\t\t<Property Name="MinSampleRate" Type="int32">
\t\t\t\t\t\t\t<ValueList>
\t\t\t\t\t\t\t\t<Value Platform="Windows">0</Value>
\t\t\t\t\t\t\t</ValueList>
\t\t\t\t\t\t</Property>
\t\t\t\t\t\t<Property Name="SampleRate" Type="int32">
\t\t\t\t\t\t\t<ValueList>
\t\t\t\t\t\t\t\t<Value Platform="Windows">0</Value>
\t\t\t\t\t\t\t</ValueList>
\t\t\t\t\t\t</Property>
\t\t\t\t\t</PropertyList>
\t\t\t\t\t<ConversionPluginInfoList>
\t\t\t\t\t\t<ConversionPluginInfo Platform="Windows">
\t\t\t\t\t\t\t<ConversionPlugin Name="" ID="{A499C085-FCF8-4A6F-B18D-3005FCC5E4B3}" PluginName="Vorbis" CompanyID="0" PluginID="4"/>
\t\t\t\t\t\t</ConversionPluginInfo>
\t\t\t\t\t</ConversionPluginInfoList>
\t\t\t\t</Conversion>
\t\t\t</ChildrenList>
\t\t</WorkUnit>
\t</Conversions>
</WwiseDocument>
"""
_VORBIS_PLUGIN = ('<ConversionPluginInfo Platform="Windows">\n{indent}\t<ConversionPlugin Name="" ID="{plugin_id}" '
                  'PluginName="Vorbis" CompanyID="0" PluginID="4"/>\n{indent}</ConversionPluginInfo>')


def vorbis_conversion_work_unit(existing: str) -> str:
    """The project's conversion work unit with Vorbis (Windows) as the default and as ``CONVERSION_NAME``.

    A project created by ``WwiseConsole create-new-project`` has only "Default Conversion Settings" (not Vorbis, and
    none of the factory presets such as "Vorbis Quality High", which is why asking for that name fell back to the
    default). Both entries are set to Vorbis with Wwise's default quality, channels and sample rate (source). The
    existing file is patched so its own ids stay valid; if it does not look as expected, the file Wwise itself wrote
    for a Vorbis default is used."""

    if CONVERSION_NAME in existing and 'PluginName="Vorbis"' in existing:
        return existing
    block = re.search(r'([ \t]*)<Conversion Name="Default Conversion Settings"[^>]*>.*?</Conversion>[ \t]*\n', existing,
                      re.S)
    if block is None:
        block_text, indent = None, ""
    else:
        block_text, indent = block.group(0), block.group(1)
    if block_text is not None:
        plugin_id = (re.search(r'<ConversionPlugin[^>]*\bID="(\{[0-9A-Fa-f-]+\})"', block_text) or [None, CONVERSION_PLUGIN_ID])[1]
        info = re.compile(r'<ConversionPluginInfo Platform="Windows">.*?</ConversionPluginInfo>', re.S)
        vorbis = _VORBIS_PLUGIN.format(indent=indent + "\t\t\t", plugin_id=plugin_id)
        if info.search(block_text):
            patched = info.sub(lambda _m: vorbis, block_text, count=1)
        elif "</Conversion>" in block_text:
            patched = block_text.replace(
                "</Conversion>", f"\t<ConversionPluginInfoList>\n{indent}\t\t\t{vorbis}\n"
                                 f"{indent}\t</ConversionPluginInfoList>\n{indent}</Conversion>", 1)
        else:
            patched = None
        if patched is not None and 'PluginName="Vorbis"' in patched:
            ours = re.sub(r'(<Conversion Name=)"Default Conversion Settings"( ID=)"\{[0-9A-Fa-f-]+\}"',
                          lambda m: f'{m.group(1)}"{CONVERSION_NAME}"{m.group(2)}"{CONVERSION_ID}"', patched, count=1)
            ours = re.sub(r'(<ConversionPlugin Name="" ID=)"\{[0-9A-Fa-f-]+\}"',
                          lambda m: f'{m.group(1)}"{CONVERSION_PLUGIN_ID}"', ours, count=1)
            if CONVERSION_NAME in ours:
                return existing.replace(block_text, patched + ours, 1)
    return _fallback_with_ours()


def _fallback_with_ours() -> str:
    block = re.search(r'([ \t]*)<Conversion Name="Default Conversion Settings".*?</Conversion>\n', _FALLBACK_WORK_UNIT, re.S)
    text = block.group(0)
    ours = re.sub(r'(<Conversion Name=)"Default Conversion Settings"( ID=)"\{[0-9A-Fa-f-]+\}"',
                  lambda m: f'{m.group(1)}"{CONVERSION_NAME}"{m.group(2)}"{CONVERSION_ID}"', text, count=1)
    ours = re.sub(r'(<ConversionPlugin Name="" ID=)"\{[0-9A-Fa-f-]+\}"',
                  lambda m: f'{m.group(1)}"{CONVERSION_PLUGIN_ID}"', ours, count=1)
    return _FALLBACK_WORK_UNIT.replace(text, text + ours, 1)


def _plain(path: Path) -> str:
    """A normal absolute path for Wwise (the build folder may carry the \\\\?\\ long-path prefix, which Wwise's
    own path handling is not known to accept); the build's Wwise paths stay well under 260 characters."""

    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        return "\\\\" + text[8:]
    return text[4:] if text.startswith("\\\\?\\") else text


MAX_SAFE_PATH = 230                     # Wwise's own tools stop working near Windows' 260-character limit


def short_path(path: Path) -> str:
    """The path with every existing folder in its short 8.3 form (Windows), else as given.

    Wwise's tools use the classic path APIs and fail near 260 characters ("Can't open source or output file" for a
    converted file); the Studio's folder can already be 200 characters deep, and Wwise adds its own cache folders
    below the project. The short form of an existing folder is a few characters per level."""

    text = _plain(path)
    if os.name != "nt":
        return text
    import ctypes

    probe = Path(text)
    tail: List[str] = []
    while not probe.exists() and probe.parent != probe:       # shorten the part that exists, keep the rest as it is
        tail.insert(0, probe.name)
        probe = probe.parent
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(probe), buf, 1024)
    base = buf.value if 0 < n < 1024 else str(probe)
    return str(Path(base).joinpath(*tail)) if tail else base


class WwiseEncoder:
    def __init__(self, paths: AppPaths, console: Path, conversion: str = CONVERSION_NAME,
                 timeout_per_file: float = 120.0) -> None:
        self.paths = paths
        self.console = Path(console)
        self.conversion = conversion or CONVERSION_NAME
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
        if result.returncode not in (0, 2):                       # 2 = finished with warnings
            raise WwiseError("Wwise reported an error.", hint="Details are in logs\\wwise_console.log.",
                             details=output.strip()[-1500:])
        return output

    def ensure_project(self) -> Path:
        if not self.project.is_file():
            self.home.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(self.project.parent, ignore_errors=True)
            self._run(["create-new-project", short_path(self.project)], timeout=600)
            if not self.project.is_file():
                raise WwiseError("Wwise did not create its conversion project.", details=str(self.project))
        self._ensure_conversion()
        return self.project

    def _ensure_conversion(self) -> None:
        """Make the project convert to Vorbis (see ``vorbis_conversion_work_unit``)."""

        unit = self.project.parent / "Conversion Settings" / "Default Work Unit.wwu"
        try:
            existing = unit.read_text(encoding="utf-8") if unit.is_file() else ""
            wanted = vorbis_conversion_work_unit(existing)
            if wanted != existing:
                unit.parent.mkdir(parents=True, exist_ok=True)
                unit.write_text(wanted, encoding="utf-8", newline="")
        except OSError as exc:
            raise WwiseError("The Wwise conversion project could not be set up.", details=f"{unit}: {exc}") from exc

    # --------------------------------------------------------- conversion
    def convert(self, wavs: List[Path], work: Path) -> Dict[str, bytes]:
        """Convert ``wavs`` (one folder) to Wwise Vorbis; {wav stem: .wem bytes}. Every output is checked."""

        if not wavs:
            return {}
        self.ensure_project()
        longest = max(len(short_path(wavs[0])), len(short_path(work)) + len("/wem/Windows/") + len(wavs[0].name),
                      len(short_path(self.project)) + len("/.cache/Windows/SFX/") + len(wavs[0].name) + 40)
        if longest > MAX_SAFE_PATH:
            raise WwiseError("The Studio's folder is too deep for Wwise (Windows' 260-character path limit).",
                             hint="Move the whole Studio folder to a shorter location such as C:\\CSS and try again.",
                             details=f"{longest} characters even in short form: {short_path(work)}")
        root = wavs[0].parent
        if any(w.parent != root for w in wavs):
            raise ValueError("all files of one conversion must be in the same folder")
        out = work / "wem"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        listing = work / "list.wsources"
        lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                 f"<ExternalSourcesList SchemaVersion=\"1\" Root={quoteattr(short_path(root))}>"]
        lines += [f"  <Source Path={quoteattr(w.name)} Conversion={quoteattr(self.conversion)}/>" for w in wavs]
        lines.append("</ExternalSourcesList>")
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        output = self._run(["convert-external-source", short_path(self.project), "--source-file", short_path(listing),
                            "--output", short_path(out)], timeout=300 + self.timeout_per_file * len(wavs))
        produced = {p.stem: p for p in out.rglob("*.wem")}     # Wwise writes into <output>\Windows\
        results: Dict[str, bytes] = {}
        for w in wavs:
            path = produced.get(w.stem)
            if path is None:
                raise WwiseError("Wwise did not produce a converted file.", hint="See logs\\wwise_console.log.",
                                 details=f"{w.name}\n{output.strip()[-1200:]}")
            data = path.read_bytes()
            info = wem.read_wem_info(data)
            if not info.is_vorbis:
                raise WwiseError("Wwise converted the music, but not to Vorbis.",
                                 hint="The Studio sets its Wwise project to Vorbis itself; if it was changed by hand, delete "
                                      "data\\wwise\\CssConvert so it is created again.",
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
