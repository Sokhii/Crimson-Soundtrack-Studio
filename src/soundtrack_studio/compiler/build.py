"""Mod build pipeline.

    accepted mapping ─► plan ─► render user audio onto each segment timeline
        ─► .wem per source (music slice or silence): Wwise Vorbis made by the user's Wwise
           (encoder "wwise_vorbis", like every working community mod), or PCM (encoder "pcm")
        ─► read + verify original banks (read-only) ─► patch sources, rebuild DIDX/DATA
        ─► write package into a workspace ─► independent validation
        ─► move to output/<mod name>/ (+ optional .zip)

The game installation and the user's music are only read. Everything is
written under the application folder (``temp/`` workspace, then ``output/``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .. import __version__
from ..app_paths import AppPaths, is_file, os_path
from ..errors import OperationCancelled
from ..game_model.model import GameMusicModel
from ..matching.store import MappingEntry
from . import bnk, wem
from .archive import CompileError, GameFileReader
from .audio import TARGET_RATE, FitSettings, map_channels, render_timeline, slice_timeline
from .loudness import integrated_lufs, level
from .plan import build_plan, timeline_slice_frames
from .validate import ValidationResult, report_file_name, validate_output

log = logging.getLogger(__name__)

LAYOUTS = ("crimson_browser", "package_folders")


@dataclass
class BuildSettings:
    mod_name: str = "Crimson Soundtrack Replacement"
    author: str = ""
    version: str = "1.0.0"
    description: str = ""
    layout: str = "crimson_browser"      # crimson_browser (manifest.json + files/) | package_folders (<pkg>/<path>)
    loudness_mode: str = "match"         # match (each cue as loud as the original it replaces) | fixed | off
    target_lufs: float = -16.0           # fixed mode; in match mode for cues whose original was not measured
    ceiling_dbtp: float = -1.0           # no replacement's true peak goes above this (no limiter, no distortion)
    match_floor_db: float = 6.0          # match mode: a cue is never aimed more than this far below target_lufs
    make_zip: bool = True
    encoder: str = "pcm"                 # wwise_vorbis (needs Wwise) | pcm (no extra software; not proven in game)


ENCODERS = ("wwise_vorbis", "pcm")
LOUDNESS_MODES = ("match", "fixed", "off")
WWISE_BATCH = 24                         # sources per Wwise run (each run has a start-up cost; WAVs are large)


@dataclass
class BuildResult:
    output_dir: Path
    zip_path: Optional[Path]
    report: Dict[str, Any]
    validation: ValidationResult
    warnings: List[str] = field(default_factory=list)
    report_path: Optional[Path] = None


def safe_name(name: str) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip(" .")
    return text[:80] or "Crimson Soundtrack Replacement"


def _game_fingerprint(game_root: Path, rel_paths: List[str]) -> Dict[str, Any]:
    out = {}
    for rel in sorted(set(rel_paths)):
        try:
            st = os.stat(game_root / rel)
            out[rel] = (st.st_size, st.st_mtime_ns)
        except OSError:
            out[rel] = None
    return out


class ModBuilder:
    def __init__(self, paths: AppPaths, game_root: Path, analyzer_conn, installation_id: int, model: GameMusicModel,
                 track_paths: Dict[int, Path], settings: BuildSettings, wwise=None,
                 references: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.paths = paths
        self.wwise = wwise                # compiler.wwise.WwiseEncoder for encoder "wwise_vorbis"
        self.references = references or {}  # cue key -> measured original {"lufs": ..., "rms": ...}
        self.game_root = Path(game_root)
        self.conn = analyzer_conn
        self.installation_id = installation_id
        self.model = model
        self.track_paths = track_paths
        self.settings = settings
        self.reader = GameFileReader(self.game_root, analyzer_conn)

    def build(self, mapping: List[MappingEntry], progress: Optional[Callable[[str, int, int], None]] = None,
              cancel: Optional[Callable[[], bool]] = None) -> BuildResult:
        started = time.monotonic()
        report_progress = progress or (lambda *a: None)
        if self.settings.layout not in LAYOUTS:
            raise CompileError("Unknown package layout.", details=self.settings.layout)
        if self.settings.loudness_mode not in LOUDNESS_MODES:
            raise CompileError("Unknown loudness setting.", details=self.settings.loudness_mode)
        if self.settings.encoder not in ENCODERS:
            raise CompileError("Unknown audio format.", details=self.settings.encoder)
        vorbis = self.settings.encoder == "wwise_vorbis"
        if not mapping:
            raise CompileError("No replacements are confirmed yet.",
                               hint="Accept or choose tracks on the Matching page first.")
        for entry in mapping:
            path = self.track_paths.get(entry.track_id)
            if path is None or not is_file(path):
                raise CompileError("A chosen music file is missing.", hint="Rescan the music library or choose "
                                   "another track.", details=str(path or entry.track_id))
        if vorbis and self.wwise is None:
            raise CompileError("Wwise is needed to build in the Vorbis format, and it was not found.",
                               hint="Install Wwise (Build page, 'Get Wwise') or choose the PCM format.")

        report_progress("Planning the build", 0, 0)
        plan = build_plan(self.model, self.conn, self.installation_id, mapping)
        if not plan.jobs:
            raise CompileError("Nothing to build: none of the confirmed cues can be replaced.",
                               details="; ".join(plan.warnings))
        paz_paths = [f"{self.reader.entry_for_asset(a)['package']}/{self.reader.entry_for_asset(a)['paz_index']}.paz"
                     for a in plan.banks() if self.reader.entry_for_asset(a)["origin"] == "archive"]
        game_before = _game_fingerprint(self.game_root, paz_paths)

        # The work folder sits inside the app folder, which may itself be deep in the user's folders; a mod's file paths
        # (files/0004/sound/windows/media/...) then pass Windows' 260-character limit. Everything below derives from a
        # path with the extended-length prefix, which lifts that limit.
        work = Path(os_path(self.paths.temp / f"build-{uuid.uuid4().hex[:10]}", force=True))
        mod_name = safe_name(self.settings.mod_name)
        mod_dir = work / mod_name
        files_root = mod_dir / "files" if self.settings.layout == "crimson_browser" else mod_dir
        files_root.mkdir(parents=True)
        written: List[str] = []
        sources_report: Dict[str, Any] = {}
        cue_report: Dict[str, Any] = {}
        embedded: Dict[int, bytes] = {}
        prefetch: Dict[int, bytes] = {}
        pending: List[Any] = []
        total_jobs = len(plan.jobs)
        converted = [0]

        def flush() -> None:
            report_progress("Converting with Wwise", converted[0], total_jobs)
            self._convert_pending(pending, work, files_root, written, sources_report, embedded, prefetch)
            converted[0] += len(pending)
            pending.clear()
            report_progress("Converting with Wwise", converted[0], total_jobs)

        try:
            # 1. one cue at a time: render its timeline, write its sources, release the audio
            #    (a full soundtrack would not fit in memory if all timelines were kept)
            jobs_by_cue: Dict[str, list] = {}
            for job in plan.jobs.values():
                jobs_by_cue.setdefault(job.cue_key, []).append(job)
            for i, cue_key in enumerate(sorted(jobs_by_cue)):
                if cancel and cancel():
                    raise OperationCancelled()
                entry = plan.cues[cue_key]
                report_progress("Preparing replacement audio", i, len(jobs_by_cue))
                timeline = None
                if any(j.role == "music" for j in jobs_by_cue[cue_key]):
                    fit = FitSettings(entry.fit_mode, entry.start_offset_s)
                    timeline, info = render_timeline(self.track_paths[entry.track_id], plan.cue_durations[cue_key],
                                                     plan.cue_channels[cue_key], fit)
                    timeline, loud = self._level(cue_key, timeline)
                    info.update(loud)
                    cue_report[cue_key] = {"track_id": entry.track_id,
                                           "track_file": self.track_paths[entry.track_id].name,
                                           "duration_s": plan.cue_durations[cue_key], "decided_by": entry.decided_by,
                                           **info}
                for job in jobs_by_cue[cue_key]:
                    if vorbis:
                        self._queue_source(job, timeline, work, sources_report)
                        pending.append(job)
                    else:
                        self._write_source(job, timeline, files_root, written, sources_report, embedded)
                del timeline
                if vorbis and len(pending) >= WWISE_BATCH:
                    if cancel and cancel():
                        raise OperationCancelled()
                    flush()
            if pending:
                flush()

            # 2. patch every affected bank (read-only access to the game)
            banks = plan.banks()
            for i, (asset_id, vpath) in enumerate(sorted(banks.items())):
                if cancel and cancel():
                    raise OperationCancelled()
                report_progress("Patching soundbanks", i, len(banks))
                original = self.reader.read(asset_id)
                patches = []
                for job in plan.jobs.values():
                    for use in job.uses:
                        if use.bank_asset_id == asset_id:
                            patches.append(bnk.SourcePatch(use.object_id, job.source_id, use.plugin_id, use.stream_type,
                                                           use.in_memory_size if use.stream_type else None,
                                                           embedded.get(job.source_id),
                                                           codec="vorbis" if vorbis else "pcm",
                                                           prefetch_data=prefetch.get(job.source_id)))
                patched, notes = bnk.patch_bank(original, patches)
                rel = f"{self._package(vpath)}/{vpath}"
                target = files_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(patched)
                written.append(rel)
                log.info("Patched %s: %d sources; %s", vpath, len(patches), "; ".join(notes) or "no media changes")

            # 3. manifest, readme, report
            report = {
                "generator": f"Crimson Soundtrack Studio {__version__}", "format": "css-build-report", "version": 1,
                "mod_name": mod_name, "layout": self.settings.layout, "codec": "vorbis" if vorbis else "pcm",
                "files_dir": "files" if self.settings.layout == "crimson_browser" else ".",
                "files": sorted(written), "sources": sources_report, "cues": cue_report,
                "analyzer_installation": self.installation_id, "warnings": plan.warnings,
                "loudness": loudness_summary(cue_report, self.settings),
                "settings": asdict(self.settings),
            }
            if self.settings.layout == "crimson_browser":
                (mod_dir / "manifest.json").write_text(json.dumps({
                    "format": "crimson_browser_mod_v1", "id": re.sub(r"[^a-z0-9]+", "-", mod_name.lower()).strip("-"),
                    "name": mod_name, "version": self.settings.version, "author": self.settings.author,
                    "description": self.settings.description or "Music replacement built with Crimson Soundtrack Studio.",
                    "files_dir": "files"}, indent=2, ensure_ascii=False), encoding="utf-8")
            (mod_dir / "README.txt").write_text(_readme(mod_name, report, self.settings), encoding="utf-8")

            # 4. validate what was written, then publish
            report_progress("Validating the mod", 0, 0)
            validation = validate_output(mod_dir, report)
            game_after = _game_fingerprint(self.game_root, paz_paths)
            if game_after != game_before:
                validation.error("The game files changed during the build (they must never be modified).")
            report["validation"] = asdict(validation)
            report["elapsed_s"] = round(time.monotonic() - started, 1)
            report_text = json.dumps(report, indent=2, ensure_ascii=False)
            if not validation.ok:
                self.paths.logs.mkdir(parents=True, exist_ok=True)
                (self.paths.logs / "failed_build_report.json").write_text(report_text, encoding="utf-8")
                raise CompileError("The built mod failed validation and was not saved.",
                                   details="\n".join(validation.errors[:20]))
            self.paths.output.mkdir(parents=True, exist_ok=True)
            shown = self.paths.output / mod_name
            final = Path(os_path(shown, force=True))
            if final.exists():
                backup = Path(os_path(self.paths.output / f"{mod_name}.previous", force=True))
                shutil.rmtree(backup, ignore_errors=True)
                final.replace(backup)
            shutil.move(str(mod_dir), str(final))
            # The report is the Studio's own record. It is kept next to the mod, not in it: mod managers read the
            # JSON files they find in a mod folder (DMM logged parse errors for ours), and the game does not need it.
            report_path = self.paths.output / report_file_name(mod_name)
            report_path.write_text(report_text, encoding="utf-8")
            zip_path = None
            if self.settings.make_zip:
                report_progress("Creating the ZIP file", 0, 0)
                zip_path = self.paths.output / f"{mod_name}.zip"
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
                    for path in sorted(final.rglob("*")):
                        if path.is_file():
                            zf.write(path, f"{mod_name}/{path.relative_to(final).as_posix()}")
            return BuildResult(shown, zip_path, report, validation, plan.warnings, report_path)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _level(self, cue_key: str, timeline: np.ndarray):
        """One gain for the cue's audio: towards the original's loudness (or the fixed target), never past the
        true-peak ceiling. Returns the audio and what was done, for the report."""

        mode = self.settings.loudness_mode
        measured = integrated_lufs(timeline, TARGET_RATE)
        ref = self.references.get(cue_key) or {}
        if mode == "off":
            target, source = None, "own level (only lowered if its peaks were too high)"
        elif mode == "match" and ref.get("lufs") is not None:
            target, source = float(ref["lufs"]), "original (measured loudness)"
        elif mode == "match" and ref.get("rms") is not None and measured is not None:
            ours_rms = 20 * np.log10(max(float(np.sqrt(np.mean(timeline.astype(np.float64) ** 2))), 1e-9))
            target, source = measured + (float(ref["rms"]) - ours_rms), "original (estimated from its average level)"
        else:
            target = self.settings.target_lufs
            source = "fixed target" if mode == "fixed" else "fixed target (the original was not measured)"
        floored = False
        if mode == "match" and target is not None:
            # A measured original can be far quieter than a full mix should be (one layer of several, an ambient bed,
            # a quiet file the game turns up in its banks): never aim more than match_floor_db below the target.
            floor = self.settings.target_lufs - max(0.0, self.settings.match_floor_db)
            if target < floor:
                target, floored = floor, True
        out, result = level(timeline, TARGET_RATE, target, self.settings.ceiling_dbtp, measured=measured)
        info = result.to_dict()
        info["loudness_reference"] = source
        if floored:
            info["raised_to_floor"] = True
        if ref.get("lufs") is not None:
            info["original_lufs"] = ref["lufs"]
        return out, info

    @staticmethod
    def _piece(job, timeline: Optional[np.ndarray]) -> np.ndarray:
        start, frames = timeline_slice_frames(job, TARGET_RATE)
        if job.role == "music" and timeline is not None:
            piece = slice_timeline(timeline, start, frames)
            if piece.shape[1] != job.channels:
                piece = map_channels(piece, job.channels)
            return piece
        return np.zeros((frames, job.channels), np.float32)

    def _report_source(self, job, frames: int, sources_report: Dict[str, Any], codec: str) -> Dict[str, Any]:
        bank_rels = sorted({f"{self._package(u.bank_vpath)}/{u.bank_vpath}" for u in job.uses})
        entry = {"cue": job.cue_key, "role": job.role, "stream_type": job.stream_type, "frames": frames,
                 "channels": job.channels, "banks": bank_rels, "play_at_s": job.play_at_s, "codec": codec,
                 "bank_stream_types": sorted({u.stream_type for u in job.uses})}
        sources_report[str(job.source_id)] = entry
        return entry

    def _place(self, job, data: bytes, files_root: Path, written: List[str], entry: Dict[str, Any],
               embedded: Dict[int, bytes]) -> None:
        if job.stream_type == 0:
            embedded[job.source_id] = data
            return
        rel = f"{self._package(job.wem_vpath)}/{job.wem_vpath}"
        target = files_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written.append(rel)
        entry["file"] = rel

    def _write_source(self, job, timeline: Optional[np.ndarray], files_root: Path, written: List[str],
                      sources_report: Dict[str, Any], embedded: Dict[int, bytes]) -> None:
        piece = self._piece(job, timeline)
        data = wem.build_pcm_wem(piece, TARGET_RATE)
        entry = self._report_source(job, len(piece), sources_report, "pcm")
        self._place(job, data, files_root, written, entry, embedded)

    def _queue_source(self, job, timeline: Optional[np.ndarray], work: Path, sources_report: Dict[str, Any]) -> None:
        """Vorbis: write the source's audio as a .wav for the next Wwise run."""

        piece = self._piece(job, timeline)
        wav_dir = work / "wav"
        wav_dir.mkdir(exist_ok=True)
        wem.write_wav(wav_dir / f"{job.source_id}.wav", piece, TARGET_RATE)
        self._report_source(job, len(piece), sources_report, "vorbis")

    def _convert_pending(self, jobs: List[Any], work: Path, files_root: Path, written: List[str],
                         sources_report: Dict[str, Any], embedded: Dict[int, bytes], prefetch: Dict[int, bytes]) -> None:
        if not jobs:
            return
        wav_dir = work / "wav"
        wavs = [wav_dir / f"{job.source_id}.wav" for job in jobs]
        try:
            results = self.wwise.convert(wavs, work / "wwise")
        finally:
            for w in wavs:
                w.unlink(missing_ok=True)
        for job in jobs:
            data = results[str(job.source_id)]
            info = wem.read_wem_info(data)
            entry = sources_report[str(job.source_id)]
            if info.channels != job.channels or info.sample_rate != TARGET_RATE:
                raise CompileError("Wwise changed the channel count or sample rate of the music.",
                                   hint="Use a Vorbis conversion setting without channel or rate changes.",
                                   details=f"source {job.source_id}: {info.channels} ch {info.sample_rate} Hz, "
                                           f"expected {job.channels} ch {TARGET_RATE} Hz")
            entry["samples"] = info.samples
            entry["bytes"] = len(data)
            self._place(job, data, files_root, written, entry, embedded)
            if job.stream_type != 0:
                prefetch[job.source_id] = wem.prefetch_prefix(data)
                entry["prefetch_bytes"] = len(prefetch[job.source_id])

    def _package(self, vpath: str) -> str:
        package = self.reader.package_for_vpath(vpath, self.installation_id)
        if package is None:
            raise CompileError("The Analyzer database does not say which game package holds a file.", details=vpath)
        return package


def loudness_summary(cues: Dict[str, Any], settings: BuildSettings) -> Dict[str, Any]:
    """How the cues came out: how many reached their target, how many stayed below it to keep peaks clean."""

    short = [c.get("short_of_target_db") or 0.0 for c in cues.values()]
    below = [x for x in short if x >= 0.5]
    sources: Dict[str, int] = {}
    for c in cues.values():
        key = c.get("loudness_reference", "")
        sources[key] = sources.get(key, 0) + 1
    return {"mode": settings.loudness_mode, "ceiling_dbtp": settings.ceiling_dbtp, "cues": len(cues),
            "reached_target": len(short) - len(below), "below_target": len(below),
            "most_below_db": round(max(below), 1) if below else 0.0, "references": sources,
            "raised_to_floor": sum(1 for c in cues.values() if c.get("raised_to_floor")),
            "floor_lufs": settings.target_lufs - settings.match_floor_db if settings.loudness_mode == "match" else None,
            "estimated": sum(1 for c in cues.values() if "estimated" in c.get("loudness_reference", ""))}


def _readme(name: str, report: Dict[str, Any], settings: BuildSettings) -> str:
    lines = [name, "=" * len(name), "", "Built with Crimson Soundtrack Studio. This mod replaces game music with music",
             "the mod's author supplied. It contains no original game audio.", "",
             "Install", "-------",
             "Add this folder (or the ZIP) to your mod manager (DMM or CDUMM) and apply it.",
             "The mod contains replacement files only; the mod manager builds the game overlay.",
             "Layout: " + ("manifest.json + files/<package>/<path> (Crimson Browser format)"
                           if settings.layout == "crimson_browser" else "<package>/<path> folders"), "",
             f"Replaced cues: {len(report['cues'])}", f"Files: {len(report['files'])}", ""]
    for key, cue in sorted(report["cues"].items()):
        lines.append(f"  segment {key}: {cue['track_file']} ({cue['fit']})")
    if report["warnings"]:
        lines += ["", "Notes", "-----"] + [f"  - {w}" for w in report["warnings"]]
    return "\n".join(lines) + "\n"
