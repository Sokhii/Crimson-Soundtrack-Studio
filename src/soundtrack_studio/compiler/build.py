"""Mod build pipeline.

    accepted mapping ─► plan ─► render user audio onto each segment timeline
        ─► PCM .wem per source (music slice or silence)
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
from ..app_paths import AppPaths
from ..errors import OperationCancelled
from ..game_model.model import GameMusicModel
from ..matching.store import MappingEntry
from . import bnk, wem
from .archive import CompileError, GameFileReader
from .audio import TARGET_RATE, FitSettings, map_channels, render_timeline
from .plan import build_plan, timeline_slice_frames
from .validate import ValidationResult, validate_output

log = logging.getLogger(__name__)

LAYOUTS = ("crimson_browser", "package_folders")


@dataclass
class BuildSettings:
    mod_name: str = "Crimson Soundtrack Replacement"
    author: str = ""
    version: str = "1.0.0"
    description: str = ""
    layout: str = "crimson_browser"      # crimson_browser (manifest.json + files/) | package_folders (<pkg>/<path>)
    normalize: bool = True
    target_rms_dbfs: float = -18.0
    make_zip: bool = True


@dataclass
class BuildResult:
    output_dir: Path
    zip_path: Optional[Path]
    report: Dict[str, Any]
    validation: ValidationResult
    warnings: List[str] = field(default_factory=list)


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
                 track_paths: Dict[int, Path], settings: BuildSettings) -> None:
        self.paths = paths
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
        if not mapping:
            raise CompileError("No replacements are confirmed yet.",
                               hint="Accept or choose tracks on the Matching page first.")
        for entry in mapping:
            path = self.track_paths.get(entry.track_id)
            if path is None or not path.is_file():
                raise CompileError("A chosen music file is missing.", hint="Rescan the music library or choose "
                                   "another track.", details=str(path or entry.track_id))

        report_progress("Planning the build", 0, 0)
        plan = build_plan(self.model, self.conn, self.installation_id, mapping)
        if not plan.jobs:
            raise CompileError("Nothing to build: none of the confirmed cues can be replaced.",
                               details="; ".join(plan.warnings))
        paz_paths = [f"{self.reader.entry_for_asset(a)['package']}/{self.reader.entry_for_asset(a)['paz_index']}.paz"
                     for a in plan.banks() if self.reader.entry_for_asset(a)["origin"] == "archive"]
        game_before = _game_fingerprint(self.game_root, paz_paths)

        work = self.paths.temp / f"build-{uuid.uuid4().hex[:10]}"
        mod_name = safe_name(self.settings.mod_name)
        mod_dir = work / mod_name
        files_root = mod_dir / "files" if self.settings.layout == "crimson_browser" else mod_dir
        files_root.mkdir(parents=True)
        written: List[str] = []
        sources_report: Dict[str, Any] = {}
        cue_report: Dict[str, Any] = {}
        embedded: Dict[int, bytes] = {}
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
                    fit = FitSettings(entry.fit_mode, entry.start_offset_s, self.settings.normalize,
                                      self.settings.target_rms_dbfs)
                    timeline, info = render_timeline(self.track_paths[entry.track_id], plan.cue_durations[cue_key],
                                                     plan.cue_channels[cue_key], fit)
                    cue_report[cue_key] = {"track_id": entry.track_id,
                                           "track_file": self.track_paths[entry.track_id].name,
                                           "duration_s": plan.cue_durations[cue_key], "decided_by": entry.decided_by,
                                           **info}
                for job in jobs_by_cue[cue_key]:
                    self._write_source(job, timeline, files_root, written, sources_report, embedded)
                del timeline

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
                                                           embedded.get(job.source_id)))
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
                "mod_name": mod_name, "layout": self.settings.layout,
                "files_dir": "files" if self.settings.layout == "crimson_browser" else ".",
                "files": sorted(written), "sources": sources_report, "cues": cue_report,
                "analyzer_installation": self.installation_id, "warnings": plan.warnings,
                "settings": asdict(self.settings),
            }
            if self.settings.layout == "crimson_browser":
                (mod_dir / "manifest.json").write_text(json.dumps({
                    "format": "crimson_browser_mod_v1", "id": re.sub(r"[^a-z0-9]+", "-", mod_name.lower()).strip("-"),
                    "name": mod_name, "version": self.settings.version, "author": self.settings.author,
                    "description": self.settings.description or "Music replacement built with Crimson Soundtrack Studio.",
                    "files_dir": "files"}, indent=2, ensure_ascii=False), encoding="utf-8")
            (mod_dir / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            (mod_dir / "README.txt").write_text(_readme(mod_name, report, self.settings), encoding="utf-8")

            # 4. validate what was written, then publish
            report_progress("Validating the mod", 0, 0)
            validation = validate_output(mod_dir, report)
            game_after = _game_fingerprint(self.game_root, paz_paths)
            if game_after != game_before:
                validation.error("The game files changed during the build (they must never be modified).")
            report["validation"] = asdict(validation)
            report["elapsed_s"] = round(time.monotonic() - started, 1)
            (mod_dir / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            if not validation.ok:
                raise CompileError("The built mod failed validation and was not saved.",
                                   details="\n".join(validation.errors[:20]))
            self.paths.output.mkdir(parents=True, exist_ok=True)
            final = self.paths.output / mod_name
            if final.exists():
                backup = self.paths.output / f"{mod_name}.previous"
                shutil.rmtree(backup, ignore_errors=True)
                final.replace(backup)
            shutil.move(str(mod_dir), str(final))
            zip_path = None
            if self.settings.make_zip:
                report_progress("Creating the ZIP file", 0, 0)
                zip_path = self.paths.output / f"{mod_name}.zip"
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
                    for path in sorted(final.rglob("*")):
                        if path.is_file():
                            zf.write(path, f"{mod_name}/{path.relative_to(final).as_posix()}")
            return BuildResult(final, zip_path, report, validation, plan.warnings)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _write_source(self, job, timeline: Optional[np.ndarray], files_root: Path, written: List[str],
                      sources_report: Dict[str, Any], embedded: Dict[int, bytes]) -> None:
        start, frames = timeline_slice_frames(job, TARGET_RATE)
        if job.role == "music" and timeline is not None:
            piece = np.zeros((frames, timeline.shape[1]), np.float32)
            available = max(0, min(frames, len(timeline) - start))
            if available:
                piece[:available] = timeline[start:start + available]
            if piece.shape[1] != job.channels:
                piece = map_channels(piece, job.channels)
        else:
            piece = np.zeros((frames, job.channels), np.float32)
        data = wem.build_pcm_wem(piece, TARGET_RATE)
        bank_rels = sorted({f"{self._package(u.bank_vpath)}/{u.bank_vpath}" for u in job.uses})
        sources_report[str(job.source_id)] = {
            "cue": job.cue_key, "role": job.role, "stream_type": job.stream_type, "frames": frames,
            "channels": job.channels, "banks": bank_rels, "play_at_s": job.play_at_s}
        if job.stream_type == 0:
            embedded[job.source_id] = data
            return
        rel = f"{self._package(job.wem_vpath)}/{job.wem_vpath}"
        target = files_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written.append(rel)
        sources_report[str(job.source_id)]["file"] = rel

    def _package(self, vpath: str) -> str:
        package = self.reader.package_for_vpath(vpath, self.installation_id)
        if package is None:
            raise CompileError("The Analyzer database does not say which game package holds a file.", details=vpath)
        return package


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
