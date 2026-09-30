"""Application service layer.

The GUI (and the CLI/self-test) talk only to :class:`Studio`. It coordinates
the independent components - project store, Analyzer DB import/reader, game
model, game installation check and music library - without containing their
logic. Long operations accept ``progress`` and ``cancel`` callbacks so the UI
can run them in worker threads.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .analyzer_db import compat
from .analyzer_db.importer import ImportedDatabase, import_database, load_imported, source_changed
from .analyzer_db.reader import AnalyzerReader
from .app_paths import AppPaths
from .config import Settings
from .errors import AnalyzerDbError, GameInstallError, LibraryError, ProjectError
from .game_model.builder import load_or_build
from .game_model.model import GameMusicModel
from .library.cache import AnalysisCache
from .library.scanner import LibraryScanner, ScanProgress, ScanStats
from .project.store import Project, ProjectInfo, list_projects

log = logging.getLogger(__name__)

OK, WARN, MISSING, UNAVAILABLE = "ok", "warning", "missing", "unavailable"


@dataclass
class WorkflowStep:
    number: int
    title: str
    state: str          # ok | warning | missing | unavailable
    detail: str = ""


@dataclass
class ProjectStatus:
    name: str
    steps: List[WorkflowStep] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    game_path: str = ""
    game_check: Optional[Dict[str, Any]] = None
    analyzer: Optional[Dict[str, Any]] = None
    library_path: str = ""
    library_counts: Dict[str, int] = field(default_factory=dict)
    last_build: str = ""


class Studio:
    def __init__(self, paths: AppPaths, settings: Optional[Settings] = None) -> None:
        self.paths = paths
        self.settings = settings or Settings.load(paths)
        self.project: Optional[Project] = None
        self._model: Optional[GameMusicModel] = None
        self._cache: Optional[AnalysisCache] = None

    # ------------------------------------------------------------ lifecycle
    def shutdown(self) -> None:
        self.close_project()
        if self._cache is not None:
            self._cache.close()
            self._cache = None
        self.settings.save(self.paths)

    @property
    def analysis_cache(self) -> AnalysisCache:
        if self._cache is None:
            self._cache = AnalysisCache(self.paths.cache / "audio_analysis.sqlite3")
        return self._cache

    # ------------------------------------------------------------- projects
    def list_projects(self) -> List[ProjectInfo]:
        return list_projects(self.paths)

    def create_project(self, name: str) -> Project:
        self.close_project()
        self.project = Project.create(self.paths, name)
        self._remember()
        return self.project

    def open_project(self, folder: Path) -> Project:
        project = Project.open(Path(folder))
        self.close_project()
        self.project = project
        self._remember()
        return project

    def open_last_project(self) -> Optional[Project]:
        folder = self.paths.from_stored(self.settings.last_project)
        if folder is None or not (folder / "project.sqlite3").is_file():
            return None
        try:
            return self.open_project(folder)
        except ProjectError as exc:
            log.warning("Last project could not be reopened: %s", exc.message)
            return None

    def close_project(self) -> None:
        if self.project is not None:
            self.project.close()
        self.project = None
        self._model = None

    def _remember(self) -> None:
        assert self.project is not None
        self.settings.remember_project(self.paths.to_stored(self.project.folder))
        self.settings.save(self.paths)

    def require_project(self) -> Project:
        if self.project is None:
            raise ProjectError("Create or open a project first.")
        return self.project

    # ------------------------------------------------------------- analyzer
    def import_analyzer(self, source: Path, progress: Optional[Callable[[str, int, int], None]] = None,
                        cancel: Optional[Callable[[], bool]] = None) -> ImportedDatabase:
        project = self.require_project()
        imported = import_database(Path(source), self.paths, progress, cancel)
        inst = imported.report.selected_installation_id
        scan_id = None
        for i in imported.report.installations:
            if i.id == inst:
                scan_id = i.latest_scan_id
        project.set_analyzer(sha256=imported.sha256, snapshot_stored=self.paths.to_stored(imported.snapshot_path),
                             source_path=str(imported.manifest.get("source_path", source)),
                             schema_version=imported.schema_version, installation_id=inst, scan_id=scan_id,
                             validation=imported.report.to_dict())
        project.add_event("info", "analyzer", f"Analyzer database imported (schema {imported.schema_version}).")
        for issue in imported.report.warnings:
            project.add_event("warning", "analyzer", issue.message, issue.details)
        self._model = None
        if project.get("game_path"):
            self.check_game()
        return imported

    def active_import(self) -> Optional[ImportedDatabase]:
        project = self.require_project()
        ref = project.active_analyzer()
        if not ref:
            return None
        snapshot = self.paths.from_stored(ref["snapshot_path"])
        return load_imported(self.paths, snapshot) if snapshot else None

    def game_model(self, progress: Optional[Callable[[str], None]] = None) -> Optional[GameMusicModel]:
        project = self.require_project()
        ref = project.active_analyzer()
        if not ref:
            return None
        if self._model is not None and self._model.analyzer_sha256 == ref["sha256"]:
            return self._model
        snapshot = self.paths.from_stored(ref["snapshot_path"])
        if snapshot is None or not snapshot.is_file():
            raise AnalyzerDbError("The imported Analyzer database is missing from the Studio folder.",
                                  hint="Select the Analyzer database again.", details=str(snapshot))
        self._model = load_or_build(self.paths, snapshot, ref["sha256"], ref["installation_id"], progress)
        return self._model

    # ----------------------------------------------------------------- game
    def set_game_path(self, path: Path, *, allow_inside_app: bool = False) -> compat.CompatReport:
        project = self.require_project()
        path = Path(path)
        if not path.is_dir():
            raise GameInstallError("The selected Crimson Desert folder does not exist.", details=str(path))
        if self.paths.is_inside(path) and not allow_inside_app:  # the self-test uses a fake install in temp/
            raise GameInstallError("The game folder cannot be inside the Crimson Soundtrack Studio folder.",
                                   details=str(path))
        project.set("game_path", str(path.resolve()))
        return self.check_game()

    def check_game(self) -> compat.CompatReport:
        project = self.require_project()
        game_path = project.get("game_path")
        if not game_path:
            raise GameInstallError("No Crimson Desert installation has been selected.")
        ref = project.active_analyzer()
        if ref is None:
            ok = compat.looks_like_game_folder(Path(game_path))
            report = compat.CompatReport(status=compat.UNVERIFIABLE if ok else compat.NOT_GAME, game_path=game_path)
            if ok:
                report.notes.append("Select an Analyzer database to verify that it matches this installation.")
        else:
            snapshot = self.paths.from_stored(ref["snapshot_path"])
            with AnalyzerReader(snapshot) as reader:
                inst = reader.installation(ref["installation_id"]) or {}
                files = reader.source_files(ref["installation_id"])
            report = compat.check_installation(Path(game_path), files, inst.get("root_path", ""))
        project.record_game_check(game_path, ref["sha256"] if ref else None, report.status, report.to_dict())
        log.info("Game installation check: %s (%d files compared, %d missing, %d size mismatches)",
                 report.status, report.checked_files, len(report.missing), len(report.size_mismatch))
        return report

    # -------------------------------------------------------------- library
    def set_library_path(self, path: Path) -> None:
        project = self.require_project()
        path = Path(path)
        if not path.is_dir():
            raise LibraryError("The selected music folder does not exist.", details=str(path))
        project.set("music_library_path", str(path.resolve()))

    def scan_library(self, progress: Optional[Callable[[ScanProgress], None]] = None,
                     cancel: Optional[Callable[[], bool]] = None) -> ScanStats:
        project = self.require_project()
        path = project.get("music_library_path")
        if not path:
            raise LibraryError("Choose a music folder first.")
        stats = LibraryScanner(project, self.analysis_cache).scan(Path(path), progress, cancel)
        if stats.errors:
            project.add_event("warning", "library", f"{stats.errors} music files could not be read.",
                              "\n".join(stats.error_samples))
        return stats

    def library_tracks(self) -> List[Dict[str, Any]]:
        project = self.require_project()
        rows = project.query(
            "SELECT t.id, t.rel_path, t.status, t.error, t.duplicate_of, t.size, r.path AS root, m.*, f.features_json"
            " FROM track t JOIN library_root r ON r.id=t.root_id LEFT JOIN track_metadata m ON m.track_id=t.id"
            " LEFT JOIN track_features f ON f.track_id=t.id ORDER BY r.path, t.rel_path")
        out = []
        for row in rows:
            item = dict(row)
            features = json.loads(item.pop("features_json") or "null") or {}
            item["features"] = features
            out.append(item)
        return out

    def library_counts(self) -> Dict[str, int]:
        if self.project is None:
            return {}
        counts = {r["status"]: r["c"] for r in self.project.query("SELECT status, COUNT(*) c FROM track GROUP BY status")}
        counts["duplicates"] = self.project.query_one(
            "SELECT COUNT(*) FROM track WHERE duplicate_of IS NOT NULL")[0]
        counts["total"] = sum(v for k, v in counts.items() if k in ("ok", "error", "missing"))
        return counts

    # --------------------------------------------------------------- status
    def project_status(self) -> ProjectStatus:
        project = self.require_project()
        status = ProjectStatus(name=project.name)
        steps = status.steps

        game_path = project.get("game_path") or ""
        status.game_path = game_path
        check = project.last_game_check()
        status.game_check = check
        if not game_path:
            steps.append(WorkflowStep(1, "Select Crimson Desert installation", MISSING))
        elif not Path(game_path).is_dir():
            steps.append(WorkflowStep(1, "Select Crimson Desert installation", WARN, "The folder is no longer available."))
            status.warnings.append(f"The Crimson Desert folder is no longer available: {game_path}")
        elif check and check["status"] in (compat.MATCH, compat.PROBABLE_MATCH):
            steps.append(WorkflowStep(1, "Select Crimson Desert installation", OK, check["details"].get("summary", "")))
        else:
            detail = check["details"].get("summary", "") if check else ""
            steps.append(WorkflowStep(1, "Select Crimson Desert installation", WARN, detail))
            if check and check["status"] in (compat.MISMATCH, compat.NOT_GAME):
                status.warnings.append(detail)

        ref = project.active_analyzer()
        if ref is None:
            steps.append(WorkflowStep(2, "Select Analyzer database", MISSING))
        else:
            snapshot = self.paths.from_stored(ref["snapshot_path"])
            info = {"sha256": ref["sha256"], "schema_version": ref["schema_version"], "imported_at": ref["imported_at"],
                    "source_path": ref["source_path"], "counts": ref["validation"].get("counts", {}),
                    "snapshot_present": bool(snapshot and snapshot.is_file())}
            status.analyzer = info
            if not info["snapshot_present"]:
                steps.append(WorkflowStep(2, "Select Analyzer database", WARN, "Imported copy missing; select it again."))
                status.warnings.append("The imported Analyzer database is missing from the Studio folder.")
            else:
                detail = f"Schema {ref['schema_version']}, imported {ref['imported_at']}"
                steps.append(WorkflowStep(2, "Select Analyzer database", OK, detail))
                try:
                    manifest = load_imported(self.paths, snapshot).manifest
                    if source_changed(manifest):
                        status.warnings.append("The original Analyzer database has changed since it was imported. "
                                               "Select it again to use the newer data.")
                except AnalyzerDbError:
                    pass
            for issue in ref["validation"].get("issues", []):
                if issue["severity"] == "warning":
                    status.warnings.append(issue["message"])

        library_path = project.get("music_library_path") or ""
        status.library_path = library_path
        counts = self.library_counts()
        status.library_counts = counts
        if not library_path:
            steps.append(WorkflowStep(3, "Select music library folder", MISSING))
        elif not Path(library_path).is_dir():
            steps.append(WorkflowStep(3, "Select music library folder", WARN, "The folder is no longer available."))
            status.warnings.append(f"The music folder is no longer available: {library_path}")
        else:
            steps.append(WorkflowStep(3, "Select music library folder", OK, library_path))
        steps.append(WorkflowStep(4, "Choose AI model", UNAVAILABLE, "Local AI arrives in a later version (Phase 3)."))
        if counts.get("ok"):
            detail = f"{counts.get('ok', 0)} tracks analysed"
            if counts.get("error"):
                detail += f", {counts['error']} unreadable"
            steps.append(WorkflowStep(5, "Analyze music", WARN if counts.get("error") else OK, detail))
        else:
            steps.append(WorkflowStep(5, "Analyze music", MISSING))
        if counts.get("missing"):
            status.warnings.append(f"{counts['missing']} previously scanned music files are no longer found.")
        for number, title in ((6, "Generate thematic matches"), (7, "Review / override matches"), (8, "Build mod"),
                              (9, "Validate output"), (10, "Export mod")):
            steps.append(WorkflowStep(number, title, UNAVAILABLE, "Not available in this version."))
        return status
