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

from .ai import downloader, hardware
from .ai.catalog import LocalModel, ModelCatalog, ModelError, ModelRegistry, model_status, register_custom_model
from .ai.runtime import InferenceBackend, LlamaServerBackend, find_llama_server, run_inference_check
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
from .project.store import Project, ProjectInfo, list_projects, now_iso
from .semantic.context import cue_document, track_document
from .semantic.store import EffectiveProfile, ResponseCache, RunStats, SemanticStore

log = logging.getLogger(__name__)

OK, WARN, MISSING, UNAVAILABLE = "ok", "warning", "missing", "unavailable"
SHORT_CUE_MS = 15000  # transition-length segments are kept original by default


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
        self._response_cache: Optional[ResponseCache] = None
        self._backend: Optional[InferenceBackend] = None
        self._hardware: Optional[hardware.HardwareInfo] = None
        self.registry = ModelRegistry(paths)
        # tests replace this to inject a scripted model
        self.backend_factory: Callable[[LocalModel], InferenceBackend] = self._make_llama_backend

    # ------------------------------------------------------------ lifecycle
    def shutdown(self) -> None:
        self.stop_ai()
        self.close_project()
        if self._cache is not None:
            self._cache.close()
            self._cache = None
        if self._response_cache is not None:
            self._response_cache.close()
            self._response_cache = None
        self.settings.save(self.paths)

    @property
    def analysis_cache(self) -> AnalysisCache:
        if self._cache is None:
            self._cache = AnalysisCache(self.paths.cache / "audio_analysis.sqlite3")
        return self._cache

    @property
    def response_cache(self) -> ResponseCache:
        if self._response_cache is None:
            self._response_cache = ResponseCache(self.paths.cache / "ai_responses.sqlite3")
        return self._response_cache

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

    # ------------------------------------------------------------------- AI
    def catalog(self) -> ModelCatalog:
        return ModelCatalog.load(self.paths, self.registry)

    def hardware(self) -> hardware.HardwareInfo:
        if self._hardware is None:
            self._hardware = hardware.detect()
        return self._hardware

    def runtime_path(self) -> Optional[Path]:
        return find_llama_server(self.paths, self.settings.llama_server_path)

    def model_rows(self) -> List[Dict[str, Any]]:
        info = self.hardware()
        rows = []
        for model in self.catalog().models.values():
            status = model_status(self.paths, self.registry, model)
            status.update({"model": model, "fits": hardware.fits(info, model.approximate_size_gb, model.recommended_vram_gb),
                           "selected": model.id == self.settings.ai_model_id})
            rows.append(status)
        order = {"low": 0, "medium": 1, "high": 2, "custom": 3}
        rows.sort(key=lambda r: (order.get(r["model"].tier, 9), r["model"].approximate_size_gb))
        return rows

    def _catalog_model(self, model_id: str) -> LocalModel:
        model = self.catalog().get(model_id)
        if model is None:
            raise ModelError("This model is not in the catalog.", details=model_id)
        return model

    def download_model(self, model_id: str, progress=None, cancel=None) -> Dict[str, Any]:
        model = self._catalog_model(model_id)
        self.registry.update(model.id, status="downloading")
        try:
            result = downloader.download_model(model, self.paths, progress, cancel)
        except BaseException:
            present = model.install_path(self.paths).is_file()
            self.registry.update(model.id, status="available" if present else "not_downloaded")
            raise
        self.registry.update(model.id, status="verified", sha256=result["sha256"], size=result["size"],
                             verified_at=now_iso(), hash_checked_against_source=result["hash_checked_against_source"],
                             source_url=result.get("source_url"), repository=result.get("repository"))
        log.info("Model %s downloaded and verified (source hash checked: %s)", model.id,
                 result["hash_checked_against_source"])
        return result

    def verify_model(self, model_id: str, cancel=None) -> Dict[str, Any]:
        model = self._catalog_model(model_id)
        path = model.install_path(self.paths)
        if not path.is_file():
            raise ModelError("The model file is missing.", hint="Download it again.", details=str(path))
        expected = {}
        known = self.registry.state(model.id).get("sha256")
        if known:
            expected["sha256"] = known
        result = downloader.verify_file(path, expected, cancel)
        self.registry.update(model.id, status="verified", sha256=result["sha256"], size=result["size"], verified_at=now_iso())
        return result

    def delete_model(self, model_id: str) -> None:
        model = self._catalog_model(model_id)
        if self.settings.ai_model_id == model.id:
            self.stop_ai()
        path = model.install_path(self.paths)
        if model.tier == "custom":
            self.registry.forget(model.id)  # the user's own file is never deleted
        else:
            for candidate in (path, path.with_name(path.name + ".part")):
                if self.paths.is_inside(candidate) and candidate.is_file():
                    candidate.unlink()
            self.registry.update(model.id, status="not_downloaded", sha256=None, verified_at=None, inference_ok=None)
        if self.settings.ai_model_id == model.id:
            self.settings.ai_model_id = ""
            self.settings.save(self.paths)

    def add_custom_model(self, path: Path) -> LocalModel:
        return register_custom_model(self.paths, self.registry, Path(path))

    def select_model(self, model_id: str) -> None:
        if model_id:
            self._catalog_model(model_id)
        if model_id != self.settings.ai_model_id:
            self.stop_ai()
        self.settings.ai_model_id = model_id
        self.settings.save(self.paths)

    def active_model(self) -> Optional[LocalModel]:
        if not self.settings.ai_model_id:
            return None
        model = self.catalog().get(self.settings.ai_model_id)
        return model if model and model.install_path(self.paths).is_file() else None

    def _make_llama_backend(self, model: LocalModel) -> InferenceBackend:
        return LlamaServerBackend(self.paths, model, server_path=self.runtime_path(),
                                  gpu_layers=self.settings.ai_gpu_layers, threads=self.settings.ai_threads)

    def backend(self) -> Optional[InferenceBackend]:
        model = self.active_model()
        if model is None:
            return None
        if self._backend is None or getattr(self._backend, "model_id", None) != model.id:
            self.stop_ai()
            self._backend = self.backend_factory(model)
        return self._backend

    def stop_ai(self) -> None:
        if self._backend is not None:
            try:
                self._backend.stop()
            finally:
                self._backend = None

    def test_model(self, model_id: str) -> Dict[str, Any]:
        self.select_model(model_id)
        backend = self.backend()
        if backend is None:
            raise ModelError("The model file is missing.", hint="Download the model first.")
        backend.start()
        result = run_inference_check(backend)
        self.registry.update(model_id, inference_ok=bool(result["ok"]), last_test=result)
        return result

    def model_cache_key(self, model: LocalModel) -> str:
        state = self.registry.state(model.id)
        return state.get("sha256") or f"{model.id}:{state.get('size') or ''}"

    # ------------------------------------------------------------ semantics
    def semantic_store(self) -> SemanticStore:
        return SemanticStore(self.require_project(), self.response_cache)

    def semantic_items(self, entity_type: str, include_short_cues: bool = False) -> List[tuple]:
        if entity_type == "track":
            return [(str(t["id"]), track_document(t)) for t in self.library_tracks()
                    if t["status"] == "ok" and not t.get("duplicate_of")]
        model = self.game_model()
        if model is None:
            return []
        return [(str(c.segment_id), cue_document(model, c)) for c in model.cues
                if include_short_cues or not self.is_short_cue(c)]

    @staticmethod
    def is_short_cue(cue) -> bool:
        return cue.is_transition or (cue.duration_ms is not None and cue.duration_ms < SHORT_CUE_MS)

    def analyze_semantics(self, use_ai: bool = True, progress=None, cancel=None) -> Dict[str, RunStats]:
        """Describe user tracks and game cues (rules always; local AI when a model is selected)."""

        store = self.semantic_store()
        backend = self.backend() if use_ai else None
        model_key = self.model_cache_key(self.active_model()) if backend else ""
        if backend is not None:
            if progress:
                progress("Starting the local AI model", 0, 0)
            backend.start()
        results = {}
        tracks = self.semantic_items("track")
        results["track"] = store.run("track", tracks, backend, model_key, progress, cancel)
        store.forget_missing("track", [k for k, _d in tracks])
        cues = self.semantic_items("cue", include_short_cues=True)
        results["cue"] = store.run("cue", cues, backend, model_key, progress, cancel)
        project = self.require_project()
        project.set("semantics_last_run", {"at": now_iso(), "model": backend.model_id if backend else "",
                                           "tracks": len(tracks), "cues": len(cues)})
        for kind, stats in results.items():
            if stats.llm_errors:
                project.add_event("warning", "ai", f"The local AI could not describe {stats.llm_errors} "
                                  f"{'tracks' if kind == 'track' else 'cues'}; rule-based descriptions are used for them.",
                                  "\n".join(stats.errors))
        return results

    def profiles(self, entity_type: str) -> Dict[str, EffectiveProfile]:
        return self.semantic_store().effective_all(entity_type)

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
        model = self.active_model()
        if model is not None:
            steps.append(WorkflowStep(4, "Choose AI model", OK, model.display_name))
        elif self.settings.ai_model_id:
            steps.append(WorkflowStep(4, "Choose AI model", WARN, "The selected model is not downloaded."))
        else:
            steps.append(WorkflowStep(4, "Choose AI model", WARN, "No model: rule-based descriptions only (optional)."))
        described = int(project.query_one("SELECT COUNT(DISTINCT entity_key) FROM semantic_profile WHERE entity_type='track'")[0])
        if counts.get("ok") and described:
            detail = f"{counts.get('ok', 0)} tracks scanned, {described} described"
            if counts.get("error"):
                detail += f", {counts['error']} unreadable"
            steps.append(WorkflowStep(5, "Analyze music", WARN if counts.get("error") else OK, detail))
        elif counts.get("ok"):
            steps.append(WorkflowStep(5, "Analyze music", WARN, f"{counts['ok']} tracks scanned; not described yet."))
        else:
            steps.append(WorkflowStep(5, "Analyze music", MISSING))
        if counts.get("missing"):
            status.warnings.append(f"{counts['missing']} previously scanned music files are no longer found.")
        for number, title in ((6, "Generate thematic matches"), (7, "Review / override matches"), (8, "Build mod"),
                              (9, "Validate output"), (10, "Export mod")):
            steps.append(WorkflowStep(number, title, UNAVAILABLE, "Not available in this version."))
        return status
