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
import contextlib
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

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
from .gameaudio.analysis import GameAudioAnalyzer, GameAudioCache, SourceResult
from .gameaudio.decoder import find_vgmstream
from .game_model.model import GameMusicModel
from .library.cache import AnalysisCache
from .library.scanner import LibraryScanner, ScanProgress, ScanStats
from .project.store import Project, ProjectInfo, list_projects, now_iso
from .compiler.build import BuildResult, BuildSettings, ModBuilder
from .matching.engine import Matcher, MatchSettings, score_candidate, track_infos
from .matching.store import MatchStore
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
        self._game_audio_cache: Optional[GameAudioCache] = None
        self._listening_cache = None
        self._clap = None
        self._clap_key = ""
        self._prompt_bank = None
        self._listening_epoch = 0          # bumped whenever what was heard (or the library) changes
        self._calibration = None           # (signature, Calibration)
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
        if self._game_audio_cache is not None:
            self._game_audio_cache.close()
            self._game_audio_cache = None
        self.release_listening_model()
        if self._listening_cache is not None:
            self._listening_cache.close()
            self._listening_cache = None
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

    @property
    def game_audio_cache(self) -> GameAudioCache:
        if self._game_audio_cache is None:
            self._game_audio_cache = GameAudioCache(self.paths.cache / "game_audio.sqlite3")
        return self._game_audio_cache

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
        self._listening_epoch += 1
        if stats.errors:
            project.add_event("warning", "library", f"{stats.errors} music files could not be read.",
                              "\n".join(stats.error_samples))
        return stats

    def library_tracks(self) -> List[Dict[str, Any]]:
        project = self.require_project()
        rows = project.query(
            "SELECT t.id, t.rel_path, t.status, t.error, t.duplicate_of, t.size, t.identity, r.path AS root, m.*, f.features_json"
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


    # ----------------------------------------------------------- game audio
    def game_audio_available(self) -> Tuple[bool, str]:
        """Whether the game's music can be decoded now, and if not, why (plain language)."""

        if self.project is None:
            return False, "Open a project first."
        ref = self.project.active_analyzer()
        if not ref:
            return False, "Import an Analyzer database first."
        game_path = self.project.get("game_path")
        if not game_path or not Path(game_path).is_dir():
            return False, "Select your Crimson Desert folder first."
        if find_vgmstream(self.paths, self.settings.vgmstream_path) is None:
            return False, "The audio decoder (vgmstream) is missing from the 'runtime\\vgmstream' folder."
        latest = self.project.last_game_check()
        if latest and latest.get("status") in (compat.MISMATCH, compat.NOT_GAME):
            return False, "The Crimson Desert folder does not match the Analyzer database."
        return True, ""

    @contextlib.contextmanager
    def _game_audio(self) -> Iterator[GameAudioAnalyzer]:
        from .analyzer_db.importer import open_snapshot

        ok, reason = self.game_audio_available()
        if not ok:
            raise GameInstallError(reason)
        project = self.require_project()
        ref = project.active_analyzer()
        conn = open_snapshot(self.paths.from_stored(ref["snapshot_path"]))
        try:
            yield GameAudioAnalyzer(self.paths, Path(project.get("game_path")), conn, ref["installation_id"],
                                    self.game_audio_cache, find_vgmstream(self.paths, self.settings.vgmstream_path))
        finally:
            conn.close()

    def cue_sources(self, cue) -> List[int]:
        """The audio of a cue's primary track (the music a replacement would take the place of)."""

        from .compiler.plan import primary_track

        model = self.game_model()
        segment = model.nodes.get(cue.segment_id) if model else None
        primary = primary_track(model, segment) if segment is not None else None
        if primary is None:
            return list(cue.source_ids)
        return [s for s in primary.source_ids] or list(cue.source_ids)

    def game_music_sources(self) -> List[int]:
        model = self.game_model()
        if model is None:
            return []
        out: List[int] = []
        for cue in model.cues:
            out.extend(self.cue_sources(cue))
        return list(dict.fromkeys(out))

    def analyze_game_audio(self, progress=None, cancel=None, retry_errors: bool = False,
                           listen: bool = True) -> Dict[str, Any]:
        """Decode (read-only, in temp/) and measure the game's music; listen to it when a listening model is on."""

        sources = self.game_music_sources()
        listener, listen_key = (self.listening_callable(progress) if listen else (None, ""))
        with self._game_audio() as analyzer:
            results = analyzer.analyze(sources, listener, listen_key, progress, cancel, retry_errors)
        summary = self._game_audio_summary(results)
        self._listening_epoch += 1
        project = self.require_project()
        project.set("game_audio_last_run", {"at": now_iso(), **summary})
        if summary["errors"]:
            examples = [r.error for r in results.values() if r.status == "error"][:5]
            project.add_event("warning", "game_audio", f"{summary['errors']} game music files could not be decoded; "
                              "they are described from their names only.", "\n".join(examples))
        return summary

    @staticmethod
    def _game_audio_summary(results: Dict[int, SourceResult]) -> Dict[str, Any]:
        return {"sources": len(results), "ok": sum(1 for r in results.values() if r.status == "ok"),
                "errors": sum(1 for r in results.values() if r.status == "error"),
                "missing": sum(1 for r in results.values() if r.status == "missing"),
                "listened": sum(1 for r in results.values() if r.listening)}

    def game_audio_results(self) -> Dict[int, SourceResult]:
        """Stored results only; never reads the game files. Empty when game audio is unavailable."""

        if not self.game_audio_available()[0]:
            return {}
        listen_key = self.listening_key()
        with self._game_audio() as analyzer:
            return {sid: analyzer.cached(sid, listen_key) for sid in self.game_music_sources()}

    def game_audio_check(self, count: int = 6) -> Dict[str, Any]:
        """Decode a few game music files without keeping anything; for the 'Test decoding' button."""

        from .gameaudio.decoder import version

        with self._game_audio() as analyzer:
            report = analyzer.decode_check(self.game_music_sources(), count)
            report["decoder"] = str(analyzer.vgmstream)
            try:
                report["decoder_version"] = version(analyzer.vgmstream)
            except Exception as exc:  # noqa: BLE001 - informational only
                report["decoder_version"] = f"unknown ({exc})"
        (self.paths.logs / "game_audio_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def cue_audio(self, cue, results: Dict[int, SourceResult]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """(measurements, listening) for a cue from its primary-track sources; the longest decoded source wins."""

        best: Optional[SourceResult] = None
        for sid in self.cue_sources(cue):
            r = results.get(sid)
            if r is not None and r.status == "ok" and (best is None or (r.decoded_s or 0) > (best.decoded_s or 0)):
                best = r
        if best is None:
            return None, None
        return best.features or None, best.listening or None

    # ------------------------------------------------------------ listening
    @property
    def listening_cache(self):
        from .listening.listen import ListeningCache

        if self._listening_cache is None:
            self._listening_cache = ListeningCache(self.paths.cache / "listening.sqlite3")
        return self._listening_cache

    def listening_models(self) -> List[Dict[str, Any]]:
        from .listening.catalog import LISTENING_MODELS, listening_status

        out = []
        for m in LISTENING_MODELS:
            status = listening_status(self.paths, m, self.registry.state(m.id))
            out.append({"model": m, "active": self.settings.listening_model_id == m.id, **status})
        return out

    def download_listening_model(self, model_id: str, progress=None, cancel=None) -> Dict[str, Any]:
        from .listening.catalog import get_listening_model, validate_small_file

        model = get_listening_model(model_id)
        if model is None:
            raise ModelError(f"Unknown listening model: {model_id}")
        folder = model.install_dir(self.paths)
        results = {}
        for i, f in enumerate(model.files, 1):
            target = folder / f.path
            if target.is_file() and self.registry.state(model.id).get("files", {}).get(f.path):
                continue
            label = f"Downloading listening model ({i}/{len(model.files)})"
            source = {"url": model.url(f), "sha256": f.sha256, "size": f.size, "repository": model.repository}
            results[f.path] = downloader.download_file(source, target, self.paths, progress, cancel, label, magic=None)
            if f.sha256 is None:
                try:
                    validate_small_file(target, f.role)
                except (OSError, ValueError) as exc:
                    target.unlink(missing_ok=True)
                    raise ModelError("A downloaded listening model file is not valid and was removed.",
                                     hint="Try the download again.", details=f"{f.path}: {exc}") from exc
            files = dict(self.registry.state(model.id).get("files", {}))
            files[f.path] = results[f.path]["sha256"]
            self.registry.update(model.id, files=files)
        self.registry.update(model.id, verified=True, installed_at=now_iso())
        return {"model": model.id, "downloaded": sorted(results)}

    def delete_listening_model(self, model_id: str) -> None:
        import shutil
        from .listening.catalog import get_listening_model

        model = get_listening_model(model_id)
        if model is None:
            return
        if self.settings.listening_model_id == model_id:
            self.select_listening_model("")
        folder = model.install_dir(self.paths)
        if folder.is_dir() and self.paths.is_inside(folder):
            shutil.rmtree(folder)
        self.registry.forget(model.id)

    def select_listening_model(self, model_id: str) -> None:
        self.release_listening_model()
        self.settings.listening_model_id = model_id
        self.settings.save(self.paths)

    def active_listening_model(self):
        from .listening.catalog import get_listening_model, listening_status

        model = get_listening_model(self.settings.listening_model_id) if self.settings.listening_model_id else None
        if model is None or not listening_status(self.paths, model, self.registry.state(model.id))["installed"]:
            return None
        return model

    def listening_key(self) -> str:
        """Cache key of the active listening model ("" = no listening model)."""

        from .listening.listen import LISTEN_VERSION

        model = self.active_listening_model()
        return f"{model.id}@{model.revision[:12]}:v{LISTEN_VERSION}" if model else ""

    def release_listening_model(self) -> None:
        self._clap = None
        self._clap_key = ""
        self._prompt_bank = None
        self._listening_epoch += 1

    def _load_clap(self, progress=None):
        from .listening.clap import load_model

        model = self.active_listening_model()
        if model is None:
            return None
        key = self.listening_key()
        if self._clap is None or self._clap_key != key:
            if progress:
                progress("Loading the listening model", 0, 0)
            self._clap = load_model(model.install_dir(self.paths), model.file_map(), self.settings.listening_device)
            self._clap_key = key
            self._prompt_bank = self.listening_cache.ensure_prompts(key, self._clap)
        return self._clap

    def listening_callable(self, progress=None):
        """(listener, key) for the active listening model, or (None, "")."""

        from .listening.listen import listen_file

        clap = self._load_clap(progress)
        if clap is None:
            return None, ""
        key = self._clap_key
        return (lambda wav: listen_file(clap, wav, key)), key

    def test_listening_model(self, model_id: str) -> Dict[str, Any]:
        """Load the model, listen to generated test audio and check the results are sane."""

        import time

        import numpy as np

        from .listening.catalog import get_listening_model
        from .listening.clap import load_model, version_info

        model = get_listening_model(model_id)
        if model is None:
            raise ModelError(f"Unknown listening model: {model_id}")
        started = time.monotonic()
        clap = load_model(model.install_dir(self.paths), model.file_map(), self.settings.listening_device)
        t = np.arange(48000 * 10) / 48000
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        noise = (0.1 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32)
        emb = clap.embed_audio([tone, noise])
        text = clap.embed_text(["a sine tone", "white noise"])
        sims = emb @ text.T
        ok = bool(np.all(np.isfinite(emb)) and emb.shape[0] == 2 and emb.shape[1] == text.shape[1])
        result = {"ok": ok, "provider": clap.provider, "seconds": round(time.monotonic() - started, 1),
                  "dimension": int(emb.shape[1]), "tone_prefers_tone_prompt": bool(sims[0, 0] > sims[0, 1]),
                  "noise_prefers_noise_prompt": bool(sims[1, 1] > sims[1, 0]), **version_info()}
        self.registry.update(model_id, last_test=result)
        (self.paths.logs / "listening_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def listen_to_library(self, progress=None, cancel=None) -> Dict[str, int]:
        from .errors import OperationCancelled, StudioError
        from .listening.listen import listen_file

        listener, key = self.listening_callable(progress)
        if listener is None:
            return {"tracks": 0, "listened": 0, "cached": 0, "errors": 0}
        tracks = [t for t in self.library_tracks() if t["status"] == "ok" and not t.get("duplicate_of")]
        stats = {"tracks": len(tracks), "listened": 0, "cached": 0, "errors": 0}
        errors = []
        for i, t in enumerate(tracks):
            if cancel and cancel():
                raise OperationCancelled()
            if progress:
                progress("Listening to your music", i, len(tracks))
            identity = t.get("identity") or f"{t['root']}/{t['rel_path']}:{t.get('size')}"
            if self.listening_cache.get_track(identity, key):
                stats["cached"] += 1
                continue
            try:
                result = listen_file(self._clap, Path(t["root"]) / t["rel_path"], key)
                self.listening_cache.put_track(identity, key, result)
                stats["listened"] += 1
            except StudioError as exc:
                stats["errors"] += 1
                errors.append(f"{t['rel_path']}: {exc.message} {exc.details}".strip())
        if progress:
            progress("Listening to your music", len(tracks), len(tracks))
        if errors:
            self.require_project().add_event("warning", "listening", f"{len(errors)} tracks could not be listened to.",
                                             "\n".join(errors[:20]))
        self._listening_epoch += 1
        return stats

    def track_listening(self) -> Dict[int, Dict[str, Any]]:
        key = self.listening_key()
        if not key:
            return {}
        out = {}
        for t in self.library_tracks():
            identity = t.get("identity") or f"{t['root']}/{t['rel_path']}:{t.get('size')}"
            result = self.listening_cache.get_track(identity, key)
            if result:
                out[t["id"]] = result
        return out

    def _prompt_bank_for(self, key: str):
        bank = self._prompt_bank if (self._prompt_bank is not None and key == self._clap_key) else None
        if bank is None:
            bank = self.listening_cache.prompt_bank(key)
            if not bank.complete():
                try:
                    if key == self.listening_key() and self._load_clap() is not None:
                        bank = self._prompt_bank
                except Exception as exc:  # noqa: BLE001 - summaries are optional
                    log.warning("Listening prompts unavailable: %s", exc)
                    return None
        return bank if bank is not None and bank.complete() else None

    def calibration(self):
        """Per-word baselines over everything analysed (your tracks plus the game's music); None without listening."""

        from .listening.calibration import Calibration

        key = self.listening_key()
        if not key or self.project is None:
            return None
        signature = (key, self._listening_epoch, self.project.folder)
        if self._calibration is not None and self._calibration[0] == signature:
            return self._calibration[1]
        bank = self._prompt_bank_for(key)
        if bank is None:
            return None
        model = self.game_model() if self.project.active_analyzer() else None
        tracks, cues = self.sound_embeddings(model) if model is not None else (
            {tid: e for tid, e in self._track_embeddings(key).items()}, {})
        calibration = Calibration(bank, list(tracks.values()) + list(cues.values()))
        self._calibration = (signature, calibration)
        return calibration

    def _track_embeddings(self, key: str) -> Dict[int, Any]:
        from .listening.listen import embedding_of

        return {tid: e for tid, r in self.track_listening().items()
                if r.get("model") == key and (e := embedding_of(r)) is not None}

    def heard_summary(self, listening: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """What the listening model heard: vocals (verdict + 0-100) and standout tags with 0-100 scores, or None."""

        if not listening or not listening.get("embedding"):
            return None
        calibration = self.calibration()
        return calibration.summary(listening) if calibration is not None else None

    # ------------------------------------------------------------ semantics
    def semantic_store(self) -> SemanticStore:
        return SemanticStore(self.require_project(), self.response_cache)

    def semantic_items(self, entity_type: str, include_short_cues: bool = False) -> List[tuple]:
        if entity_type == "track":
            heard = self.track_listening()
            return [(str(t["id"]), track_document(t, self.heard_summary(heard.get(t["id"]))))
                    for t in self.library_tracks() if t["status"] == "ok" and not t.get("duplicate_of")]
        model = self.game_model()
        if model is None:
            return []
        audio = self.game_audio_results()
        items = []
        for c in model.cues:
            if include_short_cues or not self.is_short_cue(c):
                measured, heard = self.cue_audio(c, audio) if audio else (None, None)
                items.append((str(c.segment_id), cue_document(model, c, measured, self.heard_summary(heard))))
        return items


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
        if self.settings.analyze_game_audio and self.game_audio_available()[0]:
            try:
                self.analyze_game_audio(progress, cancel)
            except (GameInstallError, AnalyzerDbError) as exc:
                self.require_project().add_event("warning", "game_audio", f"The game's music was not analysed: {exc.message}",
                                                 getattr(exc, "details", ""))
        if self.active_listening_model() is not None:
            self.listen_to_library(progress, cancel)
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

    # -------------------------------------------------------------- matching
    def match_store(self) -> MatchStore:
        return MatchStore(self.require_project())

    def sound_embeddings(self, model) -> Tuple[Dict[int, Any], Dict[str, Any]]:
        """Listening-model embeddings of tracks and cues heard by the *same* model ({} when not listened to)."""

        from .listening.listen import embedding_of

        key = self.listening_key()
        if not key:
            return {}, {}
        tracks = {tid: e for tid, r in self.track_listening().items()
                  if r.get("model") == key and (e := embedding_of(r)) is not None}
        cues: Dict[str, Any] = {}
        audio = self.game_audio_results()
        for cue in model.cues if audio else []:
            _measured, heard = self.cue_audio(cue, audio)
            if heard and heard.get("model") == key and (e := embedding_of(heard)) is not None:
                cues[str(cue.segment_id)] = e
        return tracks, cues

    def find_matches(self, settings: Optional[MatchSettings] = None, progress=None, cancel=None) -> Dict[str, Any]:
        settings = settings or MatchSettings()
        model = self.game_model()
        if model is None:
            raise AnalyzerDbError("Import an Analyzer database first, so the Studio knows the game's music.")
        tracks = self.library_tracks()
        if not any(t["status"] == "ok" for t in tracks):
            raise LibraryError("Scan a music folder first, so there is music to match.")
        store = self.semantic_store()
        track_profiles = store.effective_all("track")
        cue_profiles = store.effective_all("cue")
        if not track_profiles or not cue_profiles or len(track_profiles) < len(self.semantic_items("track")):
            # make sure everything has at least a rule-based description (cheap, no model)
            self.analyze_semantics(use_ai=False, progress=progress, cancel=cancel)
            track_profiles = store.effective_all("track")
            cue_profiles = store.effective_all("cue")
        use_standout = settings.mode == "standout" and self.active_listening_model() is not None
        track_emb, cue_emb = self.sound_embeddings(model) if (settings.use_sound or use_standout) else ({}, {})
        calibration = self.calibration() if use_standout else None
        track_standouts = {tid: calibration.standout_vector(e) for tid, e in track_emb.items()} if calibration else {}
        cue_standouts = {k: calibration.standout_vector(e) for k, e in cue_emb.items()} if calibration else {}
        infos = track_infos(tracks, track_profiles, track_emb if settings.use_sound else {}, track_standouts)
        backend = self.backend() if settings.use_ai else None
        if backend is not None:
            if progress:
                progress("Starting the local AI model", 0, 0)
            backend.start()
        cue_docs = {str(c.segment_id): cue_document(model, c) for c in model.cues} if backend else {}
        matches = self.match_store()
        matcher = Matcher(model, cue_profiles, infos, settings, rejected=matches.rejected_map(),
                          fixed=matches.fixed_map(), backend=backend, cue_docs=cue_docs,
                          cue_embeddings=cue_emb if settings.use_sound else {}, cue_standouts=cue_standouts,
                          calibration=calibration)
        results = matcher.run(progress, cancel)
        stats = {"cues": len(results), "proposed": sum(1 for r in results.values() if r.candidates),
                 "unmatched": sum(1 for r in results.values() if not r.candidates), "tracks": len(infos),
                 "ai_errors": len(matcher.ai_errors),
                 "compared_by_sound": (sum(1 for k in cue_emb if k in results) if len(track_emb) >= 2 else 0)
                 if settings.use_sound else 0,
                 "compared_by_standout": sum(1 for k in cue_standouts if k in results) if track_standouts else 0,
                 "mode": "standout" if (cue_standouts and track_standouts) else "legacy"}
        matches.save_run(results, settings, backend.model_id if backend else "", stats)
        if matcher.ai_errors:
            self.require_project().add_event("warning", "ai", f"The local AI could not judge {len(matcher.ai_errors)} "
                                             "cues; the deterministic ranking was used for them.",
                                             "\n".join(matcher.ai_errors[:10]))
        log.info("Matching finished: %s", stats)
        return stats

    def rank_tracks_for_cue(self, cue_key: str) -> List[Dict[str, Any]]:
        """Every usable track scored against one cue (for the manual picker)."""

        model = self.game_model()
        cue = next((c for c in model.cues if str(c.segment_id) == str(cue_key)), None) if model else None
        if cue is None:
            return []
        store = self.semantic_store()
        cue_eff = store.effective("cue", str(cue_key))
        infos = track_infos(self.library_tracks(), store.effective_all("track"))
        rows = []
        for info in infos:
            cand = score_candidate(cue, cue_eff.profile, info) if cue_eff else None
            rows.append({"track": info, "candidate": cand})
        rows.sort(key=lambda r: -(r["candidate"].score if r["candidate"] else -1))
        return rows

    def matching_rows(self, include_short: bool = False) -> List[Dict[str, Any]]:
        model = self.game_model()
        if model is None:
            return []
        matches = self.match_store()
        proposals = matches.proposals()
        decisions = matches.decisions()
        skipped = matches.skipped()
        rows = []
        for cue in model.cues:
            key = str(cue.segment_id)
            if not include_short and self.is_short_cue(cue) and key not in decisions:
                continue
            decision = decisions.get(key)
            props = proposals.get(key, [])
            if decision is None:
                status = "proposed" if props else "unmatched"
            else:
                status = {"accept": "accepted", "manual": "chosen", "reject": "rejected",
                          "keep_original": "keep original"}[decision.action]
            rows.append({"cue": cue, "key": key, "proposals": props, "decision": decision, "status": status,
                         "skipped_reason": skipped.get(key, "")})
        return rows

    # ----------------------------------------------------------------- build
    def build_settings(self) -> BuildSettings:
        project = self.require_project()
        saved = project.get("build_settings") or {}
        settings = BuildSettings(mod_name=f"{project.name} Soundtrack")
        for key, value in saved.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        return settings

    def save_build_settings(self, settings: BuildSettings) -> None:
        from dataclasses import asdict

        self.require_project().set("build_settings", asdict(settings))

    def build_mod(self, settings: Optional[BuildSettings] = None, progress=None, cancel=None) -> BuildResult:
        import json as _json
        from dataclasses import asdict

        from .errors import OperationCancelled

        project = self.require_project()
        settings = settings or self.build_settings()
        self.save_build_settings(settings)
        mapping = self.match_store().final_mapping()
        game_path = project.get("game_path")
        if not game_path:
            raise GameInstallError("Select your Crimson Desert folder on the Home page first.")
        report = self.check_game()
        if not report.usable:
            raise GameInstallError("The Crimson Desert installation does not match the Analyzer database, so the "
                                   "mod cannot be built safely.", hint=report.summary())
        ref = project.active_analyzer()
        model = self.game_model()
        tracks = {t["id"]: Path(t["root"]) / t["rel_path"] for t in self.library_tracks()}
        build_id = project.execute("INSERT INTO build(started_at, status, settings_json) VALUES (?,?,?)",
                                   (now_iso(), "running", _json.dumps(asdict(settings)))).lastrowid
        snapshot = self.paths.from_stored(ref["snapshot_path"])
        from .analyzer_db.importer import open_snapshot

        conn = open_snapshot(snapshot)
        try:
            builder = ModBuilder(self.paths, Path(game_path), conn, ref["installation_id"], model, tracks, settings)
            result = builder.build(mapping, progress, cancel)
        except OperationCancelled:
            project.execute("UPDATE build SET status='cancelled', finished_at=? WHERE id=?", (now_iso(), build_id))
            raise
        except Exception as exc:
            message = getattr(exc, "message", str(exc))
            project.execute("UPDATE build SET status='failed', finished_at=?, error=? WHERE id=?",
                            (now_iso(), message[:500], build_id))
            project.add_event("error", "build", f"Build failed: {message}", getattr(exc, "details", ""))
            raise
        finally:
            conn.close()
        summary = {"cues": len(result.report["cues"]), "files": len(result.report["files"]),
                   "warnings": result.warnings, "validation": asdict(result.validation),
                   "size_bytes": sum(p.stat().st_size for p in result.output_dir.rglob("*") if p.is_file())}
        project.execute("UPDATE build SET status='completed', finished_at=?, output_path=?, zip_path=?, summary_json=?"
                        " WHERE id=?", (now_iso(), self.paths.to_stored(result.output_dir),
                                        self.paths.to_stored(result.zip_path) if result.zip_path else None,
                                        _json.dumps(summary), build_id))
        project.add_event("info", "build", f"Mod built: {result.output_dir.name} ({summary['cues']} cues replaced).")
        return result

    def builds(self, limit: int = 20) -> List[Dict[str, Any]]:
        import json as _json

        rows = []
        for r in self.require_project().query("SELECT * FROM build ORDER BY id DESC LIMIT ?", (limit,)):
            item = dict(r)
            item["summary"] = _json.loads(item.pop("summary_json") or "{}")
            item["settings"] = _json.loads(item.pop("settings_json") or "{}")
            item["output_dir"] = self.paths.from_stored(item["output_path"]) if item["output_path"] else None
            item["zip"] = self.paths.from_stored(item["zip_path"]) if item["zip_path"] else None
            rows.append(item)
        return rows

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
        matches = self.match_store()
        run = matches.last_run()
        if run is None:
            steps.append(WorkflowStep(6, "Generate thematic matches", MISSING))
        else:
            steps.append(WorkflowStep(6, "Generate thematic matches", OK,
                                      f"{run['stats'].get('proposed', 0)} cues have proposals ({run['created_at'][:16]})"))
        counts = matches.status_counts()
        chosen = counts["accepted"] + counts["manual"]
        if chosen:
            detail = f"{chosen} replacements confirmed, {counts['proposed']} proposals not reviewed"
            steps.append(WorkflowStep(7, "Review / override matches", OK, detail))
        else:
            steps.append(WorkflowStep(7, "Review / override matches", MISSING if run is None else WARN,
                                      "" if run is None else "No replacement confirmed yet."))
        builds = self.builds(1)
        last = builds[0] if builds else None
        if last is None:
            steps += [WorkflowStep(8, "Build mod", MISSING), WorkflowStep(9, "Validate output", MISSING),
                      WorkflowStep(10, "Export mod", MISSING)]
        elif last["status"] == "completed":
            present = bool(last["output_dir"] and last["output_dir"].is_dir())
            steps.append(WorkflowStep(8, "Build mod", OK, f"{last['summary'].get('cues', 0)} cues, {last['finished_at'][:16]}"))
            steps.append(WorkflowStep(9, "Validate output", OK, "All checks passed."))
            steps.append(WorkflowStep(10, "Export mod", OK if present else WARN,
                                      str(last["zip"] or last["output_dir"]) if present else "Output folder was removed."))
            status.last_build = last["finished_at"]
        else:
            steps.append(WorkflowStep(8, "Build mod", WARN, f"Last build {last['status']}: {last.get('error') or ''}"))
            steps += [WorkflowStep(9, "Validate output", MISSING), WorkflowStep(10, "Export mod", MISSING)]
        return status
