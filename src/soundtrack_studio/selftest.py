"""End-to-end self-test that needs no game files, no music and no network.

Exercises the Phase 1 workflow on synthetic data *through the real portable
directories* (projects/, data/, logs/, models/, output/): create project ->
select game -> import Analyzer DB -> verify game -> build music model ->
scan a FLAC library (nested, Unicode, broken and duplicate files) -> rescan
(cached) -> reopen project. It also asserts that the Analyzer DB, the game
folder and the music files were not modified.

Results go to ``logs/selftest.json`` (readable even from the windowed EXE).
Unless ``keep`` is set, the self-test project and its markers are removed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from . import __version__
from .app_paths import AppPaths
from .config import Settings

log = logging.getLogger(__name__)

SELFTEST_PROJECT = "__selftest__"
SELFTEST_BUILD_PROJECT = "__selftest_build__"
SELFTEST_MOD = "__selftest_mod__"


def _tree_fingerprint(root: Path) -> Dict[str, str]:
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        st = path.stat()
        out[path.relative_to(root).as_posix()] = f"{st.st_size}:{st.st_mtime_ns}:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    return out


def run_selftest(paths: AppPaths, keep: bool = False) -> Dict[str, Any]:
    from .analyzer_db import compat
    from .services import Studio
    from .testing.fixtures import make_fake_game, write_analyzer_db, write_test_flac

    checks: Dict[str, bool] = {}
    info: Dict[str, Any] = {}
    work = paths.temp / "selftest"
    shutil.rmtree(work, ignore_errors=True)
    project_folder = paths.projects / SELFTEST_PROJECT
    shutil.rmtree(project_folder, ignore_errors=True)
    output_marker = paths.output / SELFTEST_PROJECT / "build-check.txt"
    model_probe = paths.models / "custom" / ".selftest-probe"
    studio = Studio(paths, Settings.load(paths))
    leftovers = []  # imported snapshot + derived cache of the synthetic database
    try:
        game = work / "Crimson Desert"
        game_files = make_fake_game(game)
        analyzer_db = work / "Crimson Desert Analyzer" / "data" / "database" / "studio.sqlite3"
        write_analyzer_db(analyzer_db, game_root=game, game_files=game_files)
        music = work / "Music Library"
        write_test_flac(music / "Album A" / "01 Opening.flac", seconds=45, bpm=120,
                        tags={"TITLE": "Opening", "ARTIST": "Test Ensemble", "ALBUM": "Album A", "TRACKNUMBER": "1/2",
                              "DATE": "2024-05-01", "GENRE": "Orchestral", "COMPOSER": "Nobody"})
        write_test_flac(music / "Album A" / "CD2" / "02 Ünïcødé – 音楽.flac", seconds=40, bpm=90, tone_hz=330,
                        tags={"TITLE": "Ünïcødé – 音楽", "ARTIST": "Test Ensemble", "DISCNUMBER": "2"})
        write_test_flac(music / "Loose" / "no tags mono.flac", seconds=9, bpm=None, channels=1, subtype="PCM_24")
        shutil.copyfile(music / "Album A" / "01 Opening.flac", music / "Loose" / "duplicate of opening.flac")
        (music / "Loose" / "broken.flac").write_bytes(b"fLaC\x00\x00\x00\x22" + b"\x00" * 10)
        (music / "Loose" / "notes.txt").write_text("not audio", encoding="utf-8")

        before_db = _tree_fingerprint(analyzer_db.parent)
        before_game = _tree_fingerprint(game)
        before_music = _tree_fingerprint(music)

        project = studio.create_project(SELFTEST_PROJECT)
        checks["project_created_in_app_folder"] = paths.is_inside(project.folder)
        pre = studio.set_game_path(game, allow_inside_app=True)
        checks["game_recognised_before_analyzer"] = pre.status == compat.UNVERIFIABLE

        imported = studio.import_analyzer(analyzer_db)
        from .game_model.builder import cache_path

        if not imported.reused:
            leftovers += [imported.snapshot_path.parent, cache_path(paths, imported.sha256, 1)]
        checks["analyzer_imported"] = imported.report.ok and imported.schema_version == 1
        checks["analyzer_snapshot_in_app_folder"] = paths.is_inside(imported.snapshot_path)
        checks["analyzer_db_untouched"] = _tree_fingerprint(analyzer_db.parent) == before_db

        report = studio.check_game()
        checks["game_matches_analyzer_db"] = report.status == compat.MATCH
        model = studio.game_model()
        info["game_model"] = model.stats if model else None
        checks["music_structures_found"] = bool(model and model.stats["cues"] == 4 and model.stats["switches"] == 1)
        checks["transition_segment_detected"] = bool(model and any(c.is_transition for c in model.cues))

        studio.set_library_path(music)
        first = studio.scan_library()
        info["first_scan"] = {"found": first.files_found, "analyzed": first.analyzed, "errors": first.errors,
                              "duplicates": first.duplicates}
        checks["library_scanned"] = first.files_found == 5 and first.analyzed == 4 and first.errors == 1
        checks["duplicates_detected"] = first.duplicates == 1
        tracks = {t["rel_path"]: t for t in studio.library_tracks()}
        opening = tracks.get("Album A/01 Opening.flac", {})
        checks["tags_read"] = (opening.get("title") == "Opening" and opening.get("track_number") == 1
                               and opening.get("year") == 2024 and opening.get("composer") == "Nobody")
        checks["unicode_path_read"] = tracks.get("Album A/CD2/02 Ünïcødé – 音楽.flac", {}).get("status") == "ok"
        checks["tempo_estimated"] = abs(((opening.get("features") or {}).get("tempo_bpm") or 0) - 120) < 3
        described = studio.analyze_semantics(use_ai=False)
        profiles = studio.profiles("track")
        checks["music_described_rule_based"] = (described["track"].rules_done == 3 and described["cue"].rules_done == 4
                                                 and all(p.source == "rules" for p in profiles.values()))
        match_stats = studio.find_matches()
        matches = studio.match_store()
        accepted = matches.accept_all(min_confidence=0.0)
        checks["thematic_matches_proposed"] = match_stats["proposed"] >= 1 and accepted == match_stats["proposed"]
        info["matching"] = match_stats
        second = studio.scan_library()
        checks["rescan_uses_cache"] = second.unchanged == 4 and second.analyzed == 0
        checks["music_files_untouched"] = _tree_fingerprint(music) == before_music
        checks["game_files_untouched"] = _tree_fingerprint(game) == before_game

        folder = project.folder
        studio.close_project()
        reopened = studio.open_project(folder)
        checks["project_reopens"] = (reopened.get("music_library_path") == str(music.resolve())
                                     and studio.library_counts().get("ok") == 4
                                     and studio.game_model() is not None)

        output_marker.parent.mkdir(parents=True, exist_ok=True)
        output_marker.write_text("self-test output location check\n", encoding="utf-8")
        model_probe.write_text("self-test model folder check\n", encoding="utf-8")
        checks["output_and_models_in_app_folder"] = paths.is_inside(output_marker) and paths.is_inside(model_probe)
        # a real build from the Analyzer-built synthetic installation (encrypted/compressed archives, v150 banks)
        from .compiler.build import BuildSettings
        from .testing.fixtures import ANALYZER_FAKE_INSTALL_DB, extract_analyzer_fake_install

        shutil.rmtree(paths.projects / SELFTEST_BUILD_PROJECT, ignore_errors=True)
        studio.close_project()
        studio.create_project(SELFTEST_BUILD_PROJECT)
        build_game = extract_analyzer_fake_install(work / "Analyzer Install")
        before_build_game = _tree_fingerprint(build_game)
        imported = studio.import_analyzer(ANALYZER_FAKE_INSTALL_DB)
        if not imported.reused:
            leftovers += [imported.snapshot_path.parent, cache_path(paths, imported.sha256, 1)]
        studio.set_game_path(build_game, allow_inside_app=True)
        studio.set_library_path(music)
        studio.scan_library()
        tracks = {t["title"]: t["id"] for t in studio.library_tracks() if t["title"]}
        store = studio.match_store()
        store.choose("2001", tracks["Opening"])
        store.choose("2004", tracks["Opening"])
        result = studio.build_mod(BuildSettings(mod_name=SELFTEST_MOD, make_zip=False))
        info["build"] = {"files": result.report["files"], "validation_ok": result.validation.ok}
        checks["mod_built_and_validated"] = (result.validation.ok and paths.is_inside(result.output_dir)
                                             and len(result.report["files"]) == 2)
        checks["game_untouched_by_build"] = _tree_fingerprint(build_game) == before_build_game
        studio.settings.save(paths)
        checks["settings_in_app_folder"] = paths.settings_file.is_file() and paths.is_inside(paths.settings_file)
        checks["logs_in_app_folder"] = (paths.logs / "studio.log").is_file()
    except Exception as exc:  # the report must be written whatever happens
        log.exception("Self-test failed")
        checks["no_exception"] = False
        info["exception"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    finally:
        studio.close_project()
        if studio._cache is not None:
            studio._cache.close()
            studio._cache = None
        if not keep:
            shutil.rmtree(project_folder, ignore_errors=True)
            shutil.rmtree(paths.projects / SELFTEST_BUILD_PROJECT, ignore_errors=True)
            shutil.rmtree(paths.output / SELFTEST_MOD, ignore_errors=True)
            shutil.rmtree(output_marker.parent, ignore_errors=True)
            model_probe.unlink(missing_ok=True)
            shutil.rmtree(work, ignore_errors=True)
            for leftover in leftovers:
                shutil.rmtree(leftover, ignore_errors=True) if leftover.is_dir() else leftover.unlink(missing_ok=True)
            stored_names = {paths.to_stored(project_folder), paths.to_stored(paths.projects / SELFTEST_BUILD_PROJECT)}
            settings = studio.settings
            settings.recent_projects = [p for p in settings.recent_projects if p not in stored_names]
            if settings.last_project in stored_names:
                settings.last_project = settings.recent_projects[0] if settings.recent_projects else ""
            settings.save(paths)

    result = {"version": __version__, "root": str(paths.root), "frozen": bool(getattr(sys, "frozen", False)),
              "ok": bool(checks) and all(checks.values()), "checks": checks, "info": info}
    paths.logs.mkdir(parents=True, exist_ok=True)
    (paths.logs / "selftest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Self-test %s: %s", "passed" if result["ok"] else "FAILED",
             {k: v for k, v in checks.items() if not v} or "all checks passed")
    return result
