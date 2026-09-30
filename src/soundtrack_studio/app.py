"""Entry point.

    CrimsonSoundtrackStudio.exe                       start the GUI
    CrimsonSoundtrackStudio.exe --selftest [--keep]   end-to-end check on synthetic data (logs/selftest.json)
    CrimsonSoundtrackStudio.exe --portability-check   verify nothing is written outside the app folder
    CrimsonSoundtrackStudio.exe --print-paths         show the portable directory layout
    CrimsonSoundtrackStudio.exe --print-state         settings/projects/models/caches/logs seen by this copy
    CrimsonSoundtrackStudio.exe --ai-check MODEL.gguf start the bundled AI runtime with a model and test it
    CrimsonSoundtrackStudio.exe --listen-check HOME   load the listening model found in HOME/models/listening and test it
    CrimsonSoundtrackStudio.exe --listen-file FILE    listen to one audio file with the installed listening model and
                                                      print the vocal scores per excerpt (writes logs/listen_file.json)
    CrimsonSoundtrackStudio.exe --smoke-gui           open the main window briefly and exit (CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import APP_DISPLAY_NAME, __version__
from .app_paths import AppPaths, get_paths


def _print(text: str) -> None:
    if sys.stdout is not None:  # windowed EXE has no console
        print(text)


def _state(paths: AppPaths) -> dict:
    from .config import Settings
    from .project.store import list_projects

    settings = Settings.load(paths)

    def files(folder):
        return sorted(p.relative_to(paths.root).as_posix() for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else []

    return {
        "version": __version__,
        "root": str(paths.root),
        "settings_file": paths.settings_file.is_file(),
        "last_project": settings.last_project,
        "last_project_resolves": bool(paths.from_stored(settings.last_project)
                                      and paths.from_stored(settings.last_project).is_dir()),
        "projects": [{"name": p.name, "folder": paths.to_stored(p.folder), "error": p.error} for p in list_projects(paths)],
        "models": files(paths.models),
        "databases": files(paths.databases),
        "cache": files(paths.cache)[:200],
        "logs": files(paths.logs),
        "output": files(paths.output),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="CrimsonSoundtrackStudio", description=APP_DISPLAY_NAME)
    parser.add_argument("--version", action="version", version=f"{APP_DISPLAY_NAME} {__version__}")
    parser.add_argument("--selftest", action="store_true", help="run the end-to-end self-test")
    parser.add_argument("--keep", action="store_true", help="keep the self-test project and markers")
    parser.add_argument("--portability-check", action="store_true", help="verify portable data storage")
    parser.add_argument("--print-paths", action="store_true")
    parser.add_argument("--print-state", action="store_true")
    parser.add_argument("--ai-check", metavar="GGUF", help="load a GGUF model with the bundled runtime and test it")
    parser.add_argument("--listen-check", metavar="HOME", help="test the listening model installed under HOME (CI)")
    parser.add_argument("--listen-file", metavar="FILE", help="listen to one file and print what the model heard")
    parser.add_argument("--smoke-gui", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    paths = get_paths()
    problem = paths.check_writable()
    if problem:
        return _fatal(problem)
    paths.ensure()
    paths.clean_temp()
    from .environment import configure_process_environment

    configure_process_environment(paths)  # before Qt or any other library is imported
    from .config import Settings
    from .logging_setup import setup_logging, shutdown_logging

    settings = Settings.load(paths)
    cli = (args.selftest or args.portability_check or args.print_paths or args.print_state or args.ai_check
           or args.listen_check or args.listen_file)
    setup_logging(paths, settings.log_level, console=cli and sys.stderr is not None)
    try:
        if args.print_paths:
            _print(json.dumps({d.relative_to(paths.root).as_posix() or ".": str(d) for d in [paths.root, *paths.all_dirs()]},
                              indent=2))
            return 0
        if args.print_state:
            text = json.dumps(_state(paths), indent=2, ensure_ascii=False)
            (paths.logs / "state.json").write_text(text, encoding="utf-8")  # readable from the windowed EXE
            _print(text)
            return 0
        if args.selftest:
            from .selftest import run_selftest

            result = run_selftest(paths, keep=args.keep)
            _print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["ok"] else 1
        if args.ai_check:
            result = _ai_check(paths, settings, args.ai_check)
            text = json.dumps(result, indent=2, ensure_ascii=False)
            (paths.logs / "ai_check.json").write_text(text, encoding="utf-8")
            _print(text)
            return 0 if result["ok"] else 1
        if args.listen_check:
            try:
                result = _listen_check(paths, settings, args.listen_check)
            except Exception as exc:  # noqa: BLE001 - reported in the JSON result; a windowed EXE must not block
                import traceback

                result = {"ok": False, "error": f"{getattr(exc, 'message', exc)} {getattr(exc, 'details', '')}".strip(),
                          "traceback": traceback.format_exc()[-4000:]}
            text = json.dumps(result, indent=2, ensure_ascii=False)
            (paths.logs / "listen_check.json").write_text(text, encoding="utf-8")
            _print(text)
            return 0 if result.get("ok") else 1
        if args.listen_file:
            try:
                result = _listen_file(paths, settings, args.listen_file)
            except Exception as exc:  # noqa: BLE001 - reported in the JSON result
                result = {"ok": False, "error": f"{getattr(exc, 'message', exc)} {getattr(exc, 'details', '')}".strip()}
            text = json.dumps(result, indent=2, ensure_ascii=False)
            (paths.logs / "listen_file.json").write_text(text, encoding="utf-8")
            _print(text)
            return 0 if result.get("ok") else 1
        if args.portability_check:
            from .portability import run_portability_check

            result = run_portability_check(paths)
            _print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["ok"] else 1
        from .ui.main_window import run_gui

        return run_gui(paths, settings, smoke=args.smoke_gui)
    finally:
        shutdown_logging()


def _listen_check(paths: AppPaths, settings, home: str) -> dict:
    """Load the listening model from another portable root's models folder with the bundled ONNX Runtime."""

    from pathlib import Path

    from .listening.catalog import CLAP_MUSIC_SPEECH
    from .services import Studio

    source = AppPaths(Path(home))
    studio = Studio(paths, settings)
    try:
        model_dir = CLAP_MUSIC_SPEECH.install_dir(source)
        if not model_dir.is_dir():
            return {"ok": False, "error": f"no listening model in {model_dir}"}
        import time

        import numpy as np

        from .listening.clap import load_model, version_info
        from .listening.listen import PromptBank, all_prompts

        started = time.monotonic()
        clap = load_model(model_dir, CLAP_MUSIC_SPEECH.file_map(), settings.listening_device)
        t = np.arange(480000) / 48000
        emb = clap.embed_audio([(0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)])
        prompts = all_prompts()
        bank = PromptBank(dict(zip(prompts, clap.embed_text(prompts))))
        summary = bank.summary({"embedding": emb[0].tolist()})
        return {"ok": bool(np.all(np.isfinite(emb)) and summary is not None), "provider": clap.provider,
                "seconds": round(time.monotonic() - started, 1), "summary_of_a_sine_tone": summary, **version_info()}
    finally:
        studio.shutdown()


def _listen_file(paths: AppPaths, settings, file: str) -> dict:
    """Listen to one file with the listening model installed in this program's models folder."""

    from pathlib import Path

    from .listening.catalog import CLAP_MUSIC_SPEECH, listening_status
    from .listening.clap import load_model
    from .listening.listen import LISTEN_VERSION, _vocal_decision, excerpt_embeddings, listen_file
    from .services import Studio

    path = Path(file)
    if not path.is_file():
        return {"ok": False, "error": f"file not found: {file}"}
    studio = Studio(paths, settings)
    try:
        model = CLAP_MUSIC_SPEECH
        if not listening_status(paths, model, studio.registry.state(model.id))["installed"]:
            return {"ok": False, "error": "The listening model is not installed: download it on the AI Model page first."}
        clap = load_model(model.install_dir(paths), model.file_map(), settings.listening_device)
        key = f"{model.id}@{model.revision[:12]}:v{LISTEN_VERSION}"
        bank = studio.listening_cache.ensure_prompts(key, clap)
        result = listen_file(clap, path, key)
        summary = bank.summary(result) or {}
        excerpts = excerpt_embeddings(result)
        return {"ok": True, "file": path.name, "duration_s": result.get("duration_s"), "provider": clap.provider,
                "excerpt_starts_s": result.get("excerpt_starts_s"),
                "verdict": summary.get("vocals"), "excerpts_with_singing": summary.get("vocals_excerpts"),
                "vocal_score_per_excerpt": summary.get("vocals_margins"),
                "how_to_read": "an excerpt counts as singing above 0.05; 'sung vocals' needs a third of the excerpts; "
                               "'instrumental' means none; in between is 'unclear'",
                "heard": {k: v for k, v in summary.items() if k not in ("vocals_margins",)},
                "decision_if_rescored": _vocal_decision([bank.vocal_margin(e) for e in excerpts]) if excerpts else None}
    finally:
        studio.shutdown()


def _ai_check(paths: AppPaths, settings, model_file: str) -> dict:
    """Load a model with the bundled llama.cpp runtime and run one constrained request."""

    from pathlib import Path

    from .ai.catalog import ModelRegistry, register_custom_model
    from .ai.runtime import LlamaServerBackend, find_llama_server, run_inference_check

    server = find_llama_server(paths, settings.llama_server_path)
    result = {"runtime": str(server) if server else None, "ok": False}
    if server is None:
        result["error"] = "llama.cpp runtime not found in runtime/llama"
        return result
    try:
        model = register_custom_model(paths, ModelRegistry(paths), Path(model_file))
        backend = LlamaServerBackend(paths, model, server_path=server, gpu_layers=settings.ai_gpu_layers)
        try:
            backend.start(timeout=300)
            result.update(run_inference_check(backend))
            result["args"] = backend.active_args
        finally:
            backend.stop()
    except Exception as exc:  # noqa: BLE001 - reported in the JSON result
        result["error"] = f"{getattr(exc, 'message', exc)} {getattr(exc, 'details', '')}".strip()
    return result


def _fatal(message: str) -> int:
    _print(message)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, APP_DISPLAY_NAME, message)
        del app
    except Exception:
        pass
    return 2


if __name__ == "__main__":
    sys.exit(main())
