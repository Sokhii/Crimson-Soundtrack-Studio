"""Entry point.

    CrimsonSoundtrackStudio.exe                       start the GUI
    CrimsonSoundtrackStudio.exe --selftest [--keep]   end-to-end check on synthetic data (logs/selftest.json)
    CrimsonSoundtrackStudio.exe --portability-check   verify nothing is written outside the app folder
    CrimsonSoundtrackStudio.exe --print-paths         show the portable directory layout
    CrimsonSoundtrackStudio.exe --print-state         settings/projects/models/caches/logs seen by this copy
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
    cli = args.selftest or args.portability_check or args.print_paths or args.print_state
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
        if args.portability_check:
            from .portability import run_portability_check

            result = run_portability_check(paths)
            _print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["ok"] else 1
        from .ui.main_window import run_gui

        return run_gui(paths, settings, smoke=args.smoke_gui)
    finally:
        shutdown_logging()


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
