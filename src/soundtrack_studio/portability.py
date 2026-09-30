"""Automated portability check.

Runs the application's self-test (and a short GUI start) in a child process
from a *different working directory*, with the user's original environment,
and compares the usual per-user/system application-data locations and the
registry (Windows) before and after. Any new or modified entry that belongs
to this application - or to the libraries it uses (Qt, Python, PyInstaller,
llama.cpp) - is a failure. Unrelated changes by other programs running at
the same time are listed for information only.

Usage: ``CrimsonSoundtrackStudio.exe --portability-check`` (writes
``logs/portability.json``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from . import __version__
from .app_paths import ENV_HOME_OVERRIDE, AppPaths, is_frozen
from .environment import ORIGINAL_ENV, portable_environment

APP_MARKERS = ("crimson", "soundtrack", "cstudio", "pyside", "qtproject", "qt-project", "qt6", "python",
               "pyinstaller", "_mei", "llama", "ggml")
MAX_ENTRIES = 300_000


def child_command(paths: AppPaths) -> List[str]:
    if is_frozen():
        return [sys.executable]
    return [sys.executable, "-m", "soundtrack_studio"]


def child_environment(paths: AppPaths) -> Dict[str, str]:
    """The environment the user would launch the app with (undo our own redirections)."""

    env = dict(os.environ)
    for key in portable_environment(paths):
        if key in ORIGINAL_ENV:
            env[key] = ORIGINAL_ENV[key]
        else:
            env.pop(key, None)
    if not is_frozen():
        env[ENV_HOME_OVERRIDE] = str(paths.root)
        src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def watched_locations(env: Optional[Dict[str, str]] = None) -> List[Path]:
    env = env or dict(os.environ)
    home = Path(env.get("USERPROFILE") or env.get("HOME") or Path.home())
    candidates: List[Optional[str]] = []
    if os.name == "nt":
        candidates += [env.get("APPDATA"), env.get("LOCALAPPDATA"), env.get("PROGRAMDATA"),
                       str(home / "Documents"), str(home / "Desktop"), str(home / "Downloads"),
                       env.get("TEMP"), env.get("TMP")]
    else:
        candidates += [str(home / ".config"), str(home / ".cache"), str(home / ".local"), str(home / "Documents"),
                       str(home / "Desktop"), str(home / "Downloads"), str(home),
                       env.get("TMPDIR") or "/tmp"]
    seen: Set[str] = set()
    out = []
    for c in candidates:
        if c and os.path.isdir(c) and os.path.normcase(os.path.abspath(c)) not in seen:
            seen.add(os.path.normcase(os.path.abspath(c)))
            out.append(Path(c))
    return out


def excluded_roots(paths: AppPaths) -> List[Path]:
    """The app folder itself, plus the source checkout when running from source (bytecode, dev_home)."""

    roots = [paths.root]
    if not is_frozen():
        roots.append(Path(__file__).resolve().parents[2])
    return roots


def snapshot(locations: List[Path], exclude: List[Path], depth: int = 4) -> Dict[str, int]:
    state: Dict[str, int] = {}
    excluded = tuple(os.path.normcase(os.path.abspath(e)) for e in exclude)
    for base in locations:
        base_depth = str(base).rstrip("\\/").count(os.sep)
        for dirpath, dirnames, filenames in os.walk(base, onerror=lambda e: None):
            if os.path.normcase(os.path.abspath(dirpath)).startswith(excluded):
                dirnames[:] = []
                continue
            if dirpath.count(os.sep) - base_depth >= depth:
                dirnames[:] = []
            for name in dirnames + filenames:
                full = os.path.join(dirpath, name)
                try:
                    state[full] = os.lstat(full).st_mtime_ns
                except OSError:
                    pass
                if len(state) > MAX_ENTRIES:
                    return state
    return state


def registry_snapshot() -> Set[str]:
    if os.name != "nt":
        return set()
    import winreg

    keys: Set[str] = set()

    def walk(root, path: str, level: int) -> None:
        try:
            handle = winreg.OpenKey(root, path)
        except OSError:
            return
        with handle:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(handle, i)
                except OSError:
                    break
                full = f"{path}\\{sub}"
                keys.add(full)
                if level < 2:
                    walk(root, full, level + 1)
                i += 1

    walk(winreg.HKEY_CURRENT_USER, "Software", 0)
    return keys


def _flag(path: str) -> bool:
    lower = path.lower().replace("\\", "/")
    return any(marker in lower for marker in APP_MARKERS)


def run_portability_check(paths: AppPaths, include_gui: bool = True, timeout: int = 600) -> Dict:
    env = child_environment(paths)
    locations = watched_locations(env)
    before = snapshot(locations, excluded_roots(paths))
    reg_before = registry_snapshot()
    cwd = Path(env.get("TEMP") or env.get("TMPDIR") or Path(paths.root).anchor)
    if not cwd.is_dir():
        cwd = Path(paths.root).parent
    runs = []
    for args in (["--selftest"], ["--smoke-gui"] if include_gui else None):
        if args is None:
            continue
        started = time.monotonic()
        proc = subprocess.run(child_command(paths) + args, env=env, cwd=str(cwd), timeout=timeout,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        runs.append({"args": args, "exit_code": proc.returncode, "seconds": round(time.monotonic() - started, 1),
                     "output_tail": proc.stdout.decode("utf-8", "replace")[-2000:]})
    after = snapshot(locations, excluded_roots(paths))
    reg_after = registry_snapshot()

    created = sorted(set(after) - set(before))
    modified = sorted(p for p in set(after) & set(before) if after[p] != before[p])
    flagged = [p for p in created + modified if _flag(p)]
    reg_new = sorted(reg_after - reg_before)
    reg_flagged = [k for k in reg_new if _flag(k)]
    selftest = {}
    try:
        selftest = json.loads((paths.logs / "selftest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    result = {
        "version": __version__,
        "root": str(paths.root),
        "launched_from": str(cwd),
        "watched_locations": [str(p) for p in locations],
        "runs": runs,
        "selftest_ok": bool(selftest.get("ok")),
        "flagged_external_changes": flagged,
        "flagged_registry_keys": reg_flagged,
        "other_external_changes": [p for p in created + modified if p not in flagged][:200],
        "other_registry_keys": [k for k in reg_new if k not in reg_flagged][:200],
    }
    result["ok"] = (all(r["exit_code"] == 0 for r in runs) and result["selftest_ok"]
                    and not flagged and not reg_flagged)
    (paths.logs / "portability.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
