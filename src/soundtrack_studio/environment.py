"""Keeps third-party libraries from writing outside the portable folder.

Called once at startup, before Qt or any heavy library is imported. Every
library that honours an environment variable for its temp/cache/config
location is pointed at a directory below the application root.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict

from .app_paths import AppPaths

# Environment variables that some libraries use to pick cache/config folders.
# Only variables that redirect *our own process* are set; nothing global is changed.
def portable_environment(paths: AppPaths) -> Dict[str, str]:
    cache = paths.cache
    return {
        # process temp directory (tempfile, Qt, libsndfile, subprocesses)
        "TMP": str(paths.temp),
        "TEMP": str(paths.temp),
        "TMPDIR": str(paths.temp),
        # freedesktop base dirs (Qt/fontconfig on Linux; harmless on Windows)
        "XDG_CACHE_HOME": str(cache / "xdg" / "cache"),
        "XDG_CONFIG_HOME": str(cache / "xdg" / "config"),
        "XDG_DATA_HOME": str(cache / "xdg" / "data"),
        "XDG_STATE_HOME": str(cache / "xdg" / "state"),
        # model/runtime caches used by AI tooling in later phases
        "HF_HOME": str(cache / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "huggingface" / "hub"),
        "LLAMA_CACHE": str(cache / "llama"),
        # numeric libraries that may JIT/cache
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        # Python bytecode for any source-run imports stays out of the user profile
        "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
    }


# Values of the redirected variables before startup (used by the portability check
# to know where the *system* temp and cache folders are).
ORIGINAL_ENV: Dict[str, str] = {}


def configure_process_environment(paths: AppPaths) -> Dict[str, str]:
    env = portable_environment(paths)
    for key, value in env.items():
        if key in os.environ and key not in ORIGINAL_ENV:
            ORIGINAL_ENV[key] = os.environ[key]
        os.environ[key] = value
    paths.temp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(paths.temp)
    return env
