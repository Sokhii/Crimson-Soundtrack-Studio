"""Zip the PyInstaller one-folder build as the portable release.

Only program files are packaged: runtime data folders (data/, projects/, models/, logs/, ...)
are never included, even if the build folder was run locally.
"""

from __future__ import annotations

import platform
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from soundtrack_studio import __version__  # noqa: E402

RUNTIME_DATA = {"data", "projects", "models", "output", "logs", "temp"}


def main() -> int:
    build = ROOT / "dist" / "CrimsonSoundtrackStudio"
    if not build.is_dir():
        print("run PyInstaller first (dist/CrimsonSoundtrackStudio missing)")
        return 1
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(platform.system(), platform.system().lower())
    target = ROOT / "dist" / f"CrimsonSoundtrackStudio-{__version__}-{system}-portable.zip"
    top = "Crimson Soundtrack Studio"
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(build.rglob("*")):
            rel = path.relative_to(build)
            if rel.parts[0] in RUNTIME_DATA or not path.is_file():
                continue
            zf.write(path, f"{top}/{rel.as_posix()}")
            count += 1
    print(f"{target.name}: {count} files, {target.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
