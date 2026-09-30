"""Download a prebuilt vgmstream Windows decoder into the portable build.

Used by CI. vgmstream (ISC licence) decodes the game's Wwise audio for the
read-only game-audio analysis; it is distributed alongside the application in
``runtime/vgmstream/`` so users never install anything. The archive is pinned
by URL and SHA-256 (the same build Crimson Desert Mod Workbench bundles for
previewing this game's audio).

    python tools/fetch_vgmstream.py --dest vendor/vgmstream
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

VERSION = "r1980"
BUILD_COMMIT = "21bfb6f0a513271f2e18a51322128756bb59f365"
URL = f"https://github.com/bnnm/vgmstream-builds/raw/{BUILD_COMMIT}/bin/vgmstream-{VERSION}-test-u.zip"
SHA256 = "110f9087e60057c4af6cff84e26c214159c224792421affdddd3aaa2091f2641"

LICENSE_NOTE = """vgmstream - https://github.com/vgmstream/vgmstream
Licence: ISC-style (see the upstream COPYING file). The Windows build includes
third-party decoder libraries (e.g. libvorbis/libogg: BSD; mpg123, FFmpeg: LGPL,
dynamically linked) under their own licences; see https://github.com/vgmstream/vgmstream/blob/master/doc/BUILD.md
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    print(f"vgmstream {VERSION}: downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "CrimsonSoundtrackStudio-build"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        blob = resp.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SHA256:
        print(f"checksum mismatch: expected {SHA256}, got {digest}", file=sys.stderr)
        return 1
    dest = Path(args.dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(dest)
    exe = next(iter(sorted(dest.rglob("vgmstream-cli.exe"))), None) or next(iter(sorted(dest.rglob("test.exe"))), None)
    if exe is None:
        print(f"no vgmstream-cli.exe in the archive: {[p.name for p in dest.rglob('*')][:30]}", file=sys.stderr)
        return 1
    if exe.parent != dest:  # flatten: the decoder and its DLLs must sit directly in dest
        for item in list(exe.parent.iterdir()):
            shutil.move(str(item), str(dest / item.name))
        exe = dest / exe.name
    if exe.name != "vgmstream-cli.exe":
        exe.rename(exe.with_name("vgmstream-cli.exe"))
    (dest / "LICENSE-vgmstream.txt").write_text(LICENSE_NOTE, encoding="utf-8")
    (dest / "RUNTIME_INFO.json").write_text(json.dumps({"vgmstream": VERSION, "url": URL, "sha256": SHA256}, indent=2),
                                            encoding="utf-8")
    print("files:", sorted(p.name for p in dest.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
