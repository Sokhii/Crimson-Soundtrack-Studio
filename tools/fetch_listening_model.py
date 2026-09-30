"""Download the listening model with the Studio's own downloader (pinned revision, SHA-256 checks).

Used by CI to test the download path and to provide the model to the reference tests:

    python tools/fetch_listening_model.py --home <portable root>

The files land in ``<home>/models/listening/<id>/`` exactly as the application stores them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    from soundtrack_studio.app_paths import AppPaths
    from soundtrack_studio.listening.catalog import CLAP_MUSIC_SPEECH
    from soundtrack_studio.services import Studio

    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True)
    args = parser.parse_args()
    paths = AppPaths(Path(args.home)).ensure()
    studio = Studio(paths)
    last = [0.0]

    def progress(text, done, total):
        if total and done / total - last[0] >= 0.25:
            last[0] = done / total
            print(f"{text}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)

    result = studio.download_listening_model(CLAP_MUSIC_SPEECH.id, progress)
    print("downloaded:", result)
    print("folder:", CLAP_MUSIC_SPEECH.install_dir(paths))
    studio.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
