"""Launcher for running from source and the PyInstaller entry script."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from soundtrack_studio.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
