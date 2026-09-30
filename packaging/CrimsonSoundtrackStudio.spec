# PyInstaller spec: portable one-folder build.
#   pyinstaller packaging/CrimsonSoundtrackStudio.spec --noconfirm --clean
# Output: dist/CrimsonSoundtrackStudio/CrimsonSoundtrackStudio(.exe) plus its _internal/ folder.
# All user data is created next to the executable at runtime (see src/soundtrack_studio/app_paths.py).
import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT / "src"))
from soundtrack_studio import __version__  # noqa: E402

a = Analysis(
    [str(ROOT / "CrimsonSoundtrackStudio.py")],
    pathex=[str(ROOT / "src")],
    datas=[(str(ROOT / "src" / "soundtrack_studio" / "testing" / "analyzer_schema_v1.sql"), "soundtrack_studio/testing"),
           (str(ROOT / "src" / "soundtrack_studio" / "resources" / "model_catalog.json"), "soundtrack_studio/resources"),
           (str(ROOT / "LICENSE"), "."), (str(ROOT / "THIRD_PARTY_NOTICES.md"), ".")],
    hiddenimports=["soundfile", "_soundfile", "_soundfile_data"],
    excludes=["tkinter", "matplotlib", "PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.Qt3DCore"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="CrimsonSoundtrackStudio",
    console=False,          # windowed; CLI commands write their reports to logs/
    disable_windowed_traceback=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="CrimsonSoundtrackStudio", upx=False)
