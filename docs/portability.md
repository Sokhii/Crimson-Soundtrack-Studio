# Portability

**Rule:** if `CrimsonSoundtrackStudio.exe` is in a folder, every persistent file the Studio creates is
somewhere underneath that folder. Nothing goes to `%APPDATA%`, `%LOCALAPPDATA%`, `%PROGRAMDATA%`, the
registry, Documents, Desktop, Downloads or OneDrive, and no installer is needed.

## Layout

```
Crimson Soundtrack Studio/
├── CrimsonSoundtrackStudio.exe      (+ _internal/ with the bundled Python/Qt runtime)
├── runtime/llama/                   bundled llama.cpp llama-server (read-only)
├── runtime/vgmstream/               bundled vgmstream decoder for the game's audio (read-only)
├── data/
│   ├── config/                      settings.json, model_catalog.json, models.json (no registry, no QSettings)
│   ├── cache/                       audio analysis, game-audio, listening, game-model and AI-response caches
│   └── databases/<hash>/            imported Analyzer database snapshots (read-only) + import.json
├── models/{low,medium,high,custom}/ local AI models (downloaded from inside the app, SHA-256 verified)
├── models/listening/<id>/           optional listening model (ONNX files, SHA-256 verified)
├── projects/<name>/project.sqlite3  one folder per project
├── output/                          built mods (<name>/ and <name>.zip; the previous build is kept as <name>.previous)
├── logs/                            studio.log, crash.log, llama-server.log, selftest.json, portability.json, state.json
└── temp/                            temporary files (incl. decoded game audio, deleted after each file);
                                     entries older than 24 h are removed at startup
```

## How it is enforced

- The root comes from the **executable's real location** (`sys.executable` in the frozen build), never
  from the current working directory, so shortcuts, mod managers and command prompts elsewhere behave
  the same. `CRIMSON_STUDIO_HOME` overrides it for tests; running from source uses `./dev_home`.
- At startup, before Qt or any library loads, `environment.py` points `TMP`/`TEMP`/`TMPDIR`, Python's
  `tempfile`, the XDG base directories (Qt/fontconfig on Linux), Hugging Face/llama.cpp caches and other
  library caches at folders below the root.
- Paths recorded in settings and projects are stored **relative** when inside the root (`app:projects/...`),
  so a moved folder keeps working. External, user-selected locations (game folder, music folder, original
  Analyzer file) are stored as absolute paths and reported when they are no longer available.
- File dialogs use Qt's own dialog by default, because the native Windows dialog records recently used
  folders in the registry (ComDlg32 MRU). `use_native_dialogs` in `settings.json` switches to it if wanted.
- The Analyzer database is snapshotted into `data/databases/`, so SQLite never creates side files next to
  the user's original.
- The llama.cpp child process gets `LLAMA_CACHE` inside `data/cache/` and inherits the redirected temp folders.
- Custom GGUF models the user adds from elsewhere are referenced, not copied; they are external inputs like
  the music folder.
- Game audio is read from the game archives and decoded by vgmstream *into* `temp/gameaudio/`; the files are
  deleted as soon as they are measured. The game folder is only read.
- The listening model runs in-process with ONNX Runtime; on Windows its DirectML provider uses the graphics
  driver (like the Vulkan llama.cpp runtime, the driver may keep its own shader cache).

## Verification

Automated:

- `tests/test_portability.py` runs the real entry point in a subprocess with a **fake user profile**
  (`HOME`, `USERPROFILE`, `TMP`/`TEMP`/`TMPDIR`) from an unrelated working directory, runs the self-test
  and a GUI start, and asserts that nothing was written to the profile, the system temp or the working
  directory. It then copies the whole folder to a new location and checks that settings, projects, models,
  output, imported databases, caches and logs are all found there and the project reopens.
- `--portability-check` (also in CI on Windows, against the frozen EXE) snapshots the real user-profile
  locations (`%APPDATA%`, `%LOCALAPPDATA%`, `%PROGRAMDATA%`, Documents, Desktop, Downloads, `%TEMP%`) and
  `HKCU\Software` before and after running the self-test and a GUI start from another working directory.
  Any new or modified entry mentioning this app or its libraries (Qt, Python, PyInstaller, llama.cpp) fails
  the check; unrelated changes by other programs are listed for information in `logs/portability.json`.
- The `windows-portable` workflow runs the frozen EXE from a copy under `D\...`, launched from another
  directory, then copies it to `E\Tools\...` and verifies with `--print-state` that it kept its state.

Manual check (Windows):

1. Extract to `D:\CrimsonSoundtrackStudio\`, run it, create a project, import an Analyzer DB, scan music.
2. Run `CrimsonSoundtrackStudio.exe --portability-check` and read `logs\portability.json`.
3. Copy the folder to `E:\Tools\CrimsonSoundtrackStudio\`, start it there: the project, settings, caches
   and logs are all present.

Verified during Phase 1 development on Linux (source run and a PyInstaller one-folder build): self-test,
GUI start, outside-write check and copy-to-new-location all pass. The Windows checks run in GitHub Actions.
