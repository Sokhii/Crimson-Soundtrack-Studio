# Portability

**Rule:** if `CrimsonSoundtrackStudio.exe` is in a folder, every persistent file the Studio creates is
somewhere underneath that folder. Nothing goes to `%APPDATA%`, `%LOCALAPPDATA%`, `%PROGRAMDATA%`, the
registry, Documents, Desktop, Downloads or OneDrive, and no installer is needed.

## Layout

```
Crimson Soundtrack Studio/
├── CrimsonSoundtrackStudio.exe      (+ _internal/ with the bundled runtime)
├── data/
│   ├── config/settings.json         application settings (no registry, no QSettings)
│   ├── cache/                       audio analysis cache, game-model cache, redirected library caches
│   └── databases/<hash>/            imported Analyzer database snapshots (read-only) + import.json
├── models/{low,medium,high,custom}/ local AI models (Phase 3; downloaded from inside the app)
├── projects/<name>/project.sqlite3  one folder per project
├── output/                          built mods (later phases)
├── logs/                            studio.log, crash.log, selftest.json, portability.json, state.json
└── temp/                            temporary files; entries older than 24 h are removed at startup
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
