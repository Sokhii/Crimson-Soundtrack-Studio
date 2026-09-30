# Crimson Soundtrack Studio

A portable Windows application for replacing **Crimson Desert**'s music with your own legally obtained
music, packaged as a **separate mod** for a mod manager such as DMM. Your game installation is never modified.

The Studio does not reverse-engineer the game itself. It reads the knowledge database produced by
[Crimson Desert Analyzer](https://github.com/Sokhii/Crimson-Desert-Analyzer), a separate tool that
decodes the game's Wwise audio structures.

```
Crimson Desert ──► Crimson Desert Analyzer ──► Analyzer SQLite DB ──► Crimson Soundtrack Studio ──► DMM mod
                                                                        ▲            ▲
                                                           your music library   local AI (planned)
```

> **Status: Phase 1 (foundation) + music library scanning.** You can create projects, select the game,
> import and validate an Analyzer database, browse the game's music structures, and scan/analyse a FLAC
> library. Local AI, thematic matching and the mod compiler are planned (see [docs/roadmap.md](docs/roadmap.md)).
> No mod can be built yet.

## What works now

- **Projects** stored inside the program folder (`projects/`), reopenable without rescanning.
- **Crimson Desert installation check**: compares the archive files recorded by the Analyzer with your
  installation (size/date) to detect game updates or a wrong folder, without opening any game file.
- **Analyzer database import** by drag and drop (the file or the whole Analyzer folder). The Studio
  validates the schema version, required tables and contents, explains problems in plain language, and works
  on its **own read-only copy**, so the original file is never touched (not even by SQLite side files).
- **Game Data browser**: the interactive music hierarchy (switch → playlist → segment → track → audio),
  the states that select each piece (e.g. `BGM_Region=Desert`), events, durations, tempo, markers,
  transition segments, streaming type and soundbanks. Technical details are available on request.
- **Music library**: recursive FLAC scanning with metadata (tags, duration, sample rate, channels, bit depth)
  and deterministic measurements (level, peak, dynamics, spectrum, stereo width, estimated tempo, energy/
  brightness indices). Unchanged files are skipped on rescan; analysis is cached by audio content, so
  renamed, moved or re-tagged files are not re-analysed. Unreadable, missing and duplicate files are reported.
- **Fully portable**: everything the program creates stays in its own folder. See
  [docs/portability.md](docs/portability.md).

## Quick start (portable build)

1. Download `CrimsonSoundtrackStudio-<version>-windows-portable.zip` (Releases, or the latest
   `windows-portable` Actions run).
2. Extract it to a folder you can write to (not *Program Files*), e.g. `D:\Tools\Crimson Soundtrack Studio\`.
3. Run `CrimsonSoundtrackStudio.exe`, create a project, then follow the steps on the Home page:
   choose the game folder, drop in the Analyzer database (`data\database\studio.sqlite3` inside the
   Analyzer folder), choose your music folder.

## Running from source

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python CrimsonSoundtrackStudio.py                    # GUI; data goes to ./dev_home (git-ignored)
python CrimsonSoundtrackStudio.py --selftest         # end-to-end check on synthetic data
python CrimsonSoundtrackStudio.py --portability-check
python -m pytest -q                                  # test suite (no game files or music needed)
```

On Linux, Qt needs a few system libraries (`libegl1 libxkbcommon0 libfontconfig1 libgl1`); tests run with
`QT_QPA_PLATFORM=offscreen`.

Tests against a real Analyzer database are local-only:
`CSS_REAL_ANALYZER_DB=... [CSS_REAL_GAME_DIR=...] python -m pytest tests/test_real_analyzer_db.py -s`.

## Building

`pyinstaller packaging/CrimsonSoundtrackStudio.spec --noconfirm --clean` produces a one-folder build in
`dist/CrimsonSoundtrackStudio/`; `python tools/build_portable.py` zips it. The `windows-portable` workflow
does this on Windows, runs the self-test and the portability check from a moved copy launched from an
unrelated working directory, copies the folder again and verifies it keeps its state, then uploads the ZIP
(and publishes a release for `v*` tags).

## Documentation

- [docs/architecture.md](docs/architecture.md): components, data flow, the Analyzer contract, design decisions
- [docs/portability.md](docs/portability.md): the portable-data rules and how they are verified
- [docs/roadmap.md](docs/roadmap.md): development phases
- [docs/research/modding_format.md](docs/research/modding_format.md): what is known (and not yet known) about building the mod

## Legal

MIT licence. This repository contains **no** game files, music recordings or AI model weights; test audio is
generated. You supply your own legally obtained music. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
