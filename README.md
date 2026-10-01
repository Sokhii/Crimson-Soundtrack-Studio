# Crimson Soundtrack Studio

A portable Windows application for replacing **Crimson Desert**'s music with your own legally obtained
music, packaged as a **separate mod** for a mod manager such as DMM or CDUMM. Your game installation is
never modified.

The Studio does not reverse-engineer the game itself. It reads the knowledge database produced by
[Crimson Desert Analyzer](https://github.com/Sokhii/Crimson-Desert-Analyzer), a separate tool that
decodes the game's Wwise audio structures.

```
Crimson Desert ──► Crimson Desert Analyzer ──► Analyzer SQLite DB ──► Crimson Soundtrack Studio ──► mod package
                                                                   ▲             ▲
                                                      your music library   local AI (llama.cpp)
```

> **Status: 0.14.2 (feature-complete beta).** The whole workflow works end to end: project → game → Analyzer
> database → music library → local AI (optional) → thematic matching → review → build → validated mod
> package. **It has not yet been verified in the game itself** (the development environment has no game
> install). Please report results, including which package layout your mod manager imports (see
> [docs/research/modding_format.md](docs/research/modding_format.md)).

## Workflow

1. **Project**: everything is stored in the program's own `projects/` folder.
2. **Crimson Desert folder**: checked against the Analyzer database (archive sizes/dates) without opening any game file.
3. **Analyzer database**: drag and drop the file or the whole Analyzer folder. The Studio works on its own
   read-only copy (not even SQLite side files touch your original) and validates schema and contents.
4. **Music library**: recursive FLAC scan with tags and measured audio features (level, dynamics, spectrum,
   stereo width, estimated tempo, energy/brightness). Unchanged, renamed and re-tagged files are not re-analysed.
5. **AI model (optional)**: choose Low / Medium / High / Custom. The model downloads into `models/`, is verified
   by SHA-256 and runs locally through the bundled llama.cpp runtime; nothing else to install. Without a
   model, rule-based descriptions are used.
   **Listening model (optional, alongside)**: CLAP listens to the audio itself (your tracks and the game's music)
   and recognises instruments, vocals, mood, style, rhythm and recording texture (about 340 words, each scored
   0-100 against all the music analysed), and how much two pieces sound alike. It runs on the
   graphics card via DirectML (any DirectX 12 GPU) or on the CPU, and adds its findings to the descriptions and to
   matching; it replaces nothing.
6. **Describe music**: the game's music is first decoded **read-only** (with the bundled vgmstream, into the
   program's `temp/` folder, deleted straight after) and measured like your tracks; with the listening model on,
   both sides are also listened to. Every track and game cue then gets a structured description (mood, emotion,
   atmosphere, instrumentation, style, themes, energy/darkness/tension/valence, vocals) from tags, names,
   community notes, measurements and what was heard. Vocals are only judged from listening or clear tags. Results
   are cached and shared by all projects. You can edit any description; your edits always win.
7. **Find matches**: proposals by musical character, not gameplay category, with reasons, warnings and a
   calibrated confidence. With the listening model on, pieces are compared by what *stands out* about each one
   (0-100 scores per word, relative to all the music analysed, so words that fit everything stop counting) and by
   how much a track *sounds like* the original. "Match by standout scores" on the Matching page switches back to
   the legacy tag-overlap matching at any time. Optionally the local AI judges the top five candidates per cue.
8. **Review**: accept, reject, choose another track, keep the original, set the fit mode (trim / loop / play
   once) and a start offset. Only what you accept or choose is built; re-running matching never changes your
   decisions.
9. **Build**: each accepted track is rendered onto the segment's exact timeline (48 kHz), converted to Wwise
   Vorbis by your own Wwise installation (the format the game and the working Nexus music mods use; uncompressed
   PCM is still available without Wwise, but has not been confirmed to play in game), and the affected
   soundbanks are patched so the game's interactive music structure (durations, markers,
   transitions, playlists) is preserved. The result is validated and saved to `output/<mod name>/` (+ ZIP).
10. **Install** the package with your mod manager. Your game folder is never modified.

## Quick start (portable build)

1. Download `CrimsonSoundtrackStudio-<version>-windows-portable.zip` (Releases, or the latest
   `windows-portable` Actions run).
2. Extract it to a folder you can write to (not *Program Files*), e.g. `D:\Tools\Crimson Soundtrack Studio\`.
3. Run `CrimsonSoundtrackStudio.exe`, create a project and follow the steps on the Home page.

**Wwise (for the default Vorbis format).** Wwise is Audiokinetic's proprietary audio tool (free for non-commercial
use) and cannot be shipped with the Studio. On the Build page, *Get Wwise…* opens Audiokinetic's download page and
explains the install (Audiokinetic Launcher → Wwise 2023.1, the version that matches the game's soundbanks);
*Find Wwise* / *Test Wwise* then check it. The Studio only runs `WwiseConsole.exe`; all converted files stay in
its own folder, but Wwise itself is installed in Program Files and keeps settings in your user profile.
Without Wwise you can choose the PCM format (about 11.5 MB per stereo minute; not proven to play in game).

## Running from source

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python CrimsonSoundtrackStudio.py                    # GUI; data goes to ./dev_home (git-ignored)
python CrimsonSoundtrackStudio.py --selftest         # end-to-end check on synthetic data, including a real build
python CrimsonSoundtrackStudio.py --portability-check
python -m pytest -q                                  # test suite (no game files or music needed)
```

On Linux, Qt needs a few system libraries (`libegl1 libxkbcommon0 libfontconfig1 libgl1`); tests run with
`QT_QPA_PLATFORM=offscreen`. For AI features from source, put a llama.cpp build in `dev_home/runtime/llama/`.

Optional integration tests: `CSS_LLAMA_SERVER=<llama-server>` runs the real runtime with a generated tiny
model (`tools/make_tiny_gguf.py`); `CSS_VGMSTREAM=<vgmstream-cli>` decodes the compiler's output and game audio
with vgmstream; `CSS_CLAP_HOME=<folder>` (after `tools/fetch_listening_model.py --home <folder>`, plus PyTorch and
transformers) compares the listening pipeline with the original implementation;
`CSS_REAL_ANALYZER_DB=... [CSS_REAL_GAME_DIR=...]` checks a real Analyzer database.

## Building

`pyinstaller packaging/CrimsonSoundtrackStudio.spec --noconfirm --clean` produces a one-folder build in
`dist/CrimsonSoundtrackStudio/`; `python tools/build_portable.py` zips it. The `windows-portable` workflow does
this on Windows: it fetches the llama.cpp runtime (`tools/fetch_llama_runtime.py`, Vulkan build with CPU
fallback) and vgmstream (`tools/fetch_vgmstream.py`, pinned by SHA-256) and tests them (a real model load; the
listening model through the frozen EXE), bundles it next to the EXE, runs the self-test (including a real
mod build) and the portability check from a moved copy launched from an unrelated folder, copies the folder
again to verify it keeps its state, and uploads the ZIP (a release for `v*` tags).

## Documentation

- [docs/architecture.md](docs/architecture.md): components, data flow, the Analyzer contract, design decisions
- [docs/research/modding_format.md](docs/research/modding_format.md): mod format and Wwise evidence behind the compiler
- [docs/research/listening.md](docs/research/listening.md): game-audio decoding and the listening model: choices and evidence
- [docs/portability.md](docs/portability.md): the portable-data rules and how they are verified
- [docs/roadmap.md](docs/roadmap.md): phases and open questions

## Legal

MIT licence. This repository contains **no** game files, music recordings or AI model weights; test audio is
generated. You supply your own legally obtained music; each AI model is under its own licence, shown before
download. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
