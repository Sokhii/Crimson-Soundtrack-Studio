# Architecture

## Starting point (repository inspection, Phase 1)

The repository contained only an MIT `LICENSE`. The companion **Crimson Desert Analyzer** repository was
inspected (read-only) to learn its database contract and reusable conventions:

- Python 3.11+, PySide6 GUI, PyInstaller one-folder build, GitHub Actions on `windows-latest`,
  executable-relative portable folders, llama.cpp bundled as `llama-server` (no Python/Ollama/LM Studio).
- Its SQLite schema is versioned with `PRAGMA user_version` (currently **1**) and append-only migrations.
- It stores its database in **WAL** journal mode at `data/database/studio.sqlite3`.
- It records no explicit game build number, but it records size and mtime of every `.pamt`, `.paz`,
  `meta/0.papgt` and loose file it scanned (`source_file`).

The Studio therefore uses the **same stack** (one toolchain for both tools, proven Windows packaging, easy
llama.cpp reuse in Phase 3) but is a **separate package** (`soundtrack_studio`); it shares no code with the
Analyzer and never writes to Analyzer data.

## Components

```
                        ┌──────────────────── ui/ (PySide6, presentation only) ────────────────────┐
                        │ Home · Game Data · Music Library · Matching (later) · Build (later)       │
                        └──────────────────────────────┬────────────────────────────────────────────┘
                                                       │ worker threads (ui/workers.py)
                                            services.Studio  (coordination only)
          ┌──────────────┬───────────────┬─────────────┼───────────────┬────────────────┬──────────────┐
   project/store   analyzer_db/     game_model/     library/        ai/            semantic/      matching/     compiler/
   (project DB)    import·validate  builder+model   scan·features   catalog·dl     profile·rules  engine        plan·audio·wem
                   reader·compat    (cached cues)   flac·cache      runtime        llm·store      store         bnk·archive·
                   (read-only)                                      (llama.cpp)                   (decisions)   build·validate
          └──────────────┴───────────────┴──────── app_paths · environment · config · logging_setup ─────┘
```

| Package | Responsibility | Must not |
|---|---|---|
| `app_paths`, `environment` | executable-relative root, portable layout, stored-path encoding, env redirection | use the CWD, AppData, the registry |
| `analyzer_db` | import (snapshot), validation, read-only queries, installation check | write to Analyzer data or open the user's file with SQLite |
| `game_model` | Studio interpretation of the Wwise music hierarchy; cache | depend on UI, project or library |
| `project` | project database: settings, references, library facts, events | depend on the compiler or AI |
| `library` | format handlers, metadata, deterministic analysis, cache, scanning | modify user files |
| `services` | orchestrates the above for UI/CLI | contain parsing or analysis logic |
| `ai` | model catalog/registry, hardware estimate, verified downloads, llama-server backend | know about music or projects |
| `semantic` | profile schema, evidence documents, rules, LLM prompts, resumable storage | touch the GUI or the compiler |
| `matching` | candidate filtering, scoring, assignment; proposals vs user decisions | override user decisions |
| `compiler` | read-only game access, audio rendering, PCM WEMs, bank patching, packaging, validation | write outside `temp/`/`output/`, modify the game |
| `ui` | widgets | contain business logic or SQL |

Each component can be replaced independently: e.g. a new audio format is a new `AudioFormatHandler`, a new
Analyzer schema is a new entry in `analyzer_db/contract.py` (plus reader changes), and later the compiler
will consume the game model and accepted mappings without knowing about the AI.

## The Analyzer contract

`analyzer_db/contract.py` lists every table and column the Studio reads; everything else is ignored, so
additive Analyzer changes do not break the Studio. Validation (`validator.py`) produces plain-language errors
(e.g. *"Expected schema: 1 / Found schema: 2"*) and warnings, and picks the most recently scanned
installation with a completed scan.

**Why a snapshot instead of opening the file:** SQLite creates `-wal`/`-shm` files next to a WAL database
even for a `mode=ro` connection (verified during inspection). The importer therefore reads the user's file
(and its `-wal`, so committed-but-uncheckpointed data is included) with plain byte reads, consolidates the
copy with SQLite's backup API into `data/databases/<sha256>/analyzer.sqlite3`, validates it, and opens it
`mode=ro&immutable=1` from then on. Re-importing identical content reuses the snapshot; a changed original is
detected by size/mtime and reported. Projects keep working if the original file moves.

**Installation check** (`compat.py`): only `stat` calls; statuses `match`, `probable_match` (sizes equal,
dates differ, as after copying an installation), `mismatch` (files missing/different: game update or wrong
folder), `not_game_folder`, `unverifiable`.

## Game music model

`game_model/builder.py` builds the hierarchy from `wwise_object` (type codes 10 segment, 11 track,
12 switch, 13 playlist), parsed `object_ref` links (`child`, `parent`, `playlist_segment`, `switch_assoc`,
`transition_segment`), `media_source`/`wem` (audio format, streaming), `name` (hash-verified first) and
event → action → target references. Objects duplicated across banks are merged; cycles and shared children
are handled. Unknown state values stay numeric; nothing is invented. The result is cached as JSON per
(snapshot hash, installation, builder version).

**Cues.** A `MusicCue` is a view of one **MusicSegment** with its context: ancestor chain, state path
(e.g. `BGM_Region=Desert`), events, tracks, media, duration, tempo, markers and whether it is used as a
transition. This keeps the Wwise structure visible instead of assuming "one WEM = one song".
*Open question for Phase 4/5:* whether replacement should operate per segment, per playlist (a sequence of
segments forming one piece), or per track layer (the Analyzer's research notes stems such as
`1normal`/`2tensioned`, `stem1..4`). This will be decided from real-installation evidence, not assumed.

Media the Analyzer classifies as music but that is played by plain `Sound` objects (outside the interactive
music hierarchy) is counted and reported as a warning; it is not yet presented as cues.

## Project database (`projects/<name>/project.sqlite3`)

Versioned with `PRAGMA user_version` and identified by `PRAGMA application_id` (`CSS1`); a project from a
newer Studio is refused unchanged, damaged files produce a clear error. Rollback-journal mode keeps it a
single file that is safe to copy when the program is closed.

| Tables | Content |
|---|---|
| `project_meta`, `project_setting` | name, dates, game path, music path |
| `analyzer_ref` | imported database (hash, stored snapshot path, source, schema, installation, validation) |
| `game_check` | installation check history |
| `library_root`, `library_scan`, `track` | scanned files, status (ok/error/missing), audio identity, duplicates |
| `track_metadata` | deterministic container/tag facts |
| `track_features` | deterministic signal measurements (never AI output) |
| `project_event` | user-visible warnings/errors |

| `semantic_profile`, `semantic_override` (format 2) | interpretations per source (`rules`, `llm` + model and prompt version + evidence hash) and the user's edits |
| `match_run`, `match_proposal`, `match_decision` (format 3) | machine proposals (recomputed) and user decisions (never touched by reruns) |
| `build` (format 4) | build history: settings, output location, validation summary, errors |

## Music library analysis

- **Formats:** `library/formats.py` registry; FLAC today. FLAC metadata is read by a small strict parser
  (`flac_meta.py`) so the bundle stays free of GPL code; audio is decoded with libsndfile (`soundfile`).
- **Identity:** FLAC STREAMINFO MD5 of the decoded audio (+ length/rate/channels), so tag edits do not
  invalidate analysis; files without an MD5 are hashed.
- **Measurements** (`features.py`, streamed in blocks): RMS/peak/crest, level spread, silence ratio,
  spectral centroid/roll-off/flatness, band energy, zero-crossing rate, stereo width, onset rate/peakiness,
  and a **tempo estimate** from the autocorrelation of a 32-band onset envelope. Tempo is only reported when
  (a) real broadband onsets exist (frames where ≥30 % of bands rise at once; this rejects drones and beating
  partials that would otherwise produce a confident fake tempo) and (b) the periodicity is strong enough.
  Every value carries a label: *measured*, *estimate* or *heuristic*. Musical key and vocal presence are
  deliberately not computed (unreliable with simple methods).
- **Caching:** `data/cache/audio_analysis.sqlite3` maps (path, size, mtime) → probe result and
  (audio identity, analysis version) → measurements. The project keeps its own copy of the results.

## Errors and logging

User-facing errors are `StudioError` subclasses with a title, plain message, hint and technical details
(shown in an expandable section and logged). Logs rotate in `logs/studio.log`; the user's home folder and
account name are redacted, music tags are not logged; `logs/crash.log` receives fatal tracebacks.

## Local AI and semantic descriptions

- `ai/` owns a `llama-server` child process (bundled in `runtime/llama/`) bound to 127.0.0.1, with GPU-offload
  fallbacks, JSON-schema-constrained output and handling of cut-off answers. Models are catalog data
  (`data/config/model_catalog.json`), stored in `models/<tier>/<id>/`, downloaded resumably and verified
  against the Hugging Face SHA-256. Hardware detection only reads OS information (no PowerShell/WMI, which
  write caches into the user profile).
- `semantic/` builds an *evidence document* per track (tags, measurements) and per cue (Wwise names, states,
  events, structure, community notes); its hash is the cache key. Profiles use controlled vocabularies so
  sources are comparable. Precedence: user override > local AI > rules. Runs commit per item (resumable) and
  share a response cache across projects (`data/cache/ai_responses.sqlite3`).

## Game audio and the listening model

- `gameaudio/` reads each game music source through the Analyzer's records (a streamed `.wem` from the archives,
  or the copy embedded in a soundbank's `DATA` section), verifies it against the Analyzer's hash, decodes it with
  the bundled vgmstream into `temp/gameaudio/`, measures it with the same `library.features` analysis as user
  tracks, and deletes the temp files. Results are cached in `data/cache/game_audio.sqlite3` by content hash, so
  all projects and re-imported databases of the same game version share them. A cue's audio is its primary track's
  longest decoded source.
- `listening/` is optional. CLAP runs through ONNX Runtime (DirectML on Windows, CPU fallback); its audio front end
  is a NumPy re-implementation of transformers' `ClapFeatureExtractor`. Each file contributes up to six ten-second
  excerpts (read by seeking); only the mean embedding is stored (`data/cache/listening.sqlite3` for tracks, the
  game-audio cache for game music). *Heard* tags (profile vocabulary, vocals vs instrumental) are computed from the
  embedding and cached prompt embeddings; matching uses embedding similarity ("sounds alike").
- Evidence documents gain `measurements` (cues) and `heard` (both) only when available, so descriptions of
  unchanged evidence keep their cache keys. The text model is told that heard evidence beats names and tags, and
  vocal presence comes from listening (or clear tags), never from genre or album type.

## Calibrated scores and standout matching

`listening/calibration.py` turns the model's raw word similarities into 0-100 scores. Each of ~340 words
(`listening/vocabulary.py`: mood, emotion, atmosphere, instrumentation, style, rhythm, texture) is judged against
every piece analysed (the user's tracks plus the game's music): z = (similarity - word mean) / word spread,
score = 100 x Phi(z). Words that fit every piece average 50 and stop appearing; tags are shown from 85; words whose
spread is under 40% of the median are skipped; with fewer than 30 pieces the scores are relative to the piece's own
other words and flagged provisional. The profile vocabulary the text model writes from (92 words) is unchanged; heard
words outside it become themes. Standout matching compares each piece's *standout vector* (z above 0.5, capped at 3,
zero elsewhere) by cosine (a cosine of 0.6 counts as a complete match) and explains the match with the words that
stand out for both and strong cue words the track lacks. `Match by standout scores` (setting `matching_mode`) switches
between this and the legacy tag overlap; without the listening model the legacy matching is always used. Accepted,
rejected and chosen tracks are never changed by either mode.

**Game descriptions are shared between projects.** A game cue is judged against the game's music alone
(`Studio.calibration(game_only=True)`), so its evidence document, and therefore its key in the shared AI answer cache
(`data/cache/ai_responses.sqlite3`), does not depend on the user's library. A new project (or a changed library) finds
every cue already described and only the user's tracks are described. Track descriptions and matching still use the
combined reference. `Re-describe…` (AI Model and Matching pages) has the AI write the chosen descriptions again
(`analyze_semantics(redo=("cue", "track"))`, `SemanticStore.run(force=True)`), ignoring saved and cached answers; the
user's edits and matching decisions are kept.

## Matching

Unit: the MusicSegment (see `docs/research/modding_format.md`). Deterministic eligibility and filtering,
coverage-aware semantic similarity (agreement on few attributes is weak evidence; with the listening model the tag
overlap is replaced by standout-score similarity, see below, unless the user switches to legacy matching), "sounds alike" when the
listening model heard both sides (rank of the audio-embedding similarity within the library), duration and tempo fit,
optional local-AI judgement of the top five only, then a diversity-aware assignment with a reuse limit.
Every proposal carries reasons, warnings and a confidence that accounts for evidence quality. Only cues the
user accepts or chooses form the build mapping.

## Compiler

`plan` (primary track gets the music, other layers silence; all banks holding a source, including twins) →
`audio` (decode, polyphase resample to 48 kHz, channel mapping, trim/loop/pad with fades, timeline slicing per
clip) → `loudness` (one gain per cue, see below) → `wem` (Wwise PCM, or a `.wav` per source that `wwise` converts to
Wwise Vorbis through the user's `WwiseConsole.exe`, 24 sources per run) → `archive` (read original banks through the
Analyzer's entry records, verify SHA-1) → `bnk` (patch source fields in place, rebuild `DIDX`/`DATA`) →
`build` (workspace in `temp/`, manifest/README/report) → `validate` (independent re-read of everything) →
`output/<mod name>/`. Any disagreement between the game files and the Analyzer data aborts the build.

**Loudness (0.14).** `loudness.py` measures integrated loudness per ITU-R BS.1770 (K-weighting by FFT convolution
with the filters' impulse response, 400 ms gated blocks; within 0.05 LU of a reference implementation on test
signals) and true peak (4x interpolation around the loudest samples). Each cue gets one gain, never a limiter: in
*match* mode (default) towards the loudness of the original game music it replaces, measured during "Analyse game
audio" (`loudness_lufs` in the game-audio measurements; measurements made before 0.14 only have an RMS level, from
which the target is estimated); in *fixed* mode towards the Build page's target (also the fallback for cues whose
original was not measured); in *off* mode the track keeps its level. The gain is capped so the true peak stays at or
below -1 dBTP (room for the Vorbis encoder's small overshoot); a cue that cannot reach its target stays quieter, and
the report records target, gain and shortfall per cue plus a summary. In-bank volumes are not changed.

**Vorbis builds (default).** Matching the community mods that are known to play, the soundbanks keep the game's
Vorbis codec, storage type and flags; only media changes: an in-bank source gets the new `.wem` in `DATA`, a
prefetch-streamed source gets the first part of its new `.wem` (header, seek table, setup and 0.1 s of audio - a
plain byte prefix, as in the working mods) with the size field updated, and a plain streamed source needs no bank
change. Validation checks the Vorbis header (sample count within one block of the segment), that every prefetch
copy is the beginning of the written file and that in-memory sizes match. Wwise is located through the chosen path,
`%WWISEROOT%`, `Program Files\Audiokinetic\Wwise*` or `PATH`; its conversion project lives in `data/wwise/`. A project made by `create-new-project` has only a PCM "Default
Conversion Settings" and none of the factory presets, so the Studio patches the project's
`Conversion Settings/Default Work Unit.wwu` itself (both the default and its own "CSS Vorbis" entry: Vorbis, Wwise
defaults for quality/channels/sample rate; format taken from a project saved by Wwise 2023.1.19) and asks for "CSS
Vorbis". Every path given to WwiseConsole is in 8.3 short form (its tools fail near 260 characters).
