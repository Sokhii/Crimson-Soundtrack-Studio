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
   project/store   analyzer_db/     analyzer_db/    game_model/      library/          (Phase 3+)
   (project DB)    importer +       reader +        builder +        formats · flac_meta  ai/ matching/
                   validator        compat          model (cues)     features · scanner   review/ compiler/
                   (snapshot)       (read-only)     (cached)         cache
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

Later phases add, in separate tables, AI semantic metadata, proposed matches, user overrides (which always
take precedence), build configuration and build history.

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
