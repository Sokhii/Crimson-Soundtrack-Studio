# Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 Foundation | project structure, GUI, portable paths, projects, settings, logging, Analyzer DB import/validation, game path selection and verification, game data browser | **done** |
| 2 Music library | recursive FLAC scanning, metadata, deterministic audio features, caching, music browser | **done** (FLAC only; further formats plug into `library/formats.py`) |
| 3 Local AI | bundled llama.cpp `llama-server` runtime, model manager (Low/Medium/High/Custom tiers, hardware-based recommendation, resumable SHA-256-verified downloads into `models/<tier>/`), structured semantic profiles for user tracks and game cues (rule-based baseline + schema-constrained local AI), per-item resumable runs, shared response cache, user overrides that always win | **done** |
| 4 Matching | per-segment matching: eligibility and length filtering, coverage-aware semantic similarity + duration/tempo fit, optional local-AI judgement of the shortlist only, diversity-aware assignment, confidence with uncertainty warnings, Matching review page (accept / reject / choose / keep original / fit mode), decisions that reruns never change | **done** |
| 5 Compiler research | DMM/CDUMM package formats, 2.0 overlay-slot change, archive reading, v150 source layout, Wwise PCM layout, prefetch implications; evidence with trust labels in `docs/research/modding_format.md` | **done** |
| 6 Compiler | segment-timeline rendering (resample, fit, loudness), PCM WEMs, verified bank reading and patching (PCM codec, streaming, DIDX/DATA rebuild, twin banks), file-replacement package (Crimson Browser manifest or package folders), independent validation (and vgmstream cross-check in tests), build history, Build page | **done** (in-game test pending: no game in the development environment) |
| 8 Listening | read-only decoding and measurement of the game's own music (vgmstream, bundled), optional CLAP listening model alongside the description models (instruments, vocals, mood, style; "sounds alike" in matching), descriptions that trust what was heard over names and tags | **done** (verified against the reference implementation in CI; in-game audio decoding to be confirmed with "Test decoding") |
| 9 Standout scores | calibrated 0-100 scores per word relative to all the music analysed (hub words such as "solemn" stop repeating), ~340-word listening vocabulary (adds rhythm and texture), standout-score matching with a switch back to the legacy tag matching | **done** (repetition measured in CI) |
| 7 Release | llama.cpp runtime bundled and tested in the Windows build, self-test covering a real build, portability checks on the frozen EXE, versioning (0.9.0), documentation | **done** (1.0 after in-game verification) |

## Decisions still open (need evidence)

- **Replacement granularity**: decided: the MusicSegment (see docs/research/modding_format.md). Grouping the segments of one continuous playlist onto one song is a possible later refinement.
- **Game audio decoding on a real installation**: vgmstream decodes Wwise Vorbis in general and CDMW uses it for
  this game; the Studio's own tests could only use placeholder and PCM files. "Test decoding" on the Game Data page
  confirms it on a real install.
- **Listening thresholds**: the vocal/instrumental cut-offs were checked on public-domain recordings in CI
  (docs/research/listening.md); they can be tuned without listening again, because only embeddings are stored.
- **Music outside the interactive hierarchy** (played by plain `Sound` objects): currently reported, not offered as cues.
- **In-game verification** of the output and of DMM's handling of the Crimson Browser manifest.
- **Smaller audio**: Wwise ADPCM (about 4x smaller than PCM) once its block layout for this engine is verified.
- **Analyzer schema additions** that would help the Studio: a recorded game build/version (the
  installation check currently relies on file sizes/dates) and a populated `schema_meta` (generator and
  version). These are suggestions for the Analyzer project, not changes made here.
