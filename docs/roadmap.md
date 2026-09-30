# Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 Foundation | project structure, GUI, portable paths, projects, settings, logging, Analyzer DB import/validation, game path selection and verification, game data browser | **done** |
| 2 Music library | recursive FLAC scanning, metadata, deterministic audio features, caching, music browser | **done** (FLAC only; further formats plug into `library/formats.py`) |
| 3 Local AI | bundled llama.cpp `llama-server` runtime, model manager (Low/Medium/High/Custom tiers, hardware-based recommendation, resumable SHA-256-verified downloads into `models/<tier>/`), structured semantic profiles for user tracks and game cues (rule-based baseline + schema-constrained local AI), per-item resumable runs, shared response cache, user overrides that always win | **done** |
| 4 Matching | per-segment matching: eligibility and length filtering, coverage-aware semantic similarity + duration/tempo fit, optional local-AI judgement of the shortlist only, diversity-aware assignment, confidence with uncertainty warnings, Matching review page (accept / reject / choose / keep original / fit mode), decisions that reruns never change | **done** |
| 5 Compiler research | current DMM format, current overlay/PAPGT/PAMT requirements, Wwise replacement requirements (WEM encoding, BNK/HIRC edits, streaming/prefetch, segment durations and markers), evidence documented in `docs/research/` before implementation | planned |
| 6 Compiler | build pipeline into `output/`, never touching the game installation | planned |
| 7 Release | end-to-end tests, versioning, release packaging, documentation | CI + portable ZIP already in place |

## Decisions still open (need evidence)

- **Replacement granularity**: decided: the MusicSegment (see docs/research/modding_format.md). Grouping the segments of one continuous playlist onto one song is a possible later refinement.
- **Music outside the interactive hierarchy** (played by plain `Sound` objects): include as cues or not.
- **Analyzer schema additions** that would help the Studio: a recorded game build/version (the
  installation check currently relies on file sizes/dates) and a populated `schema_meta` (generator and
  version). These are suggestions for the Analyzer project, not changes made here.
