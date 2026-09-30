# Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 Foundation | project structure, GUI, portable paths, projects, settings, logging, Analyzer DB import/validation, game path selection and verification, game data browser | **done** |
| 2 Music library | recursive FLAC scanning, metadata, deterministic audio features, caching, music browser | **done** (FLAC only; further formats plug into `library/formats.py`) |
| 3 Local AI | embedded llama.cpp runtime (reusing the Analyzer's approach), model manager with Low/Medium/High/Custom tiers and resource estimates, in-app download to `models/` with SHA-256 verification, structured semantic metadata (mood, energy, darkness, tension, emotion, instrumentation, vocals, style, atmosphere, themes) stored separately from facts and editable, cached | planned |
| 4 Matching | semantic description of game cues, deterministic candidate filtering (duration, format, structure), semantic similarity, LLM reasoning only on shortlisted candidates, confidence and explanations, review UI, manual overrides that always win | planned |
| 5 Compiler research | current DMM format, current overlay/PAPGT/PAMT requirements, Wwise replacement requirements (WEM encoding, BNK/HIRC edits, streaming/prefetch, segment durations and markers), evidence documented in `docs/research/` before implementation | planned |
| 6 Compiler | build pipeline into `output/`, never touching the game installation | planned |
| 7 Release | end-to-end tests, versioning, release packaging, documentation | CI + portable ZIP already in place |

## Decisions still open (need evidence)

- **Replacement granularity**: segment vs playlist vs track layer (see architecture.md, *Cues*).
- **Music outside the interactive hierarchy** (played by plain `Sound` objects): include as cues or not.
- **Analyzer schema additions** that would help the Studio: a recorded game build/version (the
  installation check currently relies on file sizes/dates) and a populated `schema_meta` (generator and
  version). These are suggestions for the Analyzer project, not changes made here.
