# Mod format research (Phase 5 input)

This file collects **evidence** about how a Crimson Desert music mod must be built. Nothing here has been
implemented yet, and nothing should be implemented from assumptions. Every item carries its source and a
trust label: **verified** (checked against real game data), **community-confirmed** (documented by a
public tool), **unknown**.

## Known so far

Sourced from the Crimson Desert Analyzer's research notes (`docs/research/` in that repository), which cite
public MIT-licensed community tools (CDMW by Ratty123, crimson-desert-unpacker by lazorr410, CrimsonForge by
hzeem) and a scan of a real installation (2026-09-30):

| Fact | Trust |
|---|---|
| Crimson Desert ships Wwise bank version 150 (all 3,288 banks of the scanned install). | verified |
| Banks live at `sound/windows/<bank id>.bnk`, streamed media at `sound/windows/media/<source id>.wem` inside the PAZ archives. | verified |
| The main music banks are `bgm` (412724365) and `bgm_playlist` (2113151378); `1981912997.bnk` has identical counts to `bgm` (twin, name unknown). | verified |
| Music media: Wwise Vorbis, 48 kHz, mostly stereo; 1,129 prefetch-streamed, 4 streamed, 4 embedded; no `smpl` loop points (looping is done by the music hierarchy). | verified |
| For music media the WEM header duration equals the MusicTrack clip `fSrcDuration` within 0.022 s. | verified |
| `meta/0.papgt` lists mounted package directories in priority order; the first directory holding a path wins, which is how mods overlay files. | community-confirmed |
| PAMT is integrity-checked; the checksum chain runs PAPGT → PAMT. | community-confirmed |
| Overlay mods add a new numbered directory with its own `0.pamt`/`0.paz` and list it first in `meta/0.papgt`. | community-confirmed |

## Unknown, must be researched before the compiler is written

- The **current DMM** package layout, manifest and how it applies overlays (whether DMM generates the
  PAPGT/PAMT entries itself or expects them in the mod).
- Whether replacement WEMs must keep the original **duration**, or whether segment durations, markers
  (entry/exit) and track clip ranges in the BNK must be rewritten, and which of those the game tolerates.
- Wwise Vorbis **encoding** requirements (codebook set, packet format for bank v150) and whether the
  Wwise authoring tool is required or an open encoder is viable; licensing of any encoder used.
- **Prefetch streaming**: prefetch data embedded in banks must match the streamed file; how to regenerate it.
- Handling of media shared between several tracks/segments, and of the `bgm` twin bank.
- Whether existing working audio mods exist that can be legally inspected for layout (not content).
