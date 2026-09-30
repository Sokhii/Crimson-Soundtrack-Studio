# Mod format and Wwise replacement: evidence (Phase 5)

This document records **what the compiler relies on and why**. Every item has a source and a trust
label:

- **verified**: checked against real game data or reproduced by our own tests;
- **source-confirmed**: read in the source code of a public tool;
- **community-reported**: stated by public community documentation or changelogs;
- **inferred**: a reasoned conclusion from the above; stated as such;
- **unknown**: not established; the Studio must not depend on it.

Research date: 2026-09-30. Game version at that time: 2.0.x (update of 2026-08-26 and later patches).

## 1. Mod managers and the package layout

| Fact | Source | Trust |
|---|---|---|
| DMM (Definitive Mod Manager) is the primary Crimson Desert mod manager; it handles "JSON patches, textures, file replacements, audio mods, ASI plugins" and "audio soundbank injection for .bnk files". | Nexus/Modsportal listings (via web search; the Nexus page itself was not reachable from the research environment) | community-reported |
| DMM's source repository (`exodiaprivate-eng/Definitive-Mod-Manager`) returned HTTP 404 during research; its exact import rules could not be read. | direct fetch | verified (unavailable) |
| Mods that contain game package folders `0000`–`0035` are routed to DMM as file-replacement mods. | Vortex extension changelog 0.4.0 (2026-05-19), ChemBoy1 | community-reported |
| Crimson Browser format: `manifest.json` with `{"format": "crimson_browser_mod_v1", "id": ..., "files_dir": "files"}` plus loose files at `files/<package>/<virtual path>`; the manager maps each file to its PAMT entry and repacks it into its own overlay. | CDUMM `engine/crimson_browser_handler.py` (MIT) | source-confirmed |
| Game update 2.0 (2026-08-26) reserved overlay slots `0036`–`0040` for language packs; an overlay placed there is treated as a missing language pack and **never loads**. Managers now use slot `0041`+. | CDUMM changelog v3.16.1/v3.16.3/v3.16.5 | community-reported |
| Overlay packages (`NNNN/0.paz`, `NNNN/0.pamt`, `meta/0.papgt`) are integrity-checked (PAPGT → PAMT checksum chain) and slot numbers changed between game versions. | Analyzer research (CDMW `papgt_format.py`), CDUMM changelog | community-reported |
| Existing soundtrack replacement mods (e.g. "Crimson Tamriel V2") replace **WEM and BNK files** and are installed with DMM. | Nexus listing (via web search) | community-reported |

**Decision.** The Studio outputs a **file-replacement mod** and lets the mod manager build the overlay:

```
<Mod name>/
  manifest.json            crimson_browser_mod_v1 (+ name, version, author, description)
  files/<package>/<virtual path of each replaced file>
  README.txt, build_report.json
```

The package folder and virtual path of every file come from the Analyzer database (`archive_entry`),
never from assumptions. The Studio **does not** write PAZ/PAMT/PAPGT itself: those depend on the exact
game build and slot rules that managers track, and a hard-coded slot (the `0036/` layout in older
guides) is broken on 2.0. A second export layout without `manifest.json` (`<package>/<path>` at the top
level, as described for DMM by the Vortex changelog) is offered for managers that expect it.
*Unverified:* DMM's handling of the Crimson Browser manifest could not be read from source; users of DMM
should report which layout it imports.

## 2. Game archives (reading original banks)

| Fact | Source | Trust |
|---|---|---|
| The Analyzer DB records every archive entry: package, virtual path, PAZ index, offset, stored/original size, flags. | Analyzer schema v1 (`archive_entry`) | verified |
| `flags` low nibble = compression (2 = LZ4 block), next nibble = encryption (3 = ChaCha20). | Analyzer research (CDMW) | community-reported; LZ4 verified by our tests against Analyzer-built archives |
| ChaCha20 key/nonce are derived from the lowercase basename via lookup3 `hashlittle(initval 0x000C5EDE)`. | Analyzer research + published test vector | verified (test vector reproduced in the Analyzer and in our tests) |
| Order: compress, then encrypt; reading = decrypt, then decompress. | Analyzer research | community-reported |
| The Analyzer stores the SHA-1 of the full decoded bytes of every bank (`asset.content_hash`, `hash_kind = 'sha1'`). | Analyzer `scanner.py` | source-confirmed |

The compiler reads only the banks it modifies, using the Analyzer's entry records, and refuses to
continue unless the decoded bytes match the Analyzer's SHA-1. This proves both that the decoding is right
and that the game still matches the database. Nothing in the game folder is written.

## 3. Wwise structure and replacement granularity

| Fact | Source | Trust |
|---|---|---|
| Crimson Desert uses Wwise bank version 150 for all banks. | Analyzer real-install scan | verified |
| Music hierarchy: MusicSwitchContainer → MusicRandomSequenceContainer (playlist) → MusicSegment (duration, entry/exit markers, tempo grid) → MusicTrack (clips on the segment timeline: source, play_at, begin/end trim, source duration) → media. | Analyzer decoders, real scan (232,251 objects parsed, 0 failed) | verified |
| For music media, the WEM header duration equals the clip's source duration (`fSrcDuration`) within 0.022 s. | Analyzer real scan | verified |
| Several MusicTracks in one segment play simultaneously (layers/stems; e.g. `stem1..4`, `1normal/2tensioned` names). | Wwise interactive-music model; stem names in real data | inferred |

**Decision: the replaceable unit is the MusicSegment.** The user's audio is rendered onto the segment's
own timeline (its exact duration), and every clip source of the segment's primary track receives exactly
the slice of that timeline it plays (`timeline[play_at : play_at + source_duration]`). Other layer tracks
receive silence of their own source duration, so stems cannot clash with the new music. Because every
duration, marker, transition and playlist rule is left untouched, the interactive structure is preserved.
Segments shorter than 15 s and transition segments keep their original audio unless the user includes them.

## 4. Media encoding and bank changes

| Fact | Source | Trust |
|---|---|---|
| Codec plugin IDs: `0x00010001` PCM, `0x00020001` ADPCM, `0x00040001` Vorbis. | wwiser `parser/wdefs.py` | source-confirmed |
| v150 `AkBankSourceData`: `u32 pluginID; u8 StreamType (0 in bank/"data", 1 prefetch-streaming, 2 streaming); AkMediaInformation { u32 sourceID; u32 uInMemoryMediaSize; u8 uSourceBits (bit0 language-specific, bit1 prefetch, bit3 non-cachable, bit7 has-source) }`; codec plugins have no extra parameter block. | wwiser `parser/wparser.py` (`CAkBankMgr__LoadSource`, versions 113-150) | source-confirmed |
| Music media: Wwise Vorbis 48 kHz; 1,129 prefetch-streamed, 4 streamed, 4 in the bank. | Analyzer real scan | verified |
| Wwise PCM `.wem`: RIFF/WAVE, 16-bit little-endian, interleaved; `fmt ` size 0x10/0x12/0x18/0x28; format tag 0x0001 (older) or 0xFFFE; newer files carry a channel-config word at `fmt+0x14` (`numChannels | configType<<8 | channelMask<<12`). | vgmstream `meta/wwise.c` | source-confirmed |
| Community audio mods rebuild audio as PCM WEM. | web search summary of community guides | community-reported |
| CDMW's own WEM replacement writes a plain PCM RIFF but does **not** change the bank ("best-effort, not a complete Wwise rebuild"). | CDMW `core/archive_audio.py`, changelog | source-confirmed |

**Why the bank must change.** A prefetch-streamed source keeps the first part of the *original Vorbis*
stream inside the bank, and the bank declares the Vorbis codec. Replacing only the streamed file would
leave a codec mismatch and stale prefetch data (this is why a WEM-only replacement is "best-effort").

**Decision.** No proprietary encoder is needed or bundled:

- replacement audio is written as **16-bit PCM WEM** at 48 kHz with the original channel count, in the
  vgmstream-documented Wwise layout (`fmt ` 0x18, tag 0xFFFE, channel-config word);
- for each replaced source, in **every bank that contains it** (the `bgm` bank has a twin with identical
  objects): plugin ID → PCM (`0x00010001`);
  - streamed/prefetch sources → StreamType 2 (streaming), in-memory size 0, prefetch bit cleared, the stale
    prefetch copy removed from `DIDX`/`DATA`, and the new `.wem` written to the original media path;
  - in-bank sources → the PCM data replaces the old data in `DATA`; StreamType 0; in-memory size = new size;
- the patch is located inside each MusicTrack by the exact byte pattern of the fields above, cross-checked
  against the Analyzer's decoded values; a mismatch aborts the build instead of guessing;
- object sizes do not change (fixed-size fields), so no HIRC offsets move; only `DIDX`/`DATA` are rebuilt,
  keeping the original ordering and alignment.

**Trade-off.** PCM is large (about 11.5 MB per stereo minute). A full soundtrack replacement can reach
several GB. ADPCM would be about 4x smaller but its exact Wwise block layout for this engine version is
not yet verified, so it is not used.

## 5. Unknown / not verified in game

- In-game playback of the Studio's output has **not** been tested (no game in the development
  environment). The format decisions above follow verified structure and public tools; the first real
  test must be done by a user, and the Studio records everything needed for a bug report in
  `build_report.json`.
- Whether DMM imports the Crimson Browser manifest (see section 1). Both export layouts are available.
- Loudness of the original music (Vorbis is not decoded), so replacements are normalised to a common
  level (default −18 dBFS RMS, peaks limited to −1 dBFS) rather than matched to the originals.
