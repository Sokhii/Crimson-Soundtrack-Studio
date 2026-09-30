# Game-audio decoding and the listening model: choices and evidence

Trust labels as in `modding_format.md`: **verified** (reproduced by our tests or CI), **source-confirmed** (read in
source code), **community-reported**.

## Why

The description model (llama.cpp) only reads text: tags, names and signal measurements. It cannot hear, so it
guessed things like vocals from genres and titles. Game music was described from its Wwise names alone. This
phase lets the Studio work with the audio itself on both sides.

## Decoding the game's music (read-only)

| Fact | Evidence | Trust |
|---|---|---|
| All 1,137 music sources of the current game are Wwise Vorbis, 26.4 h in total; 1,129 are prefetch-streamed `.wem` files, 4 embedded, 4 streamed. | the user's Analyzer export (`music_assets.json`) | verified (counted) |
| A streamed `.wem` holds the complete stream; the bank's prefetch copy is only its start. | Wwise bank format (wwiser), compiler research | source-confirmed |
| vgmstream decodes Wwise Vorbis (`src/meta/wwise.c`, `vorbis_custom` Wwise setups) and is ISC-licensed. | vgmstream source | source-confirmed |
| Crimson Desert Mod Workbench bundles vgmstream (r1980, pinned SHA-256) to preview this game's audio. | CDMW `build_pyside6_app.ps1`, `core/archive_media_preview.py` | source-confirmed |
| The bundled vgmstream decodes the Studio's own Wwise PCM output exactly and refuses placeholder files cleanly. | frozen self-test on Windows CI (`game_audio_decoder_works`) | verified |
| Real Wwise Vorbis from the game could not be tested here (no game files). | - | open: "Test decoding" confirms it on a real install |

Design: read through the Analyzer's archive records (ChaCha20/LZ4 as in the compiler), verify the Analyzer's hash
(full or 256 KiB-prefix SHA-1), decode with `vgmstream-cli -i` (loop points ignored: the stream plays once) into
`temp/gameaudio/`, measure, delete. Only numbers are kept, keyed by content hash.

## The listening model

| Fact | Evidence | Trust |
|---|---|---|
| LAION's CLAP checkpoints `larger_clap_music` and `larger_clap_music_and_speech` are Apache-2.0. | Hugging Face model metadata (read in CI) | verified |
| Only `larger_clap_music_and_speech` has a ready ONNX export (`Xenova/…`, revision `e9fd5ac`): audio tower 282 MB (fp32), text tower 251 MB (fp16; the 127 MB int8 variant was rejected, see below). | Hugging Face listing (read in CI) | verified |
| Preprocessing: 48 kHz, 64 Slaney mel bands 50-14000 Hz, 1024-point FFT, hop 480, 10 s input, `rand_trunc` + `repeatpad`. | model `preprocessor_config.json` | verified |
| Our NumPy front end equals transformers' `ClapFeatureExtractor`: max difference 7.6e-6 dB. | `listening-reference` CI job | verified |
| Our tokenizer (tokenizers + `tokenizer.json`) produces the same ids as transformers' `RobertaTokenizer`. | `listening-reference` CI job | verified |
| ONNX audio embeddings match PyTorch (cosine 1.00000). The int8 text tower did not (batched min cosine 0.57, one prompt at a time min 0.92 / mean 0.98); the fp32 and fp16 towers match exactly (1.0000), so the fp16 one is used. | `listening-reference` CI job | verified |
| llama.cpp can run audio-input models (Ultravox, Voxtral, Qwen2.5/3-Omni, Gemma 4), but those are large or speech-oriented; the user preferred a small music-specific model. | llama.cpp `docs/multimodal.md` | source-confirmed |
| Essentia's voice/instrumental classifier is accurate but CC BY-NC-SA (non-commercial). | Essentia model documentation | community-reported; not used |

Vocals (what the app ships, fitted on freely licensed recordings in CI, 2026-09-30): each ten-second excerpt gets
c = (mean cosine to six "sung" prompts - mean cosine to three "instrumental" prompts) + 3 x the mean of three
contrasting pairs ("a song with vocals" / "a song without vocals", "singing" / "no singing", "music with a singer" /
"music without a singer"). An excerpt with c > 0.05 counts as singing; a track is "sung vocals" when a third of its
excerpts do, "instrumental" when none do, otherwise "unclear" (no claim). Averaging a whole song first, and the
descriptive prompts alone, were the earlier versions: they missed modern synthesized voices (Vocaloid: all six
excerpts on the instrumental side) and, in the reported case, an anime song with obvious vocals. The pairs alone
find nearly every voice but also fire on piano, jazz and old orchestral recordings; the two parts make different
mistakes, so the sum separates best (33 recordings: 23 of 24 sung files found, including Vocaloid, AI-made songs,
Japanese songs and a modern Japanese vocal track; 8 of 9 instrumentals kept; the exception is a 1924 jazz orchestra
recording; Scott Joplin's 1916 ragtime is "unclear"). The recordings are few and mostly older or AI-made, so the cut-off
is provisional; `CrimsonSoundtrackStudio.exe --listen-file FILE` prints the per-excerpt scores of any file so
it can be checked against real music. Only embeddings are stored, so prompts and cut-offs can be re-tuned without
listening again.

Why embeddings plus zero-shot tags instead of free text: the profile vocabulary is fixed, CLAP scores exactly those
words, results are reproducible, and "sounds alike" (embedding similarity) is available for matching at no extra
cost.
