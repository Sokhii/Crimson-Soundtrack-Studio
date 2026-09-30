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

Vocals: mean cosine to three "sung vocals" prompts minus three "instrumental" prompts; > 0.02 = sung vocals,
< -0.01 = instrumental, otherwise unclear. On public-domain recordings from Wikimedia Commons (CI diagnostics,
2026-09-30, with the text tower the app uses, identical to PyTorch): all six vocal recordings scored 0.048-0.204
(sung vocals); all four instrumental pieces scored -0.162 to -0.299 (instrumental). Only embeddings are stored, so thresholds and prompts can be re-tuned
without listening again.

Why embeddings plus zero-shot tags instead of free text: the profile vocabulary is fixed, CLAP scores exactly those
words, results are reproducible, and "sounds alike" (embedding similarity) is available for matching at no extra
cost.
