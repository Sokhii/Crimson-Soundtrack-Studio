# Third-party notices

Crimson Soundtrack Studio is MIT-licensed (see `LICENSE`). The portable build bundles:

| Component | Licence | Use |
|---|---|---|
| [Python](https://www.python.org/) | PSF License | runtime |
| [Qt for Python (PySide6)](https://www.qt.io/qt-for-python) / Qt 6 | LGPL-3.0 | GUI; shipped as separate, replaceable shared libraries in the one-folder build |
| [NumPy](https://numpy.org/) | BSD-3-Clause | signal analysis |
| [python-soundfile](https://github.com/bastibe/python-soundfile) | BSD-3-Clause | audio decoding |
| [cryptography](https://cryptography.io/) | Apache-2.0 / BSD-3-Clause | ChaCha20 decryption when reading original soundbanks |
| [python-lz4](https://github.com/python-lz4/python-lz4) | BSD-3-Clause | LZ4 decompression of game archive entries |
| [libsndfile](https://libsndfile.github.io/libsndfile/) (bundled by soundfile, with FLAC/Ogg/Vorbis/Opus) | LGPL-2.1 (FLAC/Ogg/Vorbis: BSD-style) | audio decoding; shipped as a separate shared library |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (`runtime/llama/`) | MIT | local AI runtime (separate program) |
| [ONNX Runtime](https://onnxruntime.ai/) (DirectML build on Windows, incl. `DirectML.dll`) | MIT | optional listening model |
| [tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 | listening model text prompts |
| [vgmstream](https://github.com/vgmstream/vgmstream) (`runtime/vgmstream/`) | ISC-style; its bundled decoder libraries under their own licences (libvorbis/libogg: BSD; mpg123, FFmpeg: LGPL, as separate DLLs) | decodes the game's Wwise audio for the read-only game-audio analysis (separate program) |

No GPL-licensed code is bundled (LGPL libraries are shipped as separate, replaceable shared libraries). FLAC metadata is read by the Studio's own parser
(`src/soundtrack_studio/library/flac_meta.py`) for that reason.

The Analyzer schema file `src/soundtrack_studio/testing/analyzer_schema_v1.sql` is copied from
[Crimson Desert Analyzer](https://github.com/Sokhii/Crimson-Desert-Analyzer) (MIT, same author).

No game files, music recordings or AI model weights are included in this repository or in releases.
The optional listening model (LAION CLAP `larger_clap_music_and_speech`, Apache-2.0, ONNX export by Xenova) is
downloaded from inside the application.
