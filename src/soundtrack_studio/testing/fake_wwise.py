"""A stand-in for WwiseConsole.exe used by the tests (Wwise itself cannot run in CI).

It understands the two commands the Studio uses and writes files with the real Wwise Vorbis *header* layout
(format 0xFFFF, fmt 0x42 bytes, sample count, setup/audio offsets) around placeholder bytes, at a Vorbis-like
16 kB/s. The bytes are not decodable audio; they only exercise the build, bank and validation logic.
"""

from __future__ import annotations

from pathlib import Path

SCRIPT = r'''
import hashlib, os, struct, sys
import xml.etree.ElementTree as ET

args = sys.argv[1:]
if args[:1] == ["create-new-project"]:
    project = args[1]
    os.makedirs(os.path.dirname(project), exist_ok=True)
    open(project, "w").write("<WwiseDocument/>")
    print("project created")
    sys.exit(0)
if args[:1] != ["convert-external-source"]:
    print("unknown command", args, file=sys.stderr)
    sys.exit(2)
source = args[args.index("--source-file") + 1]
output = args[args.index("--output") + 1]
tree = ET.parse(source).getroot()
root = tree.get("Root")
os.makedirs(os.path.join(output, "Windows"), exist_ok=True)
for item in tree.findall("Source"):
    if os.environ.get("FAKE_WWISE_MISSING"):
        print("Conversion\tWarning\tAudioConversion_FileOpenError\tCan't open source or output file", file=sys.stderr)
        sys.exit(2)
    if os.environ.get("FAKE_WWISE_FAIL"):
        print("Error: conversion failed", file=sys.stderr)
        sys.exit(1)
    wav = open(os.path.join(root, item.get("Path")), "rb").read()
    channels, rate = struct.unpack_from("<HI", wav, 22)
    data_size = struct.unpack_from("<I", wav, 40)[0]
    frames = data_size // (channels * 2)
    tag = 0xFFFE if os.environ.get("FAKE_WWISE_PCM") else 0xFFFF
    seek, setup = 64, 217
    audio = bytearray()
    digest = hashlib.sha256(wav[44:44 + 4096]).digest()
    payload = max(400, frames * 16000 // 48000)
    while len(audio) < payload:
        chunk = (digest * 8)[:200]
        audio += struct.pack("<H", len(chunk)) + chunk
    data = bytes(seek) + bytes(setup) + bytes(audio)
    config = channels | (1 << 8) | ((3 if channels == 2 else 4) << 12)
    fmt = struct.pack("<HHIIHHHHI", tag, channels, rate, 16000, 0, 0, 0x30, 0, config)
    fmt += struct.pack("<IIIIII", frames, 0xD9, len(data) - seek, 0, seek, seek + setup)
    fmt += bytes(0x42 - len(fmt))
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    name = os.path.splitext(item.get("Path"))[0] + ".wem"
    open(os.path.join(output, "Windows", name), "wb").write(b"RIFF" + struct.pack("<I", len(body)) + body)
print("converted")
if os.environ.get("FAKE_WWISE_WARN"):
    print("Process completed with warning(s).")
    sys.exit(2)
'''


def install_fake_wwise(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "WwiseConsole.py"
    script.write_text(SCRIPT, encoding="utf-8")
    return script
