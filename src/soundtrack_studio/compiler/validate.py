"""Independent validation of a built mod folder.

Re-reads everything that was written, without using the build's in-memory
state: manifest, every soundbank (structure, patched sources, media index) and
every ``.wem`` (format, channels, rate, exact length).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from . import bnk, wem
from .archive import CompileError

# The Studio's own report. Not named build_report.json: DMM reads a file of that name in every mod folder as its
# own format and logs a parse error for ours ('expected u32' at the "format" field).
REPORT_NAME = "css_build_report.json"


@dataclass
class ValidationResult:
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_banks: int = 0
    checked_wems: int = 0

    def error(self, text: str) -> None:
        self.ok = False
        self.errors.append(text)


def validate_output(mod_dir: Path, expected: Dict[str, Any]) -> ValidationResult:
    """``expected`` comes from css_build_report.json: files, sources (id -> stream/frames/channels/banks)."""

    result = ValidationResult()
    files_dir = mod_dir / expected.get("files_dir", "files")
    manifest_path = mod_dir / "manifest.json"
    if expected.get("layout") == "crimson_browser":
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") != "crimson_browser_mod_v1" or manifest.get("files_dir") != "files":
                result.error("manifest.json does not describe a crimson_browser_mod_v1 package")
        except (OSError, ValueError) as exc:
            result.error(f"manifest.json unreadable: {exc}")
    metadata = {"manifest.json", REPORT_NAME, "build_report.json", "README.txt"}  # package-level files, not game files
    written = ({p.relative_to(files_dir).as_posix() for p in files_dir.rglob("*") if p.is_file()} - metadata
               if files_dir.is_dir() else set())
    listed = set(expected.get("files", []))
    for missing in sorted(listed - written):
        result.error(f"missing output file {missing}")
    for extra in sorted(written - listed):
        result.error(f"unexpected output file {extra}")

    sources = {int(k): v for k, v in expected.get("sources", {}).items()}
    banks: Dict[str, bytes] = {}
    for rel in sorted(listed & written):
        path = files_dir / rel
        if rel.lower().endswith(".bnk"):
            try:
                data = path.read_bytes()
                chunks = bnk.parse_chunks(data)
                if bnk.bank_version(chunks) != 150:
                    result.error(f"{rel}: unexpected bank version")
                entries = bnk.didx_entries(chunks)
                data_chunk = next((c for c in chunks if c.tag == b"DATA"), None)
                for mid, off, size in entries:
                    if data_chunk is None or off + size > len(data_chunk.payload):
                        result.error(f"{rel}: media {mid} points outside DATA")
                banks[rel] = data
                result.checked_banks += 1
            except CompileError as exc:
                result.error(f"{rel}: {exc.message} {exc.details}")
        elif rel.lower().endswith(".wem"):
            sid = int(Path(rel).stem) if Path(rel).stem.isdigit() else None
            try:
                info = wem.read_wem_info(path.read_bytes())
            except CompileError as exc:
                result.error(f"{rel}: {exc.message}")
                continue
            result.checked_wems += 1
            spec = sources.get(sid, {}) if sid is not None else {}
            _check_wem(result, rel, info, spec)

    for sid, spec in sources.items():
        vorbis = spec.get("codec") == "vorbis"
        for bank_rel in spec.get("banks", []):
            data = banks.get(bank_rel)
            if data is None:
                continue
            refs = bnk.describe_sources(data).get(sid, [])
            if not refs:
                result.error(f"{bank_rel}: source {sid} not found after patching")
                continue
            chunks = bnk.parse_chunks(data)
            embedded = bnk.media_data(chunks, sid)
            if vorbis:
                _check_vorbis_bank(result, bank_rel, sid, refs, embedded, spec, files_dir)
                continue
            for ref in refs:
                if ref.plugin_id != bnk.PLUGIN_PCM:
                    result.error(f"{bank_rel}: source {sid} is not PCM (0x{ref.plugin_id:08x})")
                if ref.stream_type != spec["stream_type"]:
                    result.error(f"{bank_rel}: source {sid} storage {ref.stream_type}, expected {spec['stream_type']}")
                if ref.bits & bnk.BIT_PREFETCH:
                    result.error(f"{bank_rel}: source {sid} still marked as prefetch")
            if spec["stream_type"] == 0:
                if embedded is None:
                    result.error(f"{bank_rel}: in-bank source {sid} has no data")
                else:
                    try:
                        _check_wem(result, f"{bank_rel}#{sid}", wem.read_wem_info(embedded), spec)
                    except CompileError as exc:
                        result.error(f"{bank_rel}#{sid}: {exc.message}")
                    if refs and refs[0].in_memory_size != len(embedded):
                        result.error(f"{bank_rel}: source {sid} size field does not match its data")
            elif embedded is not None:
                result.error(f"{bank_rel}: stale prefetch data for streamed source {sid}")
    return result


def _check_vorbis_bank(result: ValidationResult, bank_rel: str, sid: int, refs, embedded: bytes, spec: Dict[str, Any],
                       files_dir: Path) -> None:
    """Vorbis builds keep the game's codec and storage; the in-bank copy must be the new file (or its beginning)."""

    streamed = None
    if spec.get("file"):
        try:
            streamed = (files_dir / spec["file"]).read_bytes()
        except OSError:
            streamed = None
    for ref in refs:
        if ref.plugin_id != bnk.PLUGIN_VORBIS:
            result.error(f"{bank_rel}: source {sid} is not Vorbis (0x{ref.plugin_id:08x})")
        if ref.stream_type == bnk.STREAM_STREAMING:
            continue
        if embedded is None:
            result.error(f"{bank_rel}: source {sid} has no {'prefetch ' if ref.stream_type else ''}data in the bank")
            continue
        if ref.in_memory_size != len(embedded):
            result.error(f"{bank_rel}: source {sid} size field does not match its data")
        if ref.stream_type == bnk.STREAM_IN_BANK:
            try:
                _check_wem(result, f"{bank_rel}#{sid}", wem.read_wem_info(embedded), spec)
            except CompileError as exc:
                result.error(f"{bank_rel}#{sid}: {exc.message}")
        elif streamed is not None and not streamed.startswith(embedded):
            result.error(f"{bank_rel}: prefetch data of source {sid} is not the beginning of {spec['file']}")


VORBIS_LENGTH_TOLERANCE = 2048      # samples; one Vorbis long block


def _check_wem(result: ValidationResult, rel: str, info: wem.WemInfo, spec: Dict[str, Any]) -> None:
    if spec.get("codec") == "vorbis":
        _check_vorbis_wem(result, rel, info, spec)
        return
    if info.format_tag != wem.FORMAT_WWISE_PCM or info.bits_per_sample != 16:
        result.error(f"{rel}: not 16-bit Wwise PCM")
    if info.sample_rate != 48000:
        result.error(f"{rel}: sample rate {info.sample_rate}")
    if spec:
        if info.channels != spec["channels"]:
            result.error(f"{rel}: {info.channels} channels, expected {spec['channels']}")
        if info.frames != spec["frames"]:
            result.error(f"{rel}: {info.frames} samples, expected {spec['frames']} (the segment timing would change)")


def _check_vorbis_wem(result: ValidationResult, rel: str, info: wem.WemInfo, spec: Dict[str, Any]) -> None:
    if not info.is_vorbis:
        result.error(f"{rel}: not Wwise Vorbis (format 0x{info.format_tag:04X})")
        return
    if info.sample_rate != 48000:
        result.error(f"{rel}: sample rate {info.sample_rate}")
    if info.channels != spec.get("channels", info.channels):
        result.error(f"{rel}: {info.channels} channels, expected {spec['channels']}")
    expected = spec.get("frames")
    if expected is not None and abs(info.frames - expected) > VORBIS_LENGTH_TOLERANCE:
        result.error(f"{rel}: {info.frames} samples, expected {expected} (the segment timing would change)")
    elif expected is not None and info.frames != expected:
        result.warnings.append(f"{rel}: {info.frames} samples instead of {expected} (within one Vorbis block)")
