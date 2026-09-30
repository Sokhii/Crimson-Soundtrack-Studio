"""What the Studio requires from a Crimson Desert Analyzer database.

The Analyzer tracks its schema with ``PRAGMA user_version`` (append-only
migrations, see the Analyzer's ``db/schema.py``). The Studio lists every
table and column it reads; anything else in the database is ignored, so
additive Analyzer changes do not break the Studio.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

SQLITE_MAGIC = b"SQLite format 3\x00"

# Analyzer schema versions this Studio build understands.
SUPPORTED_SCHEMA_VERSIONS: FrozenSet[int] = frozenset({1})
MIN_SCHEMA = min(SUPPORTED_SCHEMA_VERSIONS)
MAX_SCHEMA = max(SUPPORTED_SCHEMA_VERSIONS)

# Tables whose presence identifies an Analyzer database at all.
SIGNATURE_TABLES = ("installation", "scan", "wwise_object", "object_ref")

# table -> columns the Studio reads (must exist)
REQUIRED: Dict[str, FrozenSet[str]] = {
    "installation": frozenset({"id", "root_path", "label", "last_scanned"}),
    "scan": frozenset({"id", "installation_id", "started_at", "finished_at", "status", "mode", "parser_version"}),
    "source_file": frozenset({"installation_id", "rel_path", "kind", "size", "mtime_ns"}),
    "asset": frozenset({"id", "installation_id", "origin", "vpath", "ext", "size", "content_hash"}),
    "bnk": frozenset({"asset_id", "bank_id", "version", "object_count", "media_count"}),
    "wem": frozenset({"asset_id", "source_id", "container", "bank_asset_id", "valid", "codec", "channels",
                      "sample_rate", "duration_s", "loops_json"}),
    "wwise_object": frozenset({"bank_asset_id", "object_id", "type_code", "type_name", "parse_status", "fields_json"}),
    "object_ref": frozenset({"bank_asset_id", "from_object_id", "to_id", "kind", "confidence"}),
    "media_source": frozenset({"bank_asset_id", "object_id", "owner_type", "source_id", "stream_type", "details_json"}),
    "name": frozenset({"id_value", "name", "kind", "source", "hash_verified"}),
    "classification": frozenset({"installation_id", "entity_type", "entity_key", "role", "score", "confidence"}),
}

# Used when present; their absence is reported as a warning only.
OPTIONAL: Dict[str, FrozenSet[str]] = {
    "media_context": frozenset({"installation_id", "source_id", "event_ids_json", "bank_ids_json"}),
    "unknown_structure": frozenset({"installation_id", "status"}),
}

# Wwise HIRC type codes of the interactive-music hierarchy (bank v150).
MUSIC_SEGMENT = 10
MUSIC_TRACK = 11
MUSIC_SWITCH = 12
MUSIC_PLAYLIST = 13
MUSIC_TYPE_CODES = (MUSIC_SEGMENT, MUSIC_TRACK, MUSIC_SWITCH, MUSIC_PLAYLIST)
EVENT = 4
ACTION = 3

MUSIC_ROLES = ("music", "likely_music", "possible_music")
