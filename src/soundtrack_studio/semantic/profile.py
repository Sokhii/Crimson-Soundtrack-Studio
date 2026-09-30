"""Structured semantic description of a piece of music (a user track or a game cue).

Profiles are *interpretations*, stored separately from deterministic facts
and always tagged with their ``source``:

* ``rules`` - deterministic rules over measurements, tags and names (always available)
* ``llm``   - the local AI model (``model_id`` + ``prompt_version`` recorded)
* ``user``  - the user's edits (always take precedence)

Tags use controlled vocabularies so profiles from different sources can be
compared reliably; free-form ``themes`` carry anything else. Scales are 0..1.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

MOODS = ("dark", "mysterious", "melancholic", "peaceful", "dramatic", "adventurous", "tense", "heroic", "ominous",
         "triumphant", "romantic", "playful", "somber", "hopeful", "epic", "serene", "nostalgic", "aggressive",
         "whimsical", "solemn")
EMOTIONS = ("sadness", "joy", "fear", "anger", "awe", "calm", "longing", "determination", "suspense", "wonder",
            "grief", "triumph", "tenderness", "dread", "hope", "loneliness")
ATMOSPHERES = ("atmospheric", "cinematic", "ethereal", "intimate", "vast", "gritty", "mystical", "pastoral", "urban",
               "sacred", "desolate", "warm", "cold", "chaotic", "dreamlike", "ancient")
INSTRUMENTS = ("orchestra", "strings", "brass", "woodwinds", "choir", "piano", "guitar", "percussion", "drums",
               "synthesizer", "electronic beats", "ethnic instruments", "solo voice", "ambient pads", "organ", "harp",
               "bells", "flute", "cello", "violin", "bass")
STYLES = ("orchestral", "electronic", "ambient", "folk", "rock", "metal", "choral", "chamber", "minimal", "hybrid",
          "jazz", "classical", "pop", "world", "medieval", "cinematic", "celtic", "industrial", "acoustic")
SCALES = ("energy", "darkness", "tension", "valence")

CATEGORIES = {"mood": MOODS, "emotion": EMOTIONS, "atmosphere": ATMOSPHERES, "instrumentation": INSTRUMENTS,
              "style": STYLES}
MAX_TAGS = 6
MAX_THEMES = 8

# common synonyms the model (or tags) produce -> vocabulary term
SYNONYMS = {
    "sad": "melancholic", "melancholy": "melancholic", "calm": "peaceful", "tranquil": "serene", "relaxing": "peaceful",
    "scary": "ominous", "creepy": "ominous", "eerie": "mysterious", "suspenseful": "tense", "intense": "dramatic",
    "happy": "hopeful", "joyful": "playful", "uplifting": "hopeful", "angry": "aggressive", "brooding": "dark",
    "grim": "somber", "majestic": "epic", "grand": "epic", "powerful": "epic", "noble": "heroic", "sorrowful": "melancholic",
    "orchestral strings": "strings", "string section": "strings", "synth": "synthesizer", "synths": "synthesizer",
    "drum": "drums", "beats": "electronic beats",
    "pads": "ambient pads", "full orchestra": "orchestra", "symphonic": "orchestral", "film score": "cinematic",
    "soundtrack": "cinematic", "score": "cinematic", "folk instruments": "ethnic instruments",
    "suspense": "suspense", "fearful": "fear", "awe-inspiring": "awe", "atmosphere": "atmospheric",
}


@dataclass
class SemanticProfile:
    mood: List[str] = field(default_factory=list)
    emotion: List[str] = field(default_factory=list)
    atmosphere: List[str] = field(default_factory=list)
    instrumentation: List[str] = field(default_factory=list)
    style: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    energy: Optional[float] = None
    darkness: Optional[float] = None
    tension: Optional[float] = None
    valence: Optional[float] = None
    vocal_presence: Optional[bool] = None
    confidence: float = 0.3
    summary: str = ""
    source: str = "rules"
    model_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticProfile":
        known = set(cls.__dataclass_fields__)
        return normalize(cls(**{k: v for k, v in (data or {}).items() if k in known}))

    def tags(self) -> Dict[str, List[str]]:
        return {k: list(getattr(self, k)) for k in CATEGORIES}

    def describe(self) -> str:
        words = self.mood[:3] + self.atmosphere[:2] + self.style[:2] + self.instrumentation[:2]
        return ", ".join(dict.fromkeys(words)) or "no description"


def _clean_tag(tag: Any) -> str:
    text = re.sub(r"\s+", " ", str(tag or "").strip().lower())
    text = re.sub(r"[^a-z0-9 \-']", "", text)
    return SYNONYMS.get(text, text)


def _clamp(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    if number > 1.0:  # tolerate 0-100 or 0-10 scales from models
        number = number / 100.0 if number > 10 else number / 10.0
    return round(max(0.0, min(1.0, number)), 3)


def normalize(p: SemanticProfile) -> SemanticProfile:
    """Map tags to the vocabularies, clamp scales, dedupe; unknown tags move to ``themes``."""

    extras: List[str] = []
    for category, vocab in CATEGORIES.items():
        kept: List[str] = []
        for raw in getattr(p, category) or []:
            tag = _clean_tag(raw)
            if not tag:
                continue
            if tag in vocab:
                if tag not in kept:
                    kept.append(tag)
            else:
                # a tag from another category is still useful there
                placed = False
                for other, other_vocab in CATEGORIES.items():
                    if other != category and tag in other_vocab:
                        target = getattr(p, other)
                        if tag not in target:
                            target.append(tag)
                        placed = True
                        break
                if not placed:
                    extras.append(tag)
        setattr(p, category, kept[:MAX_TAGS])
    themes = []
    for raw in list(p.themes or []) + extras:
        tag = _clean_tag(raw)[:40]
        if tag and tag not in themes:
            themes.append(tag)
    p.themes = themes[:MAX_THEMES]
    for scale in SCALES:
        setattr(p, scale, _clamp(getattr(p, scale)))
    if not isinstance(p.vocal_presence, bool):
        p.vocal_presence = {"true": True, "yes": True, "false": False, "no": False}.get(
            str(p.vocal_presence).lower()) if p.vocal_presence is not None else None
    p.confidence = _clamp(p.confidence) if _clamp(p.confidence) is not None else 0.3
    p.summary = re.sub(r"\s+", " ", str(p.summary or "")).strip()[:400]
    return p


def merge(base: SemanticProfile, override: Dict[str, Any]) -> SemanticProfile:
    """Apply a partial user override (only the fields the user changed) on top of a profile."""

    data = base.to_dict()
    for key, value in (override or {}).items():
        if key in data and key not in ("source", "model_id"):
            data[key] = value
    merged = SemanticProfile.from_dict(data)
    merged.source = "user" if override else base.source
    merged.model_id = base.model_id
    if override:
        merged.confidence = 1.0
    return merged


# ------------------------------------------------------------ comparison
CATEGORY_WEIGHTS = {"mood": 0.30, "atmosphere": 0.15, "style": 0.12, "instrumentation": 0.12, "emotion": 0.11}
SCALE_WEIGHTS = {"darkness": 0.07, "energy": 0.06, "tension": 0.05, "valence": 0.02}
# moods that are close to each other count partially (symmetric)
RELATED = {
    ("dark", "ominous"), ("dark", "somber"), ("dark", "mysterious"), ("mysterious", "ominous"), ("tense", "ominous"),
    ("tense", "dramatic"), ("dramatic", "epic"), ("epic", "heroic"), ("heroic", "triumphant"), ("adventurous", "heroic"),
    ("adventurous", "epic"), ("peaceful", "serene"), ("melancholic", "somber"), ("melancholic", "nostalgic"),
    ("romantic", "nostalgic"), ("hopeful", "triumphant"), ("playful", "whimsical"), ("solemn", "somber"),
    ("aggressive", "tense"), ("orchestral", "cinematic"), ("orchestra", "strings"), ("choir", "choral"),
    ("ambient", "minimal"), ("electronic", "industrial"), ("folk", "celtic"), ("folk", "world"), ("medieval", "folk"),
    ("atmospheric", "ethereal"), ("vast", "cinematic"), ("desolate", "cold"), ("mystical", "ethereal"),
    ("awe", "wonder"), ("fear", "dread"), ("suspense", "dread"), ("sadness", "grief"), ("longing", "loneliness"),
}
RELATED |= {(b, a) for a, b in RELATED}


RELATED_MAP: Dict[str, frozenset] = {}
for _a, _b in RELATED:
    RELATED_MAP[_a] = RELATED_MAP.get(_a, frozenset()) | {_b}


def _set_similarity(a: Iterable[str], b: Iterable[str]) -> Optional[float]:
    """Symmetric tag overlap: exact match 1.0, related tag 0.5 (see RELATED)."""

    a_set, b_set = set(a), set(b)
    if not a_set or not b_set:
        return None
    empty = frozenset()
    score = 0.0
    for x in a_set:
        score += 1.0 if x in b_set else (0.5 if RELATED_MAP.get(x, empty) & b_set else 0.0)
    for y in b_set:
        score += 1.0 if y in a_set else (0.5 if RELATED_MAP.get(y, empty) & a_set else 0.0)
    return score / (len(a_set) + len(b_set))


def similarity(a: SemanticProfile, b: SemanticProfile) -> Tuple[float, Dict[str, float]]:
    """0..1 similarity plus per-component scores (components missing on either side are skipped)."""

    parts: Dict[str, float] = {}
    total_weight = 0.0
    total = 0.0
    for category, weight in CATEGORY_WEIGHTS.items():
        s = _set_similarity(getattr(a, category), getattr(b, category))
        if s is not None:
            parts[category] = round(s, 3)
            total += weight * s
            total_weight += weight
    for scale, weight in SCALE_WEIGHTS.items():
        x, y = getattr(a, scale), getattr(b, scale)
        if x is not None and y is not None:
            s = 1.0 - abs(x - y)
            parts[scale] = round(s, 3)
            total += weight * s
            total_weight += weight
    if a.vocal_presence is not None and b.vocal_presence is not None:
        s = 1.0 if a.vocal_presence == b.vocal_presence else 0.0
        parts["vocal_presence"] = s
        total += 0.05 * s
        total_weight += 0.05
    if total_weight == 0:
        return 0.0, parts
    # how much of the profiles could actually be compared (1.0 = every tag group and scale on both sides)
    parts["coverage"] = round(total_weight / (sum(CATEGORY_WEIGHTS.values()) + sum(SCALE_WEIGHTS.values()) + 0.05), 3)
    return round(total / total_weight, 4), parts


def json_schema() -> Dict[str, Any]:
    """Schema used to constrain the local model's output (enums keep tags inside the vocabularies)."""

    def tag_list(vocab):
        return {"type": "array", "items": {"type": "string", "enum": list(vocab)}, "maxItems": MAX_TAGS}

    scale = {"type": "integer", "minimum": 0, "maximum": 100}
    return {
        "type": "object",
        "properties": {
            "mood": tag_list(MOODS), "emotion": tag_list(EMOTIONS), "atmosphere": tag_list(ATMOSPHERES),
            "instrumentation": tag_list(INSTRUMENTS), "style": tag_list(STYLES),
            "themes": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": MAX_THEMES},
            "energy": scale, "darkness": scale, "tension": scale, "valence": scale,
            "vocal_presence": {"type": ["boolean", "null"]},
            "confidence": scale,
            "summary": {"type": "string", "maxLength": 300},
        },
        "required": ["mood", "emotion", "atmosphere", "instrumentation", "style", "themes", "energy", "darkness",
                     "tension", "valence", "vocal_presence", "confidence", "summary"],
        "additionalProperties": False,
    }


class ProfileParseError(ValueError):
    pass


def parse_llm_output(text: str) -> SemanticProfile:
    """Parse and validate a model reply. Raises ProfileParseError for unusable output."""

    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S)
    if fence:
        cleaned = fence.group(1).strip()
    if not cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ProfileParseError("no JSON object in the reply")
        cleaned = cleaned[start:end + 1]
    try:
        data = json.loads(cleaned)
    except ValueError as exc:
        raise ProfileParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProfileParseError("reply is not a JSON object")
    for key in ("mood", "emotion", "atmosphere", "instrumentation", "style", "themes"):
        value = data.get(key, [])
        if isinstance(value, str):
            value = [v for v in re.split(r"[,;/]", value) if v.strip()]
        if not isinstance(value, list):
            raise ProfileParseError(f"'{key}' is not a list")
        data[key] = value
    profile = SemanticProfile.from_dict(data)
    if not (profile.mood or profile.atmosphere or profile.style) and all(
            getattr(profile, s) is None for s in SCALES):
        raise ProfileParseError("the reply contains no usable description")
    return profile
