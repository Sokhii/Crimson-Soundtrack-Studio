"""Local-AI semantic descriptions (structured, schema-constrained)."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..ai.runtime import BackendError, InferenceBackend, ModelOutputError
from .context import document_text
from .profile import CATEGORIES, ProfileParseError, SemanticProfile, json_schema, parse_llm_output

log = logging.getLogger(__name__)

PROMPT_VERSION = 3

SYSTEM = (
    "You are an experienced music supervisor. You describe the thematic and emotional character of music so it can "
    "be matched with other music of similar character. Describe mood, emotion, atmosphere, instrumentation and "
    "style using ONLY the allowed words, and rate energy, darkness, tension and valence (positivity) from 0 to 100. "
    "Base your answer on the evidence given. If you recognise the piece or its composer you may use that knowledge. "
    "Never invent facts: when the evidence is thin, give fewer tags and a low confidence (0-100). "
    "Do not classify by gameplay situation; describe the music itself. Reply with JSON only."
)

HEARD_RULES = (
    " If the evidence has 'heard', a listening model listened to the audio itself: trust what it heard over names, "
    "tags and genre. Its tags carry a score out of 100 (how clearly this piece stands out for that word compared "
    "with the rest of the music analysed; 85 and above is shown, 95+ is unmistakable). Use the highest-scoring words "
    "for your tags and put other heard words (rhythm, texture, specific instruments and genres) in 'themes'."
)
VOCAL_RULES = (
    " vocal_presence: if 'heard' says 'sung vocals' use true, if it says 'instrumental' use false. Otherwise use true "
    "only when the title or tags clearly show sung lyrics (e.g. a credited vocalist or singer), false only when they "
    "clearly say instrumental, and null in every other case. A genre, a soundtrack or OST album, an artist's name or "
    "an epic-sounding title is NOT evidence of vocals. Choirs and wordless voices belong in instrumentation."
)
TRACK_INSTRUCTIONS = (
    "Describe this music track from the user's library. The measurements come from signal analysis: "
    "'energy_index' and 'brightness_index' are 0-1 heuristics; tempo is an estimate and may be missing."
    + HEARD_RULES + VOCAL_RULES
)

CUE_INSTRUCTIONS = (
    "Describe this piece of music from the video game Crimson Desert (a dark, grounded medieval-fantasy open world). "
    "Nobody can listen to it here: the evidence is its Wwise structure and internal names (region/state names, "
    "event names, file names) and sometimes public community notes. Infer its likely musical character from that "
    "evidence and state your confidence honestly. When 'measurements' are present they were measured from the "
    "decoded game audio and describe its actual sound: prefer them over what names suggest."
    + HEARD_RULES + " vocal_presence: from 'heard' as above, otherwise null unless the evidence says so."
)


def vocab_text() -> str:
    return "\n".join(f"{name}: {', '.join(words)}" for name, words in CATEGORIES.items())


def build_messages(kind: str, doc: Dict[str, Any]) -> List[Dict[str, str]]:
    instructions = TRACK_INSTRUCTIONS if kind == "track" else CUE_INSTRUCTIONS
    user = (f"{instructions}\n\nAllowed words:\n{vocab_text()}\n\nEvidence:\n{document_text(doc)}\n\n"
            "Also give 'themes' (up to 8 short free words, e.g. 'loss', 'journey', 'wilderness') and a one-sentence "
            "'summary' of the music's character.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def describe(backend: InferenceBackend, kind: str, doc: Dict[str, Any], retries: int = 1) -> SemanticProfile:
    """Ask the model for a profile. Raises ProfileParseError/BackendError when no usable answer is obtained."""

    messages = build_messages(kind, doc)
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            reply = backend.chat(messages, json_schema=json_schema(), max_tokens=900,
                                 temperature=0.2 if attempt == 0 else 0.0)
            profile = parse_llm_output(reply)
        except ModelOutputError as exc:
            last_error = exc
            log.info("Model output incomplete (%s); %s", exc.details[:120], "retrying" if attempt < retries else "giving up")
            continue
        except ProfileParseError as exc:
            last_error = exc
            log.info("Model output rejected (%s); %s", exc, "retrying" if attempt < retries else "giving up")
            messages = messages + [{"role": "assistant", "content": reply[:2000]},
                                   {"role": "user", "content": f"That reply was not usable ({exc}). Reply with the JSON "
                                                               "object only, using the allowed words."}]
            continue
        profile.source = "llm"
        profile.model_id = backend.model_id
        apply_heard_vocals(profile, doc)
        return profile
    raise ProfileParseError(str(last_error))


def apply_heard_vocals(profile: SemanticProfile, doc: Dict[str, Any]) -> None:
    """What the listening model heard decides vocal presence; the text model cannot hear."""

    vocals = (doc.get("heard") or {}).get("vocals")
    if vocals == "sung vocals":
        profile.vocal_presence = True
    elif vocals == "instrumental":
        profile.vocal_presence = False
        if "solo voice" in profile.instrumentation:
            profile.instrumentation.remove("solo voice")


RERANK_SYSTEM = (
    "You are a music supervisor choosing replacement music for a video game. For each candidate, judge how well its "
    "thematic and emotional character fits the original game music. Reply with JSON only."
)


def rerank_schema(count: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"ranking": {"type": "array", "maxItems": count, "items": {
            "type": "object",
            "properties": {"candidate": {"type": "integer", "minimum": 1, "maximum": count},
                           "fit": {"type": "integer", "minimum": 0, "maximum": 100},
                           "reason": {"type": "string", "maxLength": 240}},
            "required": ["candidate", "fit", "reason"], "additionalProperties": False}}},
        "required": ["ranking"], "additionalProperties": False,
    }


def rerank(backend: InferenceBackend, cue_profile: SemanticProfile, cue_doc: Dict[str, Any],
           candidates: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Ask the model to judge shortlisted candidates. Returns {candidate index (0-based): {fit, reason}}."""

    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(f"{i}. {c['label']}: {c['profile'].describe()}; energy {c['profile'].energy}, darkness "
                     f"{c['profile'].darkness}; {c['profile'].summary[:160]}")
    user = (f"Original game music: {cue_profile.describe()}; energy {cue_profile.energy}, darkness "
            f"{cue_profile.darkness}, tension {cue_profile.tension}. {cue_profile.summary}\n"
            f"Game context: {json.dumps({k: cue_doc.get(k) for k in ('structure_names', 'plays_when', 'media_names')}, ensure_ascii=False)}\n\n"
            "Candidates:\n" + "\n".join(lines) +
            "\n\nRate each candidate's fit (0-100) with a short reason that refers to musical character.")
    reply = backend.chat([{"role": "system", "content": RERANK_SYSTEM}, {"role": "user", "content": user}],
                         json_schema=rerank_schema(len(candidates)), max_tokens=900, temperature=0.1)
    try:
        data = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
    except ValueError as exc:
        raise BackendError("The local AI returned an unusable ranking.", details=reply[:300]) from exc
    out: Dict[int, Dict[str, Any]] = {}
    for item in data.get("ranking", []) if isinstance(data, dict) else []:
        try:
            idx = int(item["candidate"]) - 1
            fit = max(0, min(100, int(item["fit"])))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates) and idx not in out:
            out[idx] = {"fit": fit / 100.0, "reason": str(item.get("reason", ""))[:240]}
    return out
