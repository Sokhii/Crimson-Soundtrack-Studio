"""Deterministic, rule-based semantic baseline.

Always available (no model needed) and fully explainable. It maps
measurements and words found in titles, tags and Wwise names onto the profile
vocabularies. Confidence is deliberately low: these are hints, and the local
AI or the user can refine them. Gameplay words ("battle", "boss", ...) are used
only as *supporting evidence* for musical character (tense, aggressive), never
as categories.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .profile import SemanticProfile, normalize

# word -> (profile field, tag)
LEXICON: Dict[str, List[Tuple[str, str]]] = {}


def _add(words: str, *pairs: Tuple[str, str]) -> None:
    for word in words.split():
        LEXICON.setdefault(word, []).extend(pairs)


_add("battle war fight combat assault clash siege boss duel attack fury rage",
     ("mood", "tense"), ("mood", "aggressive"), ("emotion", "determination"))
_add("dark shadow shadows night death dead doom abyss darkness curse cursed grave tomb crypt blood",
     ("mood", "dark"), ("mood", "ominous"))
_add("requiem funeral lament elegy mourning tears grief sorrow farewell goodbye",
     ("mood", "melancholic"), ("emotion", "grief"))
_add("peace peaceful calm dawn morning village home hearth lullaby rest meadow spring breeze sunrise",
     ("mood", "peaceful"), ("atmosphere", "pastoral"))
_add("love heart romance tender", ("mood", "romantic"), ("emotion", "tenderness"))
_add("memories memory remember nostalgia past childhood", ("mood", "nostalgic"), ("emotion", "longing"))
_add("hero heroes glory victory triumph champion king kingdom honor honour legend", ("mood", "heroic"),
     ("mood", "triumphant"))
_add("mystery mysterious secret secrets ancient ruins riddle unknown forgotten ghost spirit spirits",
     ("mood", "mysterious"), ("atmosphere", "mystical"))
_add("journey adventure voyage travel road quest wander wanderer explore exploration", ("mood", "adventurous"))
_add("epic legendary titan colossus rising", ("mood", "epic"))
_add("tension tensioned danger threat chase hunt pursuit stalk", ("mood", "tense"), ("emotion", "suspense"))
_add("hope hopeful light rebirth new", ("mood", "hopeful"), ("emotion", "hope"))
_add("sacred holy temple church prayer hymn chant cathedral", ("atmosphere", "sacred"), ("mood", "solemn"))
_add("desert sand dunes waste wasteland", ("atmosphere", "vast"), ("atmosphere", "desolate"))
_add("snow ice winter frost frozen mountain mountains", ("atmosphere", "cold"), ("atmosphere", "vast"))
_add("forest woods grove", ("atmosphere", "pastoral"), ("mood", "mysterious"))
_add("sea ocean coast storm", ("atmosphere", "vast"))
_add("city town market tavern inn", ("atmosphere", "warm"))
_add("dream dreams dreaming sleep", ("atmosphere", "dreamlike"))
_add("ambient amb atmosphere", ("style", "ambient"), ("atmosphere", "atmospheric"))
_add("choir choral chorus", ("instrumentation", "choir"))
_add("piano", ("instrumentation", "piano"))
_add("guitar", ("instrumentation", "guitar"))
_add("violin", ("instrumentation", "violin"))
_add("cello", ("instrumentation", "cello"))
_add("flute", ("instrumentation", "flute"))
_add("harp", ("instrumentation", "harp"))
_add("drums taiko percussion", ("instrumentation", "percussion"))
_add("theme main opening overture title", ("atmosphere", "cinematic"))
_add("stinger", ("mood", "dramatic"))

GENRES: Dict[str, List[Tuple[str, str]]] = {
    "soundtrack": [("style", "cinematic")], "score": [("style", "cinematic")], "ost": [("style", "cinematic")],
    "orchestral": [("style", "orchestral"), ("instrumentation", "orchestra")],
    "classical": [("style", "classical"), ("instrumentation", "orchestra")],
    "ambient": [("style", "ambient"), ("atmosphere", "atmospheric")], "electronic": [("style", "electronic")],
    "techno": [("style", "electronic"), ("instrumentation", "electronic beats")], "rock": [("style", "rock")],
    "metal": [("style", "metal"), ("mood", "aggressive")], "folk": [("style", "folk")],
    "celtic": [("style", "celtic")], "medieval": [("style", "medieval")], "choral": [("style", "choral")],
    "jazz": [("style", "jazz")], "pop": [("style", "pop")], "world": [("style", "world")],
    "industrial": [("style", "industrial")], "acoustic": [("style", "acoustic")],
    "new age": [("style", "ambient"), ("atmosphere", "ethereal")], "game": [("style", "cinematic")],
    "anime": [("style", "cinematic")],
}


def _words(*texts: Any) -> List[str]:
    out: List[str] = []
    for text in texts:
        if not text:
            continue
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
        out += [w for w in re.split(r"[^a-z]+", spaced.lower()) if w]
    return out


def _apply_words(profile: SemanticProfile, words: List[str], evidence: List[str]) -> None:
    for word in words:
        for field_name, tag in LEXICON.get(word, ()):
            target = getattr(profile, field_name)
            if tag not in target:
                target.append(tag)
                evidence.append(f"'{word}' suggests {tag}")


def _genre(profile: SemanticProfile, genre: str, evidence: List[str]) -> None:
    lower = (genre or "").lower()
    for key, pairs in GENRES.items():
        if key in lower:
            for field_name, tag in pairs:
                if tag not in getattr(profile, field_name):
                    getattr(profile, field_name).append(tag)
                    evidence.append(f"genre '{genre}' suggests {tag}")


def _apply_measurements(p: SemanticProfile, m: Dict[str, Any], evidence: List[str]) -> None:
    energy = m.get("energy_index")
    brightness = m.get("brightness_index")
    band = m.get("band_energy") or {}
    if energy is not None:
        p.energy = energy
        evidence.append(f"measured energy index {energy:.2f}")
    if brightness is not None:
        low = band.get("low", 0.3)
        p.darkness = round(max(0.0, min(1.0, 0.65 * (1 - brightness) + 0.35 * min(1.0, low * 2))), 3)
        evidence.append(f"brightness {brightness:.2f} and low-frequency share {low:.0%} give darkness {p.darkness:.2f}")
    spread = m.get("level_spread_db")
    onsets = m.get("onsets_per_second")
    if energy is not None:
        tension = 0.5 * energy + 0.25 * min(1.0, (onsets or 0) / 6) + 0.25 * (1 - min(1.0, (spread or 10) / 20))
        p.tension = round(max(0.0, min(1.0, tension)), 3)
    if p.energy is not None and p.darkness is not None:
        p.valence = round(max(0.0, min(1.0, 0.5 + 0.35 * (p.energy - 0.5) - 0.5 * (p.darkness - 0.5))), 3)
        if p.energy < 0.35 and p.darkness < 0.5 and "peaceful" not in p.mood:
            p.mood.append("peaceful")
            evidence.append("low energy and not dark: peaceful")
        if p.energy > 0.7 and "epic" not in p.mood and p.darkness < 0.6:
            p.mood.append("epic")
            evidence.append("high energy: epic")
        if p.darkness > 0.7 and "dark" not in p.mood:
            p.mood.append("dark")
            evidence.append("dark timbre")
    tempo = m.get("tempo_bpm_estimate")
    if tempo is None and m.get("attack_ratio", 1) is not None and (m.get("attack_ratio") or 0) < 0.005:
        if "atmospheric" not in p.atmosphere:
            p.atmosphere.append("atmospheric")
            evidence.append("no distinct attacks: sustained, atmospheric texture")
    flat = m.get("spectral_flatness")
    if flat is not None and flat > 0.3 and "electronic" not in p.style:
        evidence.append("noise-like spectrum")


HEARD_FIELDS = ("mood", "emotion", "atmosphere", "instrumentation", "style")


def _apply_heard(p: SemanticProfile, heard: Dict[str, Any], evidence: List[str]) -> None:
    """Tags the listening model heard in the audio itself (``listening.summary`` format)."""

    if not heard:
        return
    for field_name in HEARD_FIELDS:
        for tag, strength in (heard.get(field_name) or {}).items():
            target = getattr(p, field_name)
            if tag not in target and (strength == "strong" or len(target) < 3):
                target.append(tag)
                evidence.append(f"heard {tag} ({strength})")
    vocals = heard.get("vocals")
    if vocals == "instrumental":
        p.vocal_presence = False
        evidence.append("heard: instrumental")
    elif vocals == "sung vocals":
        p.vocal_presence = True
        evidence.append("heard: sung vocals")
    p.confidence = max(p.confidence or 0, 0.4)


def describe_track(doc: Dict[str, Any]) -> Tuple[SemanticProfile, List[str]]:
    p = SemanticProfile(source="rules", confidence=0.25)
    evidence: List[str] = []
    _genre(p, doc.get("genre", ""), evidence)
    _apply_words(p, _words(doc.get("title"), doc.get("album"), doc.get("file_name"), doc.get("comment")), evidence)
    _apply_measurements(p, doc.get("measurements") or {}, evidence)
    _apply_heard(p, doc.get("heard") or {}, evidence)
    p.summary = "Rule-based estimate from tags and measurements: " + (p.describe() if p.describe() != "no description"
                                                                         else "little evidence available") + "."
    return normalize(p), evidence


def describe_cue(doc: Dict[str, Any]) -> Tuple[SemanticProfile, List[str]]:
    p = SemanticProfile(source="rules", confidence=0.2)
    evidence: List[str] = []
    notes = doc.get("community_notes", [])
    texts = (doc.get("structure_names", []) + doc.get("media_names", []) + doc.get("plays_when", [])
             + doc.get("started_by_events", []) + [n.get("description", "") for n in notes]
             + [n.get("context", "") for n in notes] + [n.get("original_name", "") for n in notes])
    _apply_words(p, _words(*texts), evidence)
    if notes:
        p.confidence = 0.35
        evidence.append("public community notes describe this music")
    tempo = doc.get("tempo_bpm")
    layers = doc.get("track_layers", 1)
    if tempo:
        p.energy = round(max(0.1, min(0.9, (tempo - 60) / 120)), 3)
        evidence.append(f"segment tempo {tempo:g} BPM (Wwise grid tempo, may be nominal)")
    if doc.get("is_transition"):
        p.mood.append("dramatic")
        evidence.append("short transition segment")
    if layers and layers > 1:
        evidence.append(f"{layers} simultaneous layers (stems)")
    duration = doc.get("duration_s") or 0
    if duration and duration > 150 and not p.mood:
        p.atmosphere.append("atmospheric")
        evidence.append("long, looping-length segment")
    if doc.get("measurements"):
        _apply_measurements(p, doc["measurements"], evidence)
        p.confidence = max(p.confidence or 0, 0.3)
        evidence.append("measured from the decoded game audio")
    _apply_heard(p, doc.get("heard") or {}, evidence)
    for tag in p.mood:
        if tag in ("tense", "aggressive"):
            p.tension = max(p.tension or 0, 0.75)
        if tag in ("dark", "ominous"):
            p.darkness = max(p.darkness or 0, 0.75)
        if tag in ("peaceful", "serene"):
            p.tension = min(p.tension if p.tension is not None else 1, 0.2)
            p.energy = min(p.energy if p.energy is not None else 1, 0.35)
    p.summary = ("Rule-based estimate from Wwise names, structure"
                 + (" and the decoded audio" if doc.get("measurements") else "") + ": "
                 + (p.describe() if p.describe() != "no description" else "no descriptive names available") + ".")
    return normalize(p), evidence
