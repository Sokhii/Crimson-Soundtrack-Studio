"""The listening model's vocabulary: the words it scores every piece of music against.

Larger and more specific than the profile vocabulary the text model writes from
(``semantic/profile.py``, which stays small so descriptions are stable and the editor usable):
about 340 words in seven groups, weighted towards what CLAP recognises reliably (instruments, genres,
rhythm, recording texture) and fairly coarse for moods, where near-synonyms are not separable.
Every profile word is included, so heard tags map onto profile tags where they exist; all other
heard words end up in the profile's free-form themes.

A word that scores about the same for all the music analysed cannot tell pieces apart; such words
are skipped automatically (see ``calibration.Calibration``), so extra words cost nothing.
"""

from __future__ import annotations

from typing import Dict, Tuple

from ..semantic.profile import CATEGORIES as PROFILE_CATEGORIES

_MOOD = (
    "brooding", "haunting", "eerie", "uneasy", "foreboding", "bittersweet", "wistful", "yearning", "uplifting",
    "joyful", "cheerful", "festive", "defiant", "resolute", "noble", "majestic", "grand", "intense", "frantic",
    "relentless", "furious", "menacing", "sinister", "suspenseful", "anxious", "bleak", "mournful", "tragic",
    "tender", "gentle", "soothing", "dreamy", "magical", "enchanting", "mischievous", "quirky", "comical",
    "reverent", "contemplative", "meditative", "introspective", "lonely", "relaxed", "pensive", "warm-hearted")
_EMOTION = (
    "anxiety", "despair", "courage", "resolve", "pride", "excitement", "thrill", "playfulness", "love", "affection",
    "serenity", "reverence", "mourning", "regret", "bliss", "rage", "terror", "unease", "curiosity", "yearning",
    "nostalgia", "melancholy", "devotion", "defiance")
_ATMOSPHERE = (
    "expansive", "claustrophobic", "haunted", "misty", "wintry", "arid", "stormy", "nocturnal", "seaside", "woodland",
    "battlefield", "tavern", "royal court", "cathedral", "campfire", "mountain", "ruins", "underground", "cosmic",
    "mechanical", "rainy", "sunrise", "spooky", "majestic landscape", "small village")
_INSTRUMENT = (
    "acoustic guitar", "electric guitar", "distorted guitar", "classical guitar", "banjo", "mandolin", "lute",
    "harpsichord", "accordion", "fiddle", "hurdy-gurdy", "bagpipes", "tin whistle", "recorder", "pan flute", "ocarina",
    "trumpet", "french horn", "trombone", "tuba", "clarinet", "oboe", "bassoon", "saxophone", "double bass",
    "electric bass", "timpani", "taiko drums", "hand drums", "snare drum", "cymbals", "glockenspiel", "celesta",
    "music box", "marimba", "xylophone", "vibraphone", "tubular bells", "church organ", "synth pad", "synth lead",
    "arpeggiated synthesizer", "sub bass", "drum machine", "string quartet", "string section", "solo violin",
    "solo cello", "solo piano", "solo flute", "solo guitar", "brass section", "male choir", "female choir",
    "children's choir", "male vocals", "female vocals", "whispered vocals", "chanting", "throat singing", "sitar",
    "erhu", "koto", "shamisen", "shakuhachi", "didgeridoo", "dulcimer", "kalimba", "steel drums", "handpan",
    "harmonica", "pipe organ", "electric piano", "strummed guitar", "fingerpicked guitar", "war drums", "choral pads",
    "orchestral percussion", "low strings", "high strings", "woodwind solo", "horns", "church bells")
_STYLE = (
    "dark ambient", "drone", "cinematic orchestral", "epic orchestral", "trailer music", "neoclassical",
    "romantic classical", "baroque", "film score", "minimalism", "post-rock", "progressive rock", "hard rock",
    "heavy metal", "symphonic metal", "black metal", "doom metal", "folk rock", "nordic folk", "celtic folk",
    "medieval folk", "flamenco", "blues", "country", "bluegrass", "gospel", "hip hop", "trap", "EDM", "techno",
    "house", "trance", "dubstep", "drum and bass", "synthwave", "chiptune", "lo-fi", "new age", "J-pop", "K-pop",
    "anime", "video game music", "jazz fusion", "swing", "bossa nova", "reggae", "latin", "middle eastern",
    "east asian traditional", "indie", "singer-songwriter", "ballad", "rock opera", "power metal", "punk",
    "soundtrack", "big band")
_RHYTHM = (
    "slow tempo", "mid tempo", "fast tempo", "driving beat", "pulsing", "steady pulse", "syncopated", "swinging",
    "waltz", "march", "free tempo", "rubato", "sparse", "dense", "repetitive", "building", "crescendo", "explosive",
    "flowing", "stuttering")
_TEXTURE = (
    "lo-fi", "polished studio production", "raw live recording", "vintage recording", "old phonograph recording",
    "cinematic production", "reverberant", "spacious", "dry", "intimate close-mic", "warm analog", "digital",
    "glitchy", "distorted", "lush", "thin", "layered", "minimal arrangement", "solo performance", "full band",
    "a cappella", "chamber ensemble", "big orchestra", "wall of sound")


def _merge(base: Tuple[str, ...], extra: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(tuple(base) + tuple(extra)))


# category -> words; the first five keys are the profile's categories (same names), plus rhythm and texture
VOCABULARY: Dict[str, Tuple[str, ...]] = {
    "mood": _merge(PROFILE_CATEGORIES["mood"], _MOOD),
    "emotion": _merge(PROFILE_CATEGORIES["emotion"], _EMOTION),
    "atmosphere": _merge(PROFILE_CATEGORIES["atmosphere"], _ATMOSPHERE),
    "instrumentation": _merge(PROFILE_CATEGORIES["instrumentation"], _INSTRUMENT),
    "style": _merge(PROFILE_CATEGORIES["style"], _STYLE),
    "rhythm": _RHYTHM,
    "texture": _TEXTURE,
}

# how a word becomes text prompts for the model (each template is averaged)
TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "mood": ("{} music", "a {} piece of music"),
    "emotion": ("music that expresses {}", "music full of {}"),
    "atmosphere": ("{} music", "music with a {} atmosphere"),
    "instrumentation": ("music featuring {}", "the sound of {}"),
    "style": ("{} music", "a piece of {} music"),
    "rhythm": ("{} music", "music with a {} rhythm"),
    "texture": ("{} music", "a {} recording"),
}

# a word listed in several groups (e.g. "lo-fi") is scored once per group; the tag keeps its group
WORD_COUNT = sum(len(v) for v in VOCABULARY.values())


def prompts_for(category: str, word: str) -> Tuple[str, ...]:
    return tuple(t.format(word) for t in TEMPLATES[category])
