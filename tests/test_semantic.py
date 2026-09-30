import json

import pytest

from soundtrack_studio.ai.runtime import BackendError, ModelOutputError, ScriptedBackend
from soundtrack_studio.semantic import rules
from soundtrack_studio.semantic.context import input_hash
from soundtrack_studio.semantic.profile import (ProfileParseError, SemanticProfile, json_schema, merge, normalize,
                                                parse_llm_output, similarity)
from soundtrack_studio.semantic.store import ResponseCache, SemanticStore
from soundtrack_studio.project.store import Project

GOOD = json.dumps({"mood": ["dark", "ominous"], "emotion": ["dread"], "atmosphere": ["vast"], "instrumentation": ["choir"],
                   "style": ["orchestral"], "themes": ["ruin"], "energy": 40, "darkness": 90, "tension": 70, "valence": 10,
                   "vocal_presence": False, "confidence": 80, "summary": "A dark choral piece."})


def test_normalize_maps_synonyms_and_moves_tags():
    p = normalize(SemanticProfile(mood=["Sad", "strings", "brooding", "sad", "banana"], style=["dark", "symphonic"],
                                  energy=85, darkness=0.4, tension="x", vocal_presence="yes", confidence=150))
    assert p.mood == ["melancholic", "dark"]
    assert "strings" in p.instrumentation and "orchestral" in p.style
    assert "banana" in p.themes
    assert p.energy == 0.85 and p.darkness == 0.4 and p.tension is None
    assert p.vocal_presence is True and p.confidence == 1.0


def test_parse_llm_output_variants():
    assert parse_llm_output(GOOD).darkness == 0.9
    assert parse_llm_output(f"<think>hmm</think>```json\n{GOOD}\n```").mood == ["dark", "ominous"]
    assert parse_llm_output("Sure! " + GOOD + " Hope this helps").style == ["orchestral"]
    assert parse_llm_output(json.dumps({"mood": "dark, tense", "energy": 50})).mood == ["dark", "tense"]
    for bad in ("", "no json here", "[1,2,3]", '{"mood": 5}', '{"themes": ["x"]}', "{broken"):
        with pytest.raises(ProfileParseError):
            parse_llm_output(bad)


def test_schema_uses_vocabularies():
    schema = json_schema()
    assert "dark" in schema["properties"]["mood"]["items"]["enum"]
    assert schema["properties"]["energy"]["maximum"] == 100


def test_similarity():
    dark = SemanticProfile(mood=["dark", "ominous"], style=["orchestral"], darkness=0.9, energy=0.4)
    darkish = SemanticProfile(mood=["mysterious", "dark"], style=["cinematic"], darkness=0.8, energy=0.5)
    bright = SemanticProfile(mood=["playful", "hopeful"], style=["pop"], darkness=0.1, energy=0.8)
    s1, parts = similarity(dark, darkish)
    s2, _ = similarity(dark, bright)
    assert s1 > 0.6 > s2 and "mood" in parts and "darkness" in parts
    assert similarity(SemanticProfile(), SemanticProfile())[0] == 0.0


def test_merge_override_wins():
    base = SemanticProfile(mood=["dark"], energy=0.3, source="llm", model_id="m")
    merged = merge(base, {"mood": ["peaceful"], "energy": 0.9})
    assert merged.mood == ["peaceful"] and merged.energy == 0.9 and merged.source == "user" and merged.confidence == 1.0


def test_rules_track_and_cue():
    track = {"title": "Battle in the Shadows", "genre": "Soundtrack",
             "measurements": {"energy_index": 0.8, "brightness_index": 0.2, "band_energy": {"low": 0.4}}}
    p, evidence = rules.describe_track(track)
    assert {"tense", "dark"} <= set(p.mood) and "cinematic" in p.style and p.source == "rules"
    assert p.energy == 0.8 and p.darkness > 0.6 and evidence
    cue = {"plays_when": ["BGM_Region=Desert"], "structure_names": ["BGM_Tension_Combat"], "tempo_bpm": 150,
           "community_notes": [{"description": "Ancient ruins theme"}]}
    p, evidence = rules.describe_cue(cue)
    assert "vast" in p.atmosphere and "tense" in p.mood and "mysterious" in p.mood and p.confidence == 0.35


def test_input_hash_stable_and_sensitive():
    assert input_hash("track", {"a": 1, "b": 2}) == input_hash("track", {"b": 2, "a": 1})
    assert input_hash("track", {"a": 1}) != input_hash("cue", {"a": 1})


@pytest.fixture
def store(paths):
    project = Project.create(paths, "Sem")
    cache = ResponseCache(paths.cache / "ai_responses.sqlite3")
    yield SemanticStore(project, cache)
    cache.close()
    project.close()


ITEMS = [("1", {"title": "Dark Requiem", "measurements": {"energy_index": 0.3}}),
         ("2", {"title": "Village Dawn", "measurements": {"energy_index": 0.2}})]


def test_rules_only_run_and_effective_profile(store):
    stats = store.run("track", ITEMS)
    assert stats.rules_done == 2 and stats.llm_done == 0
    eff = store.effective_all("track")
    assert eff["1"].source == "rules" and "melancholic" in eff["1"].profile.mood
    assert store.run("track", ITEMS).rules_done == 0  # unchanged evidence: nothing recomputed


def test_llm_run_is_resumable_and_cached(store, paths):
    backend = ScriptedBackend([GOOD])
    stats = store.run("track", ITEMS, backend, "model-sha")
    assert stats.llm_done == 2 and len(backend.calls) == 2
    eff = store.effective("track", "1")
    assert eff.source == "llm" and eff.llm.model_id == "scripted-test" and eff.rules is not None
    again = store.run("track", ITEMS, backend, "model-sha")
    assert again.skipped == 2 and len(backend.calls) == 2  # nothing re-asked
    changed = [("1", {"title": "Dark Requiem (Remastered)"})] + ITEMS[1:]
    store.run("track", changed, backend, "model-sha")
    assert len(backend.calls) == 3  # only the changed item
    # another project with the same evidence and model reuses the shared cache
    other = SemanticStore(Project.create(paths, "Other"), store.cache)
    second = ScriptedBackend([GOOD])
    stats = other.run("track", ITEMS, second, "model-sha")
    assert stats.llm_from_cache == 2 and not second.calls
    other.project.close()


def test_interrupted_run_resumes(store):
    calls = {"n": 0}

    def flaky(_messages):
        calls["n"] += 1
        return GOOD

    backend = ScriptedBackend([flaky])
    cancel_after_first = iter([False, True])
    with pytest.raises(Exception):
        store.run("track", ITEMS, backend, "k", cancel=lambda: next(cancel_after_first))
    assert calls["n"] == 1
    stats = store.run("track", ITEMS, backend, "k")
    assert stats.skipped == 1 and stats.llm_done == 1


def test_bad_model_output_falls_back_to_rules(store):
    backend = ScriptedBackend(["this is not json"])
    stats = store.run("track", ITEMS, backend, "k")
    assert stats.llm_errors == 2 and stats.errors
    eff = store.effective("track", "1")
    assert eff.source == "rules" and "unusable" in eff.llm_error
    retry = ScriptedBackend([GOOD])
    assert store.run("track", ITEMS, retry, "k").llm_done == 2  # failed items are retried later


def test_incomplete_output_retries_then_continues(store):
    class Incomplete(ScriptedBackend):
        def chat(self, *a, **k):
            self.calls.append(a)
            raise ModelOutputError("incomplete", details="HTTP 500: does not match")

    backend = Incomplete([])
    stats = store.run("track", ITEMS * 4, backend, "k")  # never aborts on output problems
    assert stats.llm_errors == len(ITEMS * 4)


def test_backend_failures_abort_after_threshold(store):
    class Down(ScriptedBackend):
        def chat(self, *a, **k):
            raise BackendError("down", details="connection refused")

    items = [(str(i), {"title": f"t{i}"}) for i in range(10)]
    with pytest.raises(BackendError):
        store.run("track", items, Down([]), "k", max_llm_failures=3)
    assert len(store.effective_all("track")) == 3  # rules profiles of the processed items were kept


def test_override_precedence_and_reset(store):
    store.run("track", ITEMS, ScriptedBackend([GOOD]), "k")
    store.set_override("track", "1", {"mood": ["peaceful"]})
    eff = store.effective("track", "1")
    assert eff.source == "user" and eff.profile.mood == ["peaceful"] and eff.profile.darkness == 0.9
    store.run("track", [("1", {"title": "changed"})], ScriptedBackend([GOOD]), "k")
    assert store.effective("track", "1").profile.mood == ["peaceful"]  # AI reruns never overwrite user edits
    store.clear_override("track", "1")
    assert store.effective("track", "1").source == "llm"


def test_forget_missing(store):
    store.run("track", ITEMS)
    assert store.forget_missing("track", ["1"]) == 1
    assert set(store.effective_all("track")) == {"1"}
