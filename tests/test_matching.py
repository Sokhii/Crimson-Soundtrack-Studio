import json

import pytest

from soundtrack_studio.ai.runtime import ScriptedBackend
from soundtrack_studio.matching import engine
from soundtrack_studio.matching.engine import MatchSettings, Matcher, TrackInfo, duration_fit, tempo_fit
from soundtrack_studio.semantic.profile import SemanticProfile
from soundtrack_studio.semantic.store import EffectiveProfile
from soundtrack_studio.testing.fixtures import write_test_flac

DARK = {"mood": ["dark", "ominous"], "atmosphere": ["vast"], "style": ["orchestral"], "darkness": 0.9, "energy": 0.4}
CALM = {"mood": ["peaceful", "serene"], "atmosphere": ["pastoral"], "style": ["ambient"], "darkness": 0.1, "energy": 0.2}
EPIC = {"mood": ["epic", "heroic"], "atmosphere": ["cinematic"], "style": ["orchestral"], "darkness": 0.4, "energy": 0.9}


def prof(values, confidence=0.8, source="llm"):
    p = SemanticProfile.from_dict({**values, "confidence": confidence})
    p.source = source
    return EffectiveProfile(p)


# ------------------------------------------------------------------ unit
def test_duration_fit():
    assert duration_fit(180, 180)[0] == 1.0
    assert 0.85 < duration_fit(400, 180)[0] < 1.0 and "trimmed" in duration_fit(400, 180)[1]
    assert duration_fit(60, 180)[0] < 0.5 and "loops" in duration_fit(60, 180)[1]
    assert duration_fit(20, 180)[0] == 0.0


def test_tempo_fit_handles_half_and_double():
    assert tempo_fit(120, 120) == 1.0 and tempo_fit(60, 120) == 1.0 and tempo_fit(120, 240) == 1.0
    assert tempo_fit(100, 130) < 0.5 and tempo_fit(None, 120) is None


@pytest.fixture
def model(paths, analyzer_db):
    from soundtrack_studio.analyzer_db.importer import import_database
    from soundtrack_studio.game_model.builder import load_or_build

    imported = import_database(analyzer_db, paths)
    return load_or_build(paths, imported.snapshot_path, imported.sha256, 1)


def tracks():
    return [TrackInfo(1, "Dark Hymn", 200, 2, None, prof(DARK).profile),
            TrackInfo(2, "Morning Meadow", 160, 2, None, prof(CALM).profile),
            TrackInfo(3, "Glory March", 190, 2, 120, prof(EPIC).profile)]


def cue_profiles():
    return {"2001": prof(DARK), "2002": prof(CALM), "2003": prof(EPIC), "2004": prof(DARK)}


def test_thematic_assignment_and_explanations(model):
    results = Matcher(model, cue_profiles(), tracks(), MatchSettings()).run()
    assert set(results) == {"2001", "2002", "2003"}  # the 4 s transition segment is left alone by default
    assert [results[k].candidates[0].track_id for k in ("2001", "2002", "2003")] == [1, 2, 3]
    best = results["2001"].candidates[0]
    assert any(r.startswith("Shared mood: dark, ominous") for r in best.reasons)
    assert any(r.startswith("Length:") for r in best.reasons)
    assert best.confidence > results["2001"].candidates[1].confidence
    assert engine.confidence_label(best.confidence) in ("high", "medium")
    epic = results["2003"].candidates[0]
    assert any("Compatible tempo" in r for r in epic.reasons)


def test_short_cues_included_on_request(model):
    results = Matcher(model, cue_profiles(), tracks(), MatchSettings(include_short_cues=True)).run()
    assert "2004" in results and results["2004"].candidates


def test_no_reuse_leaves_cues_unmatched(model):
    only_dark = [tracks()[0]]
    results = Matcher(model, cue_profiles(), only_dark, MatchSettings(allow_reuse=False)).run()
    matched = [k for k, r in results.items() if r.candidates]
    assert len(matched) == 1 and results["2001"].candidates[0].track_id == 1
    assert any("already used" in r.skipped_reason for r in results.values())
    reuse = Matcher(model, cue_profiles(), only_dark, MatchSettings()).run()
    assert all(r.candidates for r in reuse.values())
    assert any("also proposed" in w for r in reuse.values() for w in r.candidates[0].warnings)


def test_rejected_and_fixed_are_respected(model):
    results = Matcher(model, cue_profiles(), tracks(), MatchSettings(), rejected={"2001": {1}},
                      fixed={"2002": 1}).run()
    assert all(c.track_id != 1 for c in results["2001"].candidates)
    assert results["2002"].candidates  # still shows candidates for the user's reference


def test_too_short_tracks_are_filtered(model):
    tiny = [TrackInfo(9, "Jingle", 10, 2, None, prof(DARK).profile)]
    results = Matcher(model, cue_profiles(), tiny, MatchSettings()).run()
    assert not results["2001"].candidates and "long enough" in results["2001"].skipped_reason


def test_uncertainty_warnings(model):
    weak_cues = {k: prof(DARK, confidence=0.2, source="rules") for k in ("2001", "2002", "2003")}
    rule_tracks = [TrackInfo(1, "a", 100, 1, None, prof(DARK, source="rules").profile)]
    cand = Matcher(model, weak_cues, rule_tracks, MatchSettings()).run()["2001"].candidates[0]
    text = " ".join(cand.warnings)
    assert "guessed from internal names" in text and "rules only" in text and "Mono" in text and "loop" in text
    strong = Matcher(model, cue_profiles(), tracks(), MatchSettings()).run()["2001"].candidates[0]
    assert strong.confidence > cand.confidence


def test_ai_judging_reorders_shortlist(model):
    ranking = json.dumps({"ranking": [{"candidate": 2, "fit": 95, "reason": "Its melancholy suits the desert."},
                                      {"candidate": 1, "fit": 10, "reason": "Too heavy."}]})
    backend = ScriptedBackend([ranking])
    results = Matcher(model, cue_profiles(), tracks(), MatchSettings(use_ai=True), backend=backend).run()
    assert backend.calls
    top = results["2001"].candidates[0]
    assert top.ai_reason and top.ai_fit is not None


def test_ai_failures_fall_back_to_deterministic_ranking(model):
    matcher = Matcher(model, cue_profiles(), tracks(), MatchSettings(use_ai=True), backend=ScriptedBackend(["garbage"]))
    results = matcher.run()
    assert matcher.ai_errors and results["2001"].candidates[0].track_id == 1


# ------------------------------------------------------- store + services
@pytest.fixture
def matched(studio, tmp_path, game, analyzer_db):
    studio.set_game_path(game[0])
    studio.import_analyzer(analyzer_db)
    music = tmp_path / "Music"
    for name, title, bpm, tone in (("a.flac", "Dark Requiem", None, 110), ("b.flac", "Village Dawn", None, 440),
                                   ("c.flac", "Hero Victory", 120, 220)):
        write_test_flac(music / name, seconds=45, bpm=bpm, tone_hz=tone, tags={"TITLE": title})
    studio.set_library_path(music)
    studio.scan_library()
    stats = studio.find_matches()
    return studio, stats


def test_find_matches_describes_and_proposes(matched):
    studio, stats = matched
    assert stats["cues"] == 3 and stats["proposed"] == 3 and stats["tracks"] == 3
    rows = studio.matching_rows()
    assert len(rows) == 3 and all(r["status"] == "proposed" for r in rows)
    assert len(studio.matching_rows(include_short=True)) == 4
    status = {s.number: s.state for s in studio.project_status().steps}
    assert status[6] == "ok" and status[7] == "warning"


def test_decisions_win_over_reruns(matched):
    studio, _ = matched
    store = studio.match_store()
    proposals = store.proposals()
    first = proposals["2001"][0].track_id
    store.accept("2001", first)
    other = next(t for t in (1, 2, 3) if t != proposals["2002"][0].track_id)
    store.choose("2002", other)
    store.reject("2003", proposals["2003"][0].track_id)
    studio.find_matches()
    decisions = store.decisions()
    assert decisions["2001"].action == "accept" and decisions["2001"].track_id == first
    assert decisions["2002"].track_id == other
    new_2003 = store.proposals()["2003"]
    assert all(p.track_id != proposals["2003"][0].track_id for p in new_2003)
    mapping = store.final_mapping()
    assert [(m.cue_key, m.decided_by) for m in mapping] == [("2001", "accept"), ("2002", "manual")]
    store.keep_original("2001")
    assert [m.cue_key for m in store.final_mapping()] == ["2002"]
    store.clear("2001")
    assert "2001" not in store.decisions()


def test_fit_settings_and_bulk_accept(matched):
    studio, _ = matched
    store = studio.match_store()
    with pytest.raises(ValueError):
        store.set_fit("2001", "loop")
    count = store.accept_all(min_confidence=0.0)
    assert count == 3 and store.status_counts()["accepted"] == 3
    store.set_fit("2001", "loop", 12.5)
    entry = next(m for m in store.final_mapping() if m.cue_key == "2001")
    assert entry.fit_mode == "loop" and entry.start_offset_s == 12.5
    with pytest.raises(ValueError):
        store.set_fit("2001", "stretch")


def test_rank_tracks_for_cue(matched):
    studio, _ = matched
    rows = studio.rank_tracks_for_cue("2001")
    assert len(rows) == 3 and rows[0]["candidate"].score >= rows[-1]["candidate"].score
    assert studio.rank_tracks_for_cue("999") == []
