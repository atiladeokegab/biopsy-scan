import json
from pathlib import Path

import pytest

from bench.power_at_scale import answer, crossing

RESULT = Path(__file__).parent / "results" / "power-at-scale.json"


def pt(n, p, lo, hi):
    return {"patients": n, "power": p, "power_lo": lo, "power_hi": hi}


def test_crossing_interpolates_the_first_bracket():
    assert crossing([300, 400, 600], [0.4, 0.6, 0.9]) == pytest.approx(400 + 0.2 / 0.3 * 200)
    assert crossing([300, 400], [0.4, 0.6]) is None
    assert crossing([100, 200, 300], [0.85, 0.7, 0.9]) == 100.0  # first crossing, not the last


def test_answer_says_plainly_when_80_is_never_reached():
    a = answer([pt(300, 0.4, 0.35, 0.47), pt(1000, 0.7, 0.64, 0.75)])
    assert a["n_80"] is None and a["statement"].startswith("Simulated power never reached 80% up to 1000 patients")
    b = answer([pt(300, 0.4, 0.35, 0.47), pt(600, 0.82, 0.77, 0.86)])
    assert b["n_80"] is not None and b["n_80_lo"] <= b["n_80"] and b["n_80_hi_beyond_range"]


@pytest.mark.skipif(not RESULT.exists(), reason="power-at-scale.json not built yet")
def test_committed_result_is_a_labelled_estimate_with_its_own_answer():
    res = json.loads(RESULT.read_text())
    assert res["schema"] == "preflight-power-at-scale/1"
    assert res["status"] == "ESTIMATE" and res["label"] == "ESTIMATE: simulated"
    base = json.loads((RESULT.parent / "vs-schoenfeld-300.json").read_text())
    assert res["simulated"]["seed"] == base["simulated"]["seed"] and res["simulated"]["budget"] == base["simulated"]["budget"]
    assert res["design"] == {k: v for k, v in base["design"].items() if k != "backend"}
    ns = [p["patients"] for p in res["points"]]
    assert ns == sorted(set(ns)) and max(ns) > 300
    for p in res["points"]:
        assert 0 <= p["power_lo"] <= p["power"] <= p["power_hi"] <= 1
        assert p["power_n"] == base["simulated"]["budget"]["alternative"]
        if p["patients"] > 300:
            assert p["backend"] == "gpu" and p["hardware"] and p["wall_s"] > 0 and p["source"].startswith("bench/results/modal-power-")
    assert res["answer"] == answer(res["points"])  # the headline is recomputed from the points, never typed in
