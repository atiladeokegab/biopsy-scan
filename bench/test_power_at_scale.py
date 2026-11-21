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
    assert a["smallest_patients_pointwise_lower_bound_clears_80"] is None and "beyond 1000" in a["statement"]
    b = answer([pt(300, 0.4, 0.35, 0.47), pt(600, 0.82, 0.77, 0.86)])
    assert b["n_80"] is not None and b["n_80_pointwise_ci_crossings"][0] <= b["n_80"]
    assert b["n_80_pointwise_ci_crossings"][1] is None
    c = answer([pt(300, 0.4, 0.35, 0.47), pt(600, 0.82, 0.77, 0.86), pt(800, 0.9, 0.86, 0.93), pt(1000, 0.95, 0.92, 0.97)])
    assert c["smallest_patients_pointwise_lower_bound_clears_80"] == 800


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


def _run(outer, inner):
    base = json.loads((RESULT.parent / "vs-schoenfeld-300.json").read_text())["simulated"]
    s = {"patients": inner, "regions_per_patient": 2, "fov_um": 400.0, "eps": 0.1, "hr_per_source_sd": 2.0,
         "power": 0.9, "power_lo": 0.86, "power_hi": 0.93, "power_n": 300, "fwer": 0.05, "fwer_lo": 0.02,
         "fwer_hi": 0.1, "alternative_mean_events": 100.0, "resample_ratio": inner / 159, "backend": "gpu",
         "runtime_s": 1.0}
    m = {k: base[k] for k in ("seed", "source_hash", "grammar_hash", "budget")}
    return {"_file": "modal-power-x.json", "points": [{"patients": outer, "summary": s, "metadata": m,
                                                       "gpu": {"model": "A100"}, "in_function_s": 1.0}]}


def test_merger_refuses_a_summary_for_another_patient_count():
    from bench.power_at_scale import build

    assert [p["patients"] for p in build([_run(400, 400)])["points"]][-1] == 400
    with pytest.raises(ValueError, match="summary is for P=40"):
        build([_run(400, 40)])


def test_ci_crossings_are_called_pointwise_never_a_band():
    a = answer([pt(300, 0.4, 0.35, 0.47), pt(600, 0.82, 0.77, 0.86), pt(800, 0.9, 0.86, 0.93)])
    assert "band" not in a["statement"] and "pointwise" in a["statement"]
    lo, hi = a["n_80_pointwise_ci_crossings"]
    assert "n_80_lo" not in a and lo <= a["n_80"] <= hi
    assert a["smallest_patients_pointwise_lower_bound_clears_80"] == 800
