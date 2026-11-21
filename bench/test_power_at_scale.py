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
    assert res["complete"] == (not res["answer"]["missing_patients"])
    assert res["answer"] == answer(res["points"], res["answer"]["missing_patients"])  # recomputed, never typed in


REAL = RESULT.parent / "modal-power-a100-40gb-20261121T175238Z.json"  # a real producer run: P=40 on the A100


def real(patients=40, **summary):
    run = json.loads(REAL.read_text())
    run["_file"] = REAL.name
    point = run["points"][0]
    point["patients"] = point["summary"]["patients"] = patients
    point["summary"]["resample_ratio"] = patients / 159
    point["summary"].update(summary)
    return run


def test_a_real_replicate_of_53_merges_and_is_recorded():
    from bench.power_at_scale import build

    p40 = next(p for p in build([real()])["points"] if p["patients"] == 40)
    assert p40["source"] == "bench/results/vs-schoenfeld-300.json" and p40["replicated_by"] == [f"bench/results/{REAL.name}"]
    p400 = next(p for p in build([real(400)])["points"] if p["patients"] == 400)
    assert p400["hardware"] == "NVIDIA A100-SXM4-40GB" and p400["backend"] == "gpu"


@pytest.mark.parametrize("change, match", [
    ({"alpha": 0.5}, "different design"), ({"cohort": "upmc"}, "different design"),
    ({"backend": "cpu"}, "backend"), ({"calibration_n": 1}, "study counts"), ({"power_n": 1}, "study counts"),
    ({"seed": 1}, "seed"), ({"fwer": 2.0}, "rates"), ({"power_lo": 0.5}, "rates"), ({"power": 0.5}, "rates"),
    ({"power_hi": 0.2}, "rates"), ({"runtime_s": 0.0}, "timing"), ({"runtime_s": float("nan")}, "timing"),
    ({"resample_ratio": 1.0}, "resampl"), ({"source_n": 160}, "resampl"), ({"source_hash": "0" * 64}, "source"),
])
def test_merger_validates_the_summary_itself(change, match):
    from bench.power_at_scale import build

    with pytest.raises(ValueError, match=match):
        build([real(400, **change)])


def test_merger_refuses_wrong_pin_status_and_mismatched_p():
    from bench.power_at_scale import build

    for key, value in (("pin", "0" * 40), ("status", "ESTIMATE"), ("backend", "cpu")):
        bad = real(400)
        bad[key] = value
        with pytest.raises(ValueError):
            build([bad])
    bad = real(400)
    bad["points"][0]["summary"]["patients"] = 40
    with pytest.raises(ValueError, match="summary is for P=40"):
        build([bad])


def test_conflicting_duplicates_are_refused_in_either_order():
    from bench.power_at_scale import build

    from bench.power_at_scale import clopper_pearson

    lo, hi = clopper_pearson(30, 300)
    other = real(400, power=0.1, power_count=30, power_lo=lo, power_hi=hi)
    for order in ([real(400), other], [other, real(400)]):
        with pytest.raises(ValueError, match="P=400 twice with different results"):
            build(order)
    with pytest.raises(ValueError, match="P=40 twice with different results"):
        build([real(40, power=0.1, power_count=30, power_lo=lo, power_hi=hi)])


def test_failed_points_are_kept_and_mark_the_curve_partial():
    from bench.power_at_scale import build

    failed = real(500)
    failed["points"] = [{"patients": 500, "error": "boom"}]
    res = build([real(400), failed])
    assert res["complete"] is False and res["failed_points"] == [{"patients": 500, "error": "boom", "source": f"bench/results/{REAL.name}"}]
    assert res["answer"]["statement"].startswith("PARTIAL") and "never reached" not in res["answer"]["statement"]
    assert build([real(400)])["complete"] is True
