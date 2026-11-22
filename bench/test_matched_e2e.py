import hashlib
from pathlib import Path

import numpy as np
import pytest

from bench import matched_e2e as m


def rec(stream, i, sha="a", **kw):
    return {"stream": stream, "i": i, "x_sha": sha, "reimage_s": 0.1, **kw}


def lab_out(n=3, workers=1, wall=30.0, shas=None):
    recs = [rec(s, i, (shas or {}).get((s, i), f"{s}{i}"), p=0.01 if s == "alternative" else 0.5, maxT=3.0,
                test_s=1.0) for s in ("null", "alternative") for i in range(n)]
    return {"arm": "lab", "workers": workers, "studies_wall_s": wall, "warm_s": None,
            "setup": {"prep_s": 1.0, "adjacency_s": 0.0}, "records": recs}


def pre_out(n=5, cal=20, wall=9.0):
    rng = np.random.default_rng(0)
    recs = [rec("calibration", i, z=rng.normal(size=4), score_s=0.01) for i in range(cal)]
    for s in ("null", "alternative"):
        for i in range(n):
            z = np.zeros(4)
            z[0] = 50.0 if s == "alternative" else 0.0
            recs.append(rec(s, i, f"{s}{i}", z=z, score_s=0.01))
    return {"arm": "numba", "workers": 8, "studies_wall_s": wall, "warm_s": 2.0,
            "setup": {"prep_s": 1.0, "adjacency_s": 3.0}, "records": recs}


def plan(lab=3, cal=20, n=5):
    return {"lab": {"null": lab, "alternative": lab}, "preflight": {"calibration": cal, "null": n, "alternative": n}}


def test_the_lab_sample_must_be_a_subset_of_preflights_studies():
    p = m.plans(lab_per_stream=50, calibration=1000, per_stream=1000)
    assert p["lab"] == {"null": 50, "alternative": 50}
    assert p["preflight"] == {"calibration": 1000, "null": 1000, "alternative": 1000}
    with pytest.raises(ValueError):
        m.plans(lab_per_stream=60, calibration=1000, per_stream=50)
    for bad in (0, 18, 1.5, True):
        with pytest.raises(ValueError):
            m.plans(lab_per_stream=5, calibration=bad, per_stream=10)


def test_a_nan_or_nonpositive_cap_is_refused_before_any_run():
    m.check_cap(0.4, 0.9, 8.0)
    for cap in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError):
            m.check_cap(0.4, 0.9, cap)
    with pytest.raises(ValueError, match="refused"):
        m.check_cap(7.5, 0.9, 8.0)


def test_duplicate_or_missing_study_ids_are_refused_not_measured():
    dup = pre_out()
    dup["records"].append(dict(dup["records"][-1]))
    with pytest.raises(ValueError, match="duplicate"):
        m.compare(lab_out(), dup, plan())
    lab = lab_out()
    lab["records"][1] = dict(lab["records"][0])  # null 0 twice, null 1 missing
    with pytest.raises(ValueError, match="duplicate"):
        m.compare(lab, pre_out(), plan())
    short = pre_out()
    short["records"] = [r for r in short["records"] if (r["stream"], r["i"]) != ("calibration", 3)]
    with pytest.raises(ValueError, match="cover its plan exactly"):
        m.compare(lab_out(), short, plan())
    with pytest.raises(ValueError, match="cover its plan exactly"):
        m.compare(lab_out(n=2), pre_out(), plan())


def test_matched_walls_count_preflights_calibration_and_checks_features_per_study():
    out = m.compare(lab_out(n=3, wall=30.0), pre_out(n=5, cal=20, wall=9.0), plan())
    w = out["wall"]
    assert w["lab_s_per_study"] == pytest.approx(30.0 / 6)
    assert w["preflight_s_per_evaluated_study"] == pytest.approx(9.0 / 30)
    assert w["preflight_s_per_decided_study"] == pytest.approx(9.0 / 10)  # the 20 calibration nulls are paid for
    assert w["ratio"] == pytest.approx((30.0 / 6) / (9.0 / 10))
    assert out["parity"] == {"x_bitwise": True, "x_compared": 6}
    assert len(out["per_study"]) == 6 and all(r["agree"] for r in out["per_study"])
    assert out["decisions"]["agree"] == 6 and out["decisions"]["compared"] == 6
    assert out["status"] == "MEASURED"


def test_a_feature_mismatch_on_any_study_fails_the_result():
    out = m.compare(lab_out(n=3, shas={("null", 1): "other"}), pre_out(), plan())
    assert out["parity"]["x_bitwise"] is False and out["status"] == "FAILED"


def test_the_lab_must_be_one_worker_on_studies_preflight_also_ran():
    with pytest.raises(ValueError, match="one worker"):
        m.compare(lab_out(workers=8), pre_out(), plan())
    with pytest.raises(ValueError, match="not in the preflight arm"):
        m.compare(lab_out(n=6), pre_out(n=5), plan(lab=6))


def test_an_equal_hardware_lab_must_declare_and_match_preflights_workers():
    out = m.compare(lab_out(workers=8), pre_out(), plan(), lab_workers=8)
    assert out["status"] == "MEASURED"
    with pytest.raises(ValueError, match="8 workers, as declared"):
        m.compare(lab_out(workers=1), pre_out(), plan(), lab_workers=8)
    with pytest.raises(ValueError, match="as many workers as Preflight"):
        m.compare(lab_out(workers=4), pre_out(), plan(), lab_workers=4)
    with pytest.raises(ValueError, match="lab_workers must be 1 or 8"):
        m.run(lambda: None, plan(), workers=8, lab_workers=4)


def test_the_sweep_projection_is_analytic_and_lab_skips_calibration():
    w = {"lab_s_per_study": 7.0, "preflight_s_per_evaluated_study": 0.1}
    grid = {"points": 2, "calibration": 10, "null": 10, "alternative": 1}
    p = m.project(w, grid)
    assert p["status"] == "ANALYTIC"
    assert p["lab_s"] == pytest.approx(2 * 11 * 7.0)
    assert p["preflight_s"] == pytest.approx(2 * 21 * 0.1)


def test_build9s_harness_is_an_unedited_copy():
    got = hashlib.sha256((Path(__file__).parent / "b9_design_batch.py").read_bytes()).hexdigest()
    assert got == m.B9["sha256"]


def test_the_committed_result_is_matched_and_its_numbers_follow_from_its_fields():
    import json
    found = sorted((Path(__file__).parent / "results").glob("matched-e2e-*.json"))
    if not found:
        pytest.skip("no committed matched result yet")
    r = json.loads(found[-1].read_text())
    w = r["wall"]
    assert r["labels"] == m.LABELS and "compute through scoring" in r["headline"]
    ids = [(x["stream"], x["i"]) for x in r["per_study"]]
    assert sorted(ids) == sorted((s, i) for s, n in r["plan"]["lab"].items() for i in range(n))
    assert not any("_cpu_s" in k for a in r["arms"].values() for k in a["stages"])
    assert r["status"] == "MEASURED" and r["parity"]["x_bitwise"] and r["parity"]["x_compared"] == w["studies"]["lab"]
    lab = r["arms"]["lab"]
    assert lab["workers"] in (1, r["arms"]["preflight"]["workers"])
    assert len(lab["affinity"]) == 1 if lab["workers"] == 1 else lab["affinity"] == r["host"]["affinity"]
    assert w["ratio"] == pytest.approx(w["lab_s_per_study"] / w["preflight_s_per_decided_study"])
    assert sum(x["agree"] for x in r["per_study"]) == r["decisions"]["agree"]
    assert r["projection"]["status"] == "ANALYTIC"
    assert r["projection"]["lab_s"] == pytest.approx(m.project(w)["lab_s"])
