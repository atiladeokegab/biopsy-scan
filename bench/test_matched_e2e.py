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


def test_the_lab_sample_must_be_a_subset_of_preflights_studies():
    p = m.plans(lab_per_stream=50, calibration=1000, per_stream=1000)
    assert p["lab"] == {"null": 50, "alternative": 50}
    assert p["preflight"] == {"calibration": 1000, "null": 1000, "alternative": 1000}
    with pytest.raises(ValueError):
        m.plans(lab_per_stream=60, calibration=1000, per_stream=50)


def test_matched_walls_count_preflights_calibration_and_checks_features_per_study():
    out = m.compare(lab_out(n=3, wall=30.0), pre_out(n=5, cal=20, wall=9.0))
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
    out = m.compare(lab_out(n=3, shas={("null", 1): "other"}), pre_out())
    assert out["parity"]["x_bitwise"] is False and out["status"] == "FAILED"


def test_the_lab_must_be_one_worker_on_studies_preflight_also_ran():
    with pytest.raises(ValueError, match="one worker"):
        m.compare(lab_out(workers=8), pre_out())
    with pytest.raises(ValueError, match="not in the preflight arm"):
        m.compare(lab_out(n=6), pre_out(n=5))


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
