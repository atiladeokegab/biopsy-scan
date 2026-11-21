"""The scaling harness (bench/scaling.py) offline: launch guards, fail-closed parity and the reported ratios."""
import numpy as np
import pytest

from bench import scaling


def _points(cores=scaling.CORES, n=3, wall=lambda c: 100.0 / c):
    rec = [{"i": i, "x_sha": f"x{i}", "z_sha": f"z{i}", "maxT": 1.0 + i, "z": np.array([1.0 + i, -0.5])}
           for i in range(n)]
    return {c: {"workers": c, "studies_wall_s": wall(c), "records": [dict(r, z=r["z"].copy()) for r in rec],
                "host": {"thread_environment": dict(scaling.SINGLE)}} for c in cores}


@pytest.mark.parametrize("bad", ["0,1,8", "1,8,8", "1,2,4", "2,8", "1,8,65", "1,8,x"])
def test_bad_core_lists_stop_before_any_launch(bad):
    with pytest.raises(SystemExit):
        scaling.parse_cores(bad)


def test_core_list_is_sorted():
    assert scaling.parse_cores("64,8,1,2") == (1, 2, 8, 64)


def test_default_sweep_fits_the_five_dollar_cap_and_a_bigger_one_is_refused():
    assert scaling.check_cap(scaling.CORES, 1000, spent_usd=0.0) <= scaling.CAP_USD
    with pytest.raises(SystemExit):
        scaling.check_cap(scaling.CORES, 1000, spent_usd=scaling.CAP_USD)


def test_one_study_with_different_x_at_one_core_count_fails_with_no_timings():
    pts = _points()
    pts[32]["records"][1]["x_sha"] = "different"
    out = scaling.result(pts, scaling.CORES, {}, None)
    assert out["status"] == "FAILED" and "curve" not in out
    assert out["parity"]["by_cores"][32]["x_identical"] is False


def test_z_past_tolerance_fails_and_a_last_bit_difference_is_reported_not_hidden():
    pts = _points()
    pts[32]["records"][1]["z"][0] += 1e-6
    assert scaling.result(pts, scaling.CORES, {}, None)["status"] == "FAILED"
    pts = _points()
    pts[2]["records"][0]["z"][0] = np.nextafter(1.0, 2.0)
    pts[2]["records"][0]["z_sha"] = "last bit"
    out = scaling.result(pts, scaling.CORES, {}, None)
    row = out["parity"]["by_cores"][2]
    assert out["status"] == "MEASURED" and row["z_identical"] is False and 0 < row["z_max_rel_diff"] <= scaling.Z_TOL


def test_a_failed_or_missing_container_or_wrong_worker_count_fails():
    pts = _points()
    del pts[64]
    assert scaling.result(pts, scaling.CORES, {}, None)["status"] == "FAILED"
    assert scaling.result(_points(), scaling.CORES, {16: "timeout"}, None)["status"] == "FAILED"
    pts = _points()
    pts[4]["workers"] = 8
    assert scaling.result(pts, scaling.CORES, {}, None)["status"] == "FAILED"


def test_a_container_with_more_than_one_blas_thread_fails():
    pts = _points()
    pts[8]["host"]["thread_environment"]["OMP_NUM_THREADS"] = "8"  # what Modal set in the 20:02Z smoke
    out = scaling.result(pts, scaling.CORES, {}, None)
    assert out["status"] == "FAILED" and "curve" not in out


def test_curve_reports_wall_per_study_and_speedup_vs_one_core_only():
    out = scaling.result(_points(wall=lambda c: {1: 90.0}.get(c, 30.0)), scaling.CORES, {}, None)
    assert out["status"] == "MEASURED"
    row = {r["cores"]: r for r in out["curve"]}
    assert row[1]["speedup_vs_1_core"] == 1.0 and row[8]["speedup_vs_1_core"] == 3.0
    assert row[8]["wall_per_study_s"] == 10.0
    assert not any("efficien" in k or "ideal" in k for r in out["curve"] for k in r)


def test_eight_core_point_is_checked_against_build9_committed_arm():
    pts = _points()
    prior = {"arms": {"numba": {"studies": [{"stream": "null", "i": 0, "x_sha": "x0", "maxT": 1.0},
                                            {"stream": "alternative", "i": 1, "x_sha": "other", "maxT": 9.0},
                                            {"stream": "null", "i": 1, "x_sha": "x1", "maxT": 2.5}]}}}
    got = scaling.result(pts, scaling.CORES, {}, prior)["parity"]["against_build9_8core"]
    assert got == {"x_identical": True, "maxT_identical": False, "compared": 2}
