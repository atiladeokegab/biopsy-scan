"""The scaling harness (bench/scaling.py) offline: launch guards, fail-closed coverage and parity, reported ratios."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bench import scaling

N = 3


def _hex(tag):
    return scaling.hashlib.sha256(str(tag).encode()).hexdigest()


def _rehash(r):
    r["z_sha"] = scaling.z_sha(r["z"])
    return r


def _points(cores=scaling.CORES, n=N, wall=lambda c: 100.0 / c):
    rec = [{"i": i, "x_sha": _hex(i), "maxT": 1.0 + i, "z": np.array([1.0 + i, -0.5])} for i in range(n)]
    return {c: {"workers": c, "studies_wall_s": wall(c), "records": [_rehash(dict(r, z=r["z"].copy())) for r in rec],
                "host": {"thread_environment": dict(scaling.SINGLE)}} for c in cores}


def _result(pts, errors=None, prior=None, n=N):
    return scaling.result(pts, scaling.CORES, errors or {}, prior, n)


def _failed(out):
    return out["status"] == "FAILED" and "curve" not in out


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


@pytest.mark.parametrize("spent,cap", [(float("nan"), 5.0), (-1.0, 5.0), (float("inf"), 5.0), (0.0, float("nan")),
                                       (0.0, 0.0), (0.0, -5.0), ("0", 5.0), (True, 5.0)])
def test_malformed_spend_or_cap_is_refused(spent, cap):
    with pytest.raises(SystemExit):
        scaling.check_cap(scaling.CORES, 1000, spent_usd=spent, cap_usd=cap)


def test_a_nan_in_the_ledger_never_reaches_a_paid_call(monkeypatch, tmp_path):
    from bench import modal_scaling as ms

    ledger = tmp_path / "spend.json"
    ledger.write_text(json.dumps({"entries": [{"utc": "x", "run": "modal-scaling old", "est_usd": float("nan"),
                                                "running_total_usd": 1.0}]}))
    monkeypatch.setattr(ms, "SPEND", ledger)
    monkeypatch.setattr(ms, "_git", lambda *a: "abc1234" if a[0] == "rev-parse" else "")
    monkeypatch.setattr(ms.point, "with_options", lambda **k: pytest.fail("a paid call was dispatched"))
    with pytest.raises(SystemExit, match="bad spend"):
        ms.main(studies=10, cores="1,8")
    assert len(json.loads(ledger.read_text())["entries"]) == 1  # nothing logged as launched


def test_one_study_with_different_x_at_one_core_count_fails_with_no_timings():
    pts = _points()
    pts[32]["records"][1]["x_sha"] = _hex("other")
    out = _result(pts)
    assert _failed(out) and out["parity"]["by_cores"][32]["x_identical"] is False


def test_z_past_tolerance_fails_and_a_last_bit_difference_is_reported_not_hidden():
    pts = _points()
    pts[32]["records"][1]["z"][0] += 1e-6
    _rehash(pts[32]["records"][1])
    assert _failed(_result(pts))
    pts = _points()
    pts[2]["records"][0]["z"][0] = np.nextafter(1.0, 2.0)
    _rehash(pts[2]["records"][0])
    out = _result(pts)
    row = out["parity"]["by_cores"][2]
    assert out["status"] == "MEASURED" and row["z_identical"] is False and 0 < row["z_max_rel_diff"] <= scaling.Z_TOL


def _dup(p):
    p["records"].append(dict(p["records"][0]))


def _drop_everywhere_but_keep_count(p):
    p["records"][-1]["i"] = 0  # count unchanged, ID 2 missing, ID 0 twice


def _nan_late(p):
    p["records"][-1]["z"][1] = np.nan
    _rehash(p["records"][-1])


def _one_component(p):
    for r in p["records"]:
        r["z"] = r["z"][:1]
        _rehash(r)


def _stale_hash(p):
    p["records"][0]["z_sha"] = _hex("stale")  # z unchanged: only the recomputed hash can catch a wrong exactness report


@pytest.mark.parametrize("mutate", [
    _dup, _drop_everywhere_but_keep_count, _nan_late, _one_component, _stale_hash,
    lambda p: p.update(studies_wall_s=float("nan")), lambda p: p.update(studies_wall_s=-3.0),
    lambda p: p.update(studies_wall_s=0.0), lambda p: p["host"]["thread_environment"].pop("MKL_NUM_THREADS"),
    lambda p: p["host"]["thread_environment"].update(OMP_NUM_THREADS="8"),
    lambda p: p["host"].update(thread_environment={}), lambda p: p["records"][0].update(x_sha="x0"),
    lambda p: p["records"][0].update(i="0"),
], ids=["duplicate", "missing-id", "nan-late-z", "one-component-z", "stale-z-sha", "nan-wall", "negative-wall",
        "zero-wall", "missing-thread-key", "omp-8", "no-threads", "bad-x-sha", "string-id"])
@pytest.mark.parametrize("cores", [1, 8, 64])
def test_a_malformed_point_fails_with_no_timings(mutate, cores):
    pts = _points()
    mutate(pts[cores])
    assert _failed(_result(pts))


def test_an_id_missing_from_every_point_fails():
    pts = _points()
    for p in pts.values():
        p["records"].pop()
    assert _failed(_result(pts))


def test_a_one_component_z_never_broadcasts_against_a_constant_reference():
    pts = _points()
    for c, p in pts.items():
        for r in p["records"]:
            r["z"] = np.array([2.0, 2.0]) if c == scaling.REFERENCE_CORES else np.array([2.0])
            _rehash(r)
    assert _failed(_result(pts))


def test_a_failed_or_missing_container_or_wrong_worker_count_fails():
    pts = _points()
    del pts[64]
    assert _failed(_result(pts))
    assert _failed(_result(_points(), errors={16: "timeout"}))
    pts = _points()
    pts[4]["workers"] = 8
    assert _failed(_result(pts))


def test_curve_reports_wall_per_study_and_speedup_vs_one_core_only_with_its_limits():
    out = _result(_points(wall=lambda c: {1: 90.0}.get(c, 30.0)))
    assert out["status"] == "MEASURED"
    row = {r["cores"]: r for r in out["curve"]}
    assert row[1]["speedup_vs_1_core"] == 1.0 and row[8]["speedup_vs_1_core"] == 3.0
    assert row[8]["wall_per_study_s"] == 10.0
    assert not any("efficien" in k or "ideal" in k for r in out["curve"] for k in r)
    assert {"sweep", "hosts", "memory", "preemption", "ratios", "parity", "spend"} <= set(out["caveats"])
    assert "not a billing ceiling" in out["caveats"]["spend"] and "not CPU time" in out["fields"]["reimage_worker_wall_s"]


def test_eight_core_point_is_checked_against_build9_committed_arm():
    prior = {"arms": {"numba": {"studies": [{"stream": "null", "i": 0, "x_sha": _hex(0), "maxT": 1.0},
                                            {"stream": "alternative", "i": 1, "x_sha": "other", "maxT": 9.0},
                                            {"stream": "null", "i": 1, "x_sha": _hex(1), "maxT": 2.5}]}}}
    got = _result(_points(), prior=prior)["parity"]["against_build9_8core"]
    assert got == {"x_identical": True, "maxT_identical": False, "compared": 2}


COMMITTED = Path(__file__).parent / "results" / "modal-scaling-20261121T201008Z.json"


def test_committed_sweep_covers_the_plan_and_its_ratios_reconcile():
    """What the artifact still allows: exact IDs, X parity, threads, finite walls and the published ratios."""
    doc = json.loads(COMMITTED.read_text())
    assert doc["status"] == "MEASURED" and doc["parity"]["all"] is True
    ref = {s["i"]: s["x_sha"] for s in doc["studies_by_cores"]["8"]}
    for c, rows in doc["studies_by_cores"].items():
        assert sorted(s["i"] for s in rows) == list(range(1000))
        assert all(ref[s["i"]] == s["x_sha"] for s in rows)
        p = doc["points"][c]
        assert p["workers"] == int(c) and p["host"]["thread_environment"] == scaling.SINGLE
    base = doc["points"]["1"]["studies_wall_s"]
    for row in doc["curve"]:
        wall = doc["points"][str(row["cores"])]["studies_wall_s"]
        assert math.isfinite(wall) and wall > 0 and row["studies_wall_s"] == wall
        assert row["speedup_vs_1_core"] == base / wall and row["wall_per_study_s"] == wall / 1000


def test_committed_sweep_carries_its_limits_in_a_sidecar():
    side = json.loads(COMMITTED.with_name(COMMITTED.stem + ".caveats.json").read_text())
    assert side["result"] == "bench/results/" + COMMITTED.name and side["caveats"] == scaling.CAVEATS
    assert "not CPU time" in side["fields"]["reimage_cpu_s"] and "not a billing ceiling" in side["spend_in_ledger"]
