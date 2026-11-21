"""The simulator's predicted power checked against the real Charville cohort (plasmode, real KM baseline)."""

import json
from pathlib import Path

import numpy as np
import pytest

from atlas.runner import interval
from bench import real_check as rc

RESULT = Path(__file__).resolve().parent / "results" / "real-check-charville.json"
ONE = {0: {"time": np.array([1e9]), "mass": np.array([1.0])}}  # censoring far past every event: no censoring


def test_km_outcomes_at_hr1_redraw_the_source_event_times():
    source = {"time": np.arange(1.0, 11.0), "event": np.ones(10, int), "strata": np.zeros(10, int)}
    t, e, s = rc.km_outcomes(source, np.zeros(10), 1.0, np.zeros(20000, int), ONE, 1, 2)
    assert e.all() and (s == 0).all() and set(np.unique(t)) <= set(source["time"])
    share = np.bincount(t.astype(int), minlength=11)[1:] / len(t)
    assert np.allclose(share, 0.1, atol=0.01)


def test_km_outcomes_are_proportional_hazards_on_the_km_baseline():
    rng = np.random.default_rng(0)
    source = {"time": np.arange(1.0, 41.0), "event": rng.integers(0, 2, 40), "strata": np.zeros(40, int)}
    source["event"][-1] = 1  # the KM reaches zero, so every draw has an event time
    z = np.array([-1.0, 1.0])
    t, e, _ = rc.km_outcomes(source, z, 2.0, np.repeat([0, 1], 40000), ONE, 3, 4)
    km = rc.kaplan_meier(source["time"], source["event"])
    for j, r in ((0, 0.5), (1, 2.0)):
        tj = t[j * 40000:(j + 1) * 40000]
        for at, s0 in zip(*km):
            assert abs(np.mean(tj > at) - s0 ** r) < 0.01
    assert np.median(t[40000:]) < np.median(t[:40000])


def test_km_outcomes_censor_by_reverse_km_and_past_the_last_event():
    source = {"time": np.array([1.0, 2.0, 3.0]), "event": np.array([1, 1, 0]), "strata": np.zeros(3, int)}
    censor = {0: {"time": np.array([2.0]), "mass": np.array([1.0])}}
    t, e, _ = rc.km_outcomes(source, np.zeros(3), 1.0, np.zeros(30000, int), censor, 5, 6)
    assert (t <= 2.0).all()
    assert np.allclose(np.mean(e), 2 / 3, atol=0.01)  # T=1 or T=2 (a tie at C counts as the event), else censored
    assert e[t < 2.0].all()


def test_km_outcomes_refuse_a_stratum_without_events_or_censoring():
    source = {"time": np.array([1.0, 2.0]), "event": np.array([0, 0]), "strata": np.array([0, 1])}
    with pytest.raises(ValueError, match="stratum"):
        rc.km_outcomes(source, np.zeros(2), 2.0, np.array([0, 1]), ONE, 1, 2)


def _toy():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 5))
    X[3, 2] = np.nan
    source = {"time": rng.exponential(size=60) + 0.01, "event": rng.integers(0, 2, 60), "strata": np.repeat([0, 1], 30)}
    return X, source


def test_run_arm_counts_rejections_over_a_null_calibrated_threshold():
    X, source = _toy()
    z = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    censor = {s: {"time": np.array([5.0]), "mass": np.array([1.0])} for s in (0, 1)}
    gen = lambda hr: lambda p, a, b: rc.km_outcomes(source, z, hr, p, censor, a, b)  # noqa: E731
    arm = rc.run_arm(X, gen(1.0), gen(3.0), 40, calibration=50, alternative=30, seed=11, alpha=0.1)
    assert arm["calibration_n"] == 50 and arm["power_n"] == 30 == len(arm["rejects"])
    assert arm["power_count"] == sum(arm["rejects"]) and arm["power"] == arm["power_count"] / 30
    assert (arm["power_lo"], arm["power_hi"]) == interval(arm["power_count"], 30)
    assert np.isfinite(arm["threshold"]) and arm["mean_events"] > 0
    again = rc.run_arm(X, gen(1.0), gen(3.0), 40, calibration=50, alternative=30, seed=11, alpha=0.1)
    assert again == arm  # seeded: the same draws every time


def test_arms_share_patient_draws_so_the_comparison_is_paired():
    X, source = _toy()
    z = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    censor = {s: {"time": np.array([5.0]), "mass": np.array([1.0])} for s in (0, 1)}
    gen = lambda hr: lambda p, a, b: rc.km_outcomes(source, z, hr, p, censor, a, b)  # noqa: E731
    a = rc.run_arm(X, gen(1.0), gen(3.0), 40, calibration=50, alternative=30, seed=11, alpha=0.1)
    c = rc.compare(a, a)
    assert c["difference"] == 0 and c["difference_lo"] <= 0 <= c["difference_hi"] and c["agree"]
    assert c["patients_sha256"] == a["patients_sha256"]
    b = rc.run_arm(X, gen(1.0), gen(1.0), 40, calibration=50, alternative=30, seed=11, alpha=0.1)
    assert b["patients_sha256"] == a["patients_sha256"]
    with pytest.raises(ValueError, match="paired"):
        rc.compare(a, rc.run_arm(X, gen(1.0), gen(3.0), 40, calibration=50, alternative=30, seed=12, alpha=0.1))


def test_the_committed_result_labels_every_number_and_keeps_counts():
    res = json.loads(RESULT.read_text())
    assert res["schema"] == "preflight-real-check/1" and res["cohort"] == "charville"
    assert res["source"]["patients"] == 159 and res["source"]["events"] == 38
    assert res["design"] == {"regions_per_patient": "all", "fov_um": "full core", "eps": 0.0}
    assert res["hr_per_sd"] == 2.0 and res["alpha"] == 0.05
    assert [p["patients"] for p in res["points"]] == [50, 100, 159]
    labels = {k: res["arms"][k]["label"] for k in ("simulator", "empirical")}
    assert labels == {"simulator": "MEASURED (sim)", "empirical": "MEASURED (plasmode)"}
    for p in res["points"]:
        for k in ("simulator", "empirical"):
            a = p[k]
            assert a["power"] == a["power_count"] / a["power_n"] and a["power_n"] >= 1000
            assert [a["power_lo"], a["power_hi"]] == pytest.approx(interval(a["power_count"], a["power_n"]))
            assert a["calibration_n"] >= 2000
        assert p["difference"] == pytest.approx(p["simulator"]["power"] - p["empirical"]["power"])
        assert p["agree"] == (p["difference_lo"] <= 0 <= p["difference_hi"])
    assert res["identity_check"]["studies"] >= 1 and res["identity_check"]["equal"] is True
    assert "not the imaging-noise model" in res["scope"]


def test_compare_goes_red_when_the_arms_really_differ():
    arm = lambda flags: {"patients_sha256": "x", "power_n": len(flags), "power": sum(flags) / len(flags),  # noqa: E731
                         "rejects": flags}
    c = rc.compare(arm([True] * 400), arm([False] * 300 + [True] * 100))
    assert c["difference"] == 0.75 and c["difference_lo"] > 0 and not c["agree"] and c["discordant"] == 300
