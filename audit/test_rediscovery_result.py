"""A-16.2 (locked half): the R-16 scorer, on synthetic data only. No test here reads a label or outcome file."""
import json

import numpy as np
import pytest

from audit import rediscovery_score as rs

RULE = {"expressible": True, "feature": "Tumor (Podo+)|Stromal / Fibroblast|25um|fraction",
        "direction": "higher is worse: hazard ratio > 1; a top rank with HR < 1 counts as not rediscovered"}
SHA = rs.rule_sha256(RULE)
TYPES = ["Macrophage", "Stromal / Fibroblast", "Tumor", "Tumor (Podo+)"]


def cox_cohort(beta, n=120, seed=0):
    """Exponential survival with log-hazard beta * x, random censoring, two strata."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    t = rng.exponential(1 / np.exp(beta * (x - x.mean()) / x.std()))
    c = rng.exponential(np.median(t) * 2, n)
    return x, np.minimum(t, c), t <= c, np.arange(n) % 2


def test_a_real_harmful_effect_passes_the_one_sided_test():
    r = rs.score_feature(*cox_cohort(1.0), RULE)
    assert r["status"] == "MEASURED" and r["hr_per_sd"] > 1.5
    assert r["p_one_sided"] < 0.001 and r["locked_test_passes"]
    assert r["c_index"] > 0.6 and r["label"].startswith("exploratory")


def test_a_protective_effect_fails_the_one_sided_test():
    """HR < 1 is the wrong direction: the one-sided p is near 1, never 'passes'."""
    r = rs.score_feature(*cox_cohort(-1.0), RULE)
    assert r["hr_per_sd"] < 1 and r["p_one_sided"] > 0.99 and not r["locked_test_passes"]
    assert r["c_index"] < 0.4


def test_matches_a_direct_statsmodels_fit():
    from statsmodels.duration.hazard_regression import PHReg

    x, t, e, s = cox_cohort(0.4, n=60, seed=3)
    z = (x - x.mean()) / x.std(ddof=1)
    fit = PHReg(t, z[:, None], status=e.astype(float), strata=s, ties="breslow").fit(disp=False)
    r = rs.score_feature(x, t, e, s, RULE)
    assert r["log_hr_per_sd"] == pytest.approx(fit.params[0]) and r["se"] == pytest.approx(fit.bse[0])


def test_the_null_rejects_about_alpha_of_the_time():
    hits = [rs.score_feature(*cox_cohort(0.0, n=40, seed=k), RULE)["locked_test_passes"] for k in range(200)]
    assert 0.01 <= np.mean(hits) <= 0.10


def test_harrell_c_by_hand():
    # Comparable pairs (earlier one is an event): (0,1) (0,2) (0,3) (2,3); risk 3>1 yes, 3=3 half, 3<5 no, 3<5 no.
    c = rs.harrell_c(np.array([3.0, 1, 3, 5]), np.array([1.0, 2, 3, 4]), np.array([1, 0, 1, 0], bool))
    assert c == pytest.approx(1.5 / 4)


def test_patients_without_focal_cells_are_dropped_and_counted():
    x, t, e, s = cox_cohort(1.0)
    x[:5] = np.nan
    r = rs.score_feature(x, t, e, s, RULE)
    assert r["n_patients"] == len(x) - 5 and r["n_dropped_no_focal_cells"] == 5


def test_not_estimable_is_unmeasured_with_a_reason():
    x, t, e, s = cox_cohort(1.0, n=10)
    r = rs.score_feature(x, t, np.zeros(10, bool), s, RULE)
    assert r["status"] == "UNMEASURED" and "0 events" in r["reason"]
    r = rs.score_feature(np.ones(10), t, e, s, RULE)
    assert r["status"] == "UNMEASURED" and "SD" in r["reason"]
    assert rs.score_feature(x, t, e, s, {**RULE, "expressible": False})["status"] == "UNMEASURED"


def fixture_cells(n_fib):
    """One region per patient: a Podo+ tumour cell at the origin, n_fib fibroblasts and one macrophage
    within 10 um, a plain tumour cell 30 um away (outside 25 um). Feature = n_fib / (n_fib + 1)."""
    ty, x, y, rid = [], [], [], []
    for p, k in enumerate(n_fib):
        cells = [(3, 0, 0), (0, 0, 10)] + [(1, 5 + i, 0) for i in range(k)] + [(2, 30, 0)]
        for c, cx, cy in cells:
            ty.append(c), x.append(cx), y.append(cy), rid.append(p)
    P = len(n_fib)
    rng = np.random.default_rng(1)
    return {"cell_type": np.array(ty, np.int8), "x": np.array(x, np.float32), "y": np.array(y, np.float32),
            "region_id": np.array(rid, np.int32), "patient_of_region": np.arange(P, dtype=np.int32),
            "time": rng.uniform(100, 2000, P), "event": rng.uniform(size=P) < 0.5,
            "strata": np.zeros(P, np.int32), "type_names": TYPES}


def test_score_locked_computes_the_pre_registered_column(monkeypatch):
    n_fib = [0, 1, 2, 3, 4, 5, 6, 7]
    seen = {}
    monkeypatch.setattr(rs, "score_feature", lambda x, *a: seen.setdefault("x", x) is None or {"status": "X"})
    out = rs.score_locked(fixture_cells(n_fib), RULE, SHA, 4)
    np.testing.assert_allclose(seen["x"], [k / (k + 1) for k in n_fib])
    assert out["rule_sha256"] == SHA and out["feature"] == RULE["feature"] and "one-sided" in out["test"]


def test_an_edited_rule_is_refused():
    with pytest.raises(ValueError, match="sha256"):
        rs.score_locked(fixture_cells([1, 2]), {**RULE, "feature": RULE["feature"].replace("25um", "50um")}, SHA, 1)


def test_a_feature_the_cohort_cannot_express_is_unmeasured():
    rule = {**RULE, "feature": "Tumor (Podo+)|Vessel|25um|fraction"}
    out = rs.score_locked(fixture_cells([1, 2]), rule, rs.rule_sha256(rule), 1)
    assert out["status"] == "UNMEASURED"


def test_the_cli_never_opens_the_locked_set_without_the_flag(tmp_path, monkeypatch):
    import core.cohorts

    monkeypatch.setattr(core.cohorts, "load", lambda *a, **k: pytest.fail("the locked set was opened"))
    assert rs.main(["--out", str(tmp_path / "r.json")]) == 2
    assert rs.main(["--open-locked", "--out", str(tmp_path / "r.json")]) == 2  # no --events: power not fixed
    (tmp_path / "r.json").write_text(json.dumps({"already": "opened"}))
    assert rs.main(["--open-locked", "--events", "8", "--out", str(tmp_path / "r.json")]) == 2


def test_power_uses_no_outcomes_and_grows_with_events():
    x, _, _, s = cox_cohort(0.0, n=60)
    low, high = rs.power(x[:16], s[:16], 4, sims=200), rs.power(x, s, 40, sims=200)
    assert low < rs.MIN_POWER < high
    assert rs.power(x, s, 1) == 0.0 and rs.power(np.ones(60), s, 30) == 0.0


def test_an_underpowered_locked_set_is_unmeasured_whatever_the_p(monkeypatch):
    monkeypatch.setattr(rs, "power", lambda *a, **k: 0.3)
    monkeypatch.setattr(rs, "score_feature", lambda *a: {"status": "MEASURED", "p_one_sided": 1e-9})
    out = rs.score_locked(fixture_cells([0, 1, 2, 3]), RULE, SHA, 2)
    assert out["status"] == "UNMEASURED" and out["reason"].startswith("underpowered")
    assert out["power"]["value"] == 0.3 and out["p_one_sided"] == 1e-9
    monkeypatch.setattr(rs, "power", lambda *a, **k: 0.8)
    assert rs.score_locked(fixture_cells([0, 1, 2, 3]), RULE, SHA, 2)["status"] == "MEASURED"
