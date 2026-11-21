"""A-16.2: the R-16 scorer, on synthetic data only. No test here reads a label or outcome file."""
import json

import numpy as np
import pytest

from audit import rediscovery as rs

REAL_PREFLIGHT = rs._split_preflight

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
    hits = [rs.score_feature(*cox_cohort(0.0, n=40, seed=k), RULE)["locked_test_passes"] for k in range(120)]
    assert 0.01 <= np.mean(hits) <= 0.11


def test_harrell_c_by_hand():
    # Comparable pairs (earlier one is an event): (0,1) (0,2) (0,3) (2,3); risk 3>1 yes, 3=3 half, 3<5 no, 3<5 no.
    c = rs.harrell_c(np.array([3.0, 1, 3, 5]), np.array([1.0, 2, 3, 4]), np.array([1, 0, 1, 0], bool))
    assert c == pytest.approx(1.5 / 4)


def test_harrell_c_time_ties():
    """An event and a censoring on the same day: comparable (the censored one was still at risk). Two events on
    the same day: not comparable. Follow-up is in whole days, so this is common."""
    assert rs.harrell_c(np.array([2.0, 1]), np.array([4.0, 4]), np.array([True, False])) == 1.0
    assert np.isnan(rs.harrell_c(np.array([2.0, 1]), np.array([4.0, 4]), np.array([True, True])))


def test_patients_without_focal_cells_are_dropped_and_counted():
    x, t, e, s = cox_cohort(1.0)
    x[:5] = np.nan
    r = rs.score_feature(x, t, e, s, RULE)
    assert r["n_patients"] == len(x) - 5 and r["n_dropped_no_focal_cells"] == 5


def test_a_non_converged_fit_is_unmeasured_and_the_result_is_strict_json():
    """Review repro on #81: a separated fixture made PHReg warn, yet the scorer said MEASURED with CI [0, inf]."""
    x = np.arange(16, 0, -1.0)
    t = np.array([1.0, 2, 3] + [4] * 13)
    e = np.arange(16) < 4
    r = rs.score_feature(x, t, e, np.zeros(16), RULE)
    assert r["status"] == "UNMEASURED" and "Cox fit failed" in r["reason"]
    json.dumps(r, allow_nan=False)
    with pytest.raises(rs.FitFailed):
        rs._cox((x - x.mean()) / x.std(ddof=1), t, e, np.zeros(16))


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


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """A synthetic CLI world: no real data, no git; records every cohort read."""
    import core.cohorts

    reads = []

    def load(cohort, data_dir=None, split="discovery"):
        reads.append(split)
        return fixture_cells([0, 1, 2, 3])

    monkeypatch.setattr(core.cohorts, "load", load)
    monkeypatch.setattr(rs, "RESULTS", tmp_path)
    monkeypatch.setattr(rs, "_code_state", lambda: ("abc1234", False))
    monkeypatch.setattr(rs, "_split_preflight", lambda: rs.APPROVED_SPLIT["manifest_sha256"])
    monkeypatch.setattr(rs, "_hpv", lambda cells, d: np.zeros((len(cells["time"]), 1)))
    monkeypatch.setattr(rs, "rank_on_discovery", lambda *a: {"status": "MEASURED", "rank": 1})
    monkeypatch.setattr(rs, "score_locked", lambda *a: {"status": "UNMEASURED", "reason": "fixture"})
    return {"reads": reads, "dir": tmp_path, "run": lambda *args: rs.main(["--open-locked", *args])}


def test_the_cli_refuses_without_the_flag_or_the_approved_events(cli):
    assert rs.main(["--events", "4", "--out", str(cli["dir"] / "r.json")]) == 2
    for ev in ([], ["--events", "-7"], ["--events", "8"], ["--events", "0"]):
        assert cli["run"](*ev, "--out", str(cli["dir"] / "r.json")) == 2
    assert cli["reads"] == []


def test_the_cli_opens_the_locked_set_once_whatever_the_output_name(cli):
    assert cli["run"]("--events", "4", "--out", str(cli["dir"] / "a.json")) == 0
    assert cli["reads"] == ["discovery", "locked"]
    marker = rs.opening_record(rs.RULE_SHA256, rs.APPROVED_SPLIT["manifest_sha256"])
    rec = json.loads(marker.read_text())
    assert rec["code_sha"] == "abc1234" and rec["discovery"]["status"] == "MEASURED"
    assert json.loads((cli["dir"] / "a.json").read_text())["opening_record"] == marker.name
    assert cli["run"]("--events", "4", "--out", str(cli["dir"] / "b.json")) == 2
    assert cli["reads"] == ["discovery", "locked"] and not (cli["dir"] / "b.json").exists()


def test_the_locked_set_stays_closed_without_a_measured_discovery_rank(cli, monkeypatch):
    monkeypatch.setattr(rs, "rank_on_discovery", lambda *a: {"status": "UNMEASURED", "reason": "no stats.screen"})
    assert cli["run"]("--events", "4", "--out", str(cli["dir"] / "r.json")) == 2
    assert cli["reads"] == ["discovery"] and not list(cli["dir"].glob("upmc-locked-opened.*"))


def test_a_dirty_tree_never_reads_a_cohort(cli, monkeypatch):
    monkeypatch.setattr(rs, "_code_state", lambda: ("abc1234", True))
    assert cli["run"]("--events", "4", "--out", str(cli["dir"] / "r.json")) == 2
    assert cli["reads"] == []


def test_the_split_preflight_fails_closed_before_any_read(cli, monkeypatch, tmp_path):
    """Each preflight condition alone refuses before any cohort read; all of them together pass.
    locked_regions is stubbed, so no test reads or hashes a real data file."""
    import core.cohorts

    monkeypatch.setattr(rs, "_split_preflight", REAL_PREFLIGHT)
    monkeypatch.setattr(core.cohorts, "locked_regions", lambda *a, **k: set())
    lock = tmp_path / "lock.json"
    monkeypatch.setattr(core.cohorts, "LOCK_FILE", lock)
    split_load = core.cohorts.load
    approved = {"sha256": rs.APPROVED_SPLIT["manifest_sha256"]}
    out = str(cli["dir"] / "r.json")

    lock.write_text(json.dumps(approved))
    assert REAL_PREFLIGHT() == approved["sha256"]  # the positive control: every condition holds
    monkeypatch.setattr(core.cohorts, "load", lambda cohort, data_dir=None: cli["reads"].append("unsplit"))
    assert cli["run"]("--events", "4", "--out", out) == 2  # the pre-R-13 loader
    monkeypatch.setattr(core.cohorts, "load", split_load)
    monkeypatch.setattr(core.cohorts, "SPLITS", ("discovery", "all"))
    assert cli["run"]("--events", "4", "--out", out) == 2  # no locked split
    monkeypatch.setattr(core.cohorts, "SPLITS", ("discovery", "locked", "all"))
    monkeypatch.setattr(core.cohorts, "LOCK_FILE", None)
    assert cli["run"]("--events", "4", "--out", out) == 2  # no manifest
    monkeypatch.setattr(core.cohorts, "LOCK_FILE", lock)
    lock.write_text(json.dumps({"sha256": "0" * 64}))
    assert cli["run"]("--events", "4", "--out", out) == 2  # a different locked set
    lock.write_text(json.dumps(approved))

    def tampered(*a, **k):
        raise ValueError("sha256 does not match its content")

    monkeypatch.setattr(core.cohorts, "locked_regions", tampered)
    assert cli["run"]("--events", "4", "--out", out) == 2  # an edited manifest or labels
    assert cli["reads"] == []


def test_the_approved_split_is_the_manifest_on_this_branch():
    import core.cohorts

    assert json.loads(core.cohorts.LOCK_FILE.read_text())["sha256"] == rs.APPROVED_SPLIT["manifest_sha256"]


def test_power_uses_no_outcomes_and_grows_with_events():
    x, _, _, s = cox_cohort(0.0, n=60)
    low, high = rs.power(x[:16], s[:16], 4, sims=60), rs.power(x, s, 40, sims=60)
    assert low["value"] < rs.MIN_POWER < high["value"]
    assert rs.power(x, s, 1)["value"] == 0.0 and rs.power(np.ones(60), s, 30)["value"] == 0.0
    assert high["mean_analysable_events"] == 40 and {"sims", "seed", "event_model", "label"} <= set(high)


def test_power_plants_design_events_before_dropping_missing_features():
    """Half the patients lack the feature: the 4 design events land across all 16, so fewer than 4 stay analysable."""
    x, _, _, s = cox_cohort(0.0, n=16)
    x[::2] = np.nan
    pw = rs.power(x, s, 4, sims=60)
    assert pw["n_missing_feature"] == 8 and pw["n_with_feature"] == 8 and pw["design_events"] == 4
    assert 1 < pw["mean_analysable_events"] < 3.5 and "conditional" in pw["label"]


def test_an_underpowered_locked_set_is_unmeasured_whatever_the_p(monkeypatch):
    monkeypatch.setattr(rs, "power", lambda *a, **k: {"value": 0.3})
    monkeypatch.setattr(rs, "score_feature", lambda *a: {"status": "MEASURED", "p_one_sided": 1e-9})
    out = rs.score_locked(fixture_cells([0, 1, 2, 3]), RULE, SHA, 2)
    assert out["status"] == "UNMEASURED" and out["reason"].startswith("underpowered")
    assert out["power"]["value"] == 0.3 and out["p_one_sided"] == 1e-9
    monkeypatch.setattr(rs, "power", lambda *a, **k: {"value": 0.8})
    assert rs.score_locked(fixture_cells([0, 1, 2, 3]), RULE, SHA, 2)["status"] == "MEASURED"


def test_the_scorer_takes_the_frozen_pre_registration():
    assert rs.rule_sha256(rs.RULE) == rs.RULE_SHA256 == rs.PREREG_SHA256


def test_discovery_rank_counts_stronger_candidates_and_checks_direction():
    names = ["a", RULE["feature"], "b", "c"]
    r = rs.discovery_rank([3.0, 2.0, -2.5, 2.0], names, RULE)
    assert (r["status"], r["rank"], r["of"], r["direction_ok"]) == ("MEASURED", 3, 4, True) and "discovery" in r["label"]
    assert not rs.discovery_rank([0.1, -4.0, 0.2, 0.3], names, RULE)["direction_ok"]


def test_an_uninformative_candidate_is_never_ranked():
    """Review repro on #81: discovery_rank([NaN, 4]) ranked the unavailable rule first."""
    names = [RULE["feature"], "other", "third"]
    r = rs.discovery_rank([np.nan, 4.0, 1.0], names, RULE)
    assert r["status"] == "UNMEASURED" and "rank" not in r
    assert rs.discovery_rank([2.0, 4.0, 1.0], names, RULE, V=[0.0, 1, 1])["status"] == "UNMEASURED"
    r = rs.discovery_rank([2.0, np.nan, 1.0], names, RULE, V=[1.0, 0, 1], n=[30, 0, 30])
    assert (r["rank"], r["of"], r["of_all"], r["uninformative"], r["n_rule"]) == (1, 2, 3, 1, 30)
    json.dumps(r, allow_nan=False)


def test_rank_on_discovery_uses_the_r15_screen():
    pytest.importorskip("stats.screen")
    n_fib = list(range(12))
    cells = fixture_cells(n_fib)
    cells["time"] = 3000.0 - 200 * np.array(n_fib) + np.random.default_rng(2).uniform(0, 50, 12)  # more fibroblasts, earlier
    cells["event"] = np.ones(12, bool)
    r = rs.rank_on_discovery(cells, RULE)
    from core.grammar import feature_names, grammar
    from core.truth import truth
    from stats.screen import screen

    names = feature_names(grammar(TYPES))
    s = screen(truth(cells), cells["time"], cells["event"].astype(float), None, cells["strata"], n_perm=0)
    assert r["status"] == "MEASURED" and r == {**r, **rs.discovery_rank(s["z"], names, RULE, s["V"], s["n"])}
    assert r["direction_ok"] and r["uninformative"] > 0 and r["of"] == r["of_all"] - r["uninformative"]
    cells["time"] = cells["time"][::-1].copy()
    assert not rs.rank_on_discovery(cells, RULE)["direction_ok"]


def test_hpv_is_read_per_patient_from_label_fields(tmp_path):
    (tmp_path / "upmc_labels.csv").write_text("region_id,hpvstatus,survival_day\nr1,1,9\nr2,1,9\nr3,0,9\nr4,NA,9\n")
    cells = {"time": np.zeros(3), "region_names": ["r1", "r2", "r3", "r4"], "patient_of_region": [0, 0, 1, 2]}
    np.testing.assert_array_equal(rs._hpv(cells, tmp_path)[:, 0], [1.0, 0.0, np.nan])


def test_hpv_conflicts_and_invalid_labels_are_refused(tmp_path):
    cells = {"time": np.zeros(1), "region_names": ["r1", "r2"], "patient_of_region": [0, 0]}
    (tmp_path / "upmc_labels.csv").write_text("region_id,hpvstatus\nr1,1\nr2,0\n")
    with pytest.raises(ValueError, match="disagree"):
        rs._hpv(cells, tmp_path)
    (tmp_path / "upmc_labels.csv").write_text("region_id,hpvstatus\nr1,1\nr2,2\n")
    with pytest.raises(ValueError, match="0, 1 or NA"):
        rs._hpv(cells, tmp_path)
    (tmp_path / "upmc_labels.csv").write_text("region_id,hpvstatus\nr1,1\nr2,NA\n")
    assert rs._hpv(cells, tmp_path)[0, 0] == 1.0
