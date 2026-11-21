"""A-18.1: the Charville screen goes through stats.screen, and its output is exploratory and MEASURED."""
import json
from pathlib import Path

import numpy as np
import pytest

from audit import screen_cohort

TYPES = ["Stroma", "Tumor 1"]


def _cells(cohort, strata=None):
    rng = np.random.default_rng(1)
    return {"type_names": TYPES, "time": rng.uniform(1, 100, 8), "event": np.r_[np.ones(4), np.zeros(4)],
            "strata": np.r_[np.zeros(4), np.ones(4)] if strata is None else strata, "cohort": cohort}


def _run(cohort, calls, strata=None, **kw):
    def fake_screen(X, time, event, **k):
        calls.append(k)
        z = np.arange(X.shape[1], dtype=float)
        return {"z": z, "adjusted_p": np.ones_like(z), "p": 0.5}
    return screen_cohort.run(cohort, load_fn=lambda c, **_: _cells(c, strata),
                             truth_fn=lambda c: np.random.default_rng(2).normal(size=(8, 72)), screen_fn=fake_screen, **kw)


def test_charville_runs_with_no_covariates_and_coverslip_strata():
    calls = []
    _run("charville", calls)
    assert calls[0]["clinical"] is None and calls[0]["strata"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


@pytest.mark.parametrize("cohort,n_perm", [("upmc", 999), ("charville", 0), ("charville", -1)])
def test_unsupported_input_is_rejected_before_any_data_read(cohort, n_perm):
    def load_fn(*a, **k):
        raise AssertionError("data was read")
    with pytest.raises(ValueError):
        screen_cohort.run(cohort, n_perm=n_perm, load_fn=load_fn)


def test_missing_stratum_reaches_the_screen_as_missing():
    calls = []
    _run("charville", calls, strata=np.r_[np.zeros(4), np.nan, np.ones(3)])
    s = calls[0]["strata"]
    assert np.isnan(s[4]) and np.isfinite(np.delete(s, 4)).all()


def test_labels_and_top_candidates():
    r = _run("charville", [])
    assert r["status"] == "MEASURED" and r["inference"].startswith("exploratory")
    assert r["n_features_tested"] == 72 and r["top_candidates"][0]["feature"].startswith("Tumor 1|")
    assert [c["rank"] for c in r["top_candidates"]] == list(range(1, 11))


def test_committed_result_is_labelled_and_strict_json():
    path = Path(__file__).resolve().parents[1] / "results" / "charville-screen.json"
    r = json.loads(path.read_text(), parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert r["status"] == "MEASURED" and r["inference"].startswith("exploratory")
    assert (r["n_patients"], r["n_events"], r["n_features_tested"]) == (159, 38, 3780)
    assert 0 < r["maxT"]["p"] <= 1 and r["n_perm"] >= 1
