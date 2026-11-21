"""A-18.1: one screen path for every cohort; Charville output is exploratory and MEASURED."""
import inspect
import json
from pathlib import Path

import numpy as np

from audit import screen_cohort

TYPES = ["Stroma", "Tumor 1"]


def _cells(cohort):
    rng = np.random.default_rng(1)
    return {"type_names": TYPES, "time": rng.uniform(1, 100, 8), "event": np.r_[np.ones(4), np.zeros(4)],
            "strata": np.r_[np.zeros(4), np.ones(4)].astype(np.int32), "cohort": cohort}


def _run(cohort, calls):
    def fake_screen(X, time, event, **kw):
        calls.append(kw)
        z = np.arange(X.shape[1], dtype=float)
        return {"z": z, "adjusted_p": np.ones_like(z), "p": 0.5}
    return screen_cohort.run(cohort, load_fn=lambda c, **_: _cells(c),
                             truth_fn=lambda c: np.random.default_rng(2).normal(size=(8, 72)), screen_fn=fake_screen)


def test_same_path_for_both_cohorts_no_covariates_coverslip_strata():
    calls = []
    a, b = _run("charville", calls), _run("upmc", calls)
    assert [k for k in a if k not in ("cohort", "wall_s")] == [k for k in b if k not in ("cohort", "wall_s")]
    for kw in calls:
        assert kw["clinical"] is None and kw["strata"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    src = inspect.getsource(screen_cohort.run)
    assert "charville" not in src.lower() and "upmc" not in src.lower()


def test_labels_and_top_candidates():
    r = _run("charville", [])
    assert r["status"] == "MEASURED" and r["inference"].startswith("exploratory")
    assert r["n_features_tested"] == 72 and r["top_candidates"][0]["feature"].startswith("Tumor 1|")
    assert [c["rank"] for c in r["top_candidates"]] == list(range(1, 11))


def test_committed_result_is_labelled():
    path = Path(__file__).resolve().parents[1] / "results" / "charville-screen.json"
    if path.exists():
        r = json.loads(path.read_text())
        assert r["status"] == "MEASURED" and r["inference"].startswith("exploratory")
        assert (r["n_patients"], r["n_events"], r["n_features_tested"]) == (159, 38, 3780)
