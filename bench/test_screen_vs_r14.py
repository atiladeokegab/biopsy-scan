"""The A/B report keeps its labels, interleaves arms and computes the ratio from medians."""

import numpy as np

from bench.screen_vs_r14 import report, run


def test_interleaved_report_on_a_tiny_cohort(monkeypatch):
    import bench.screen_vs_r14 as ab

    rng = np.random.default_rng(0)
    n, p = 30, 4
    d = {"X": rng.normal(size=(n, p)), "time": rng.exponential(size=n), "event": (rng.random(n) < 0.7).astype(float),
         "C": np.zeros((n, 0)), "strata": rng.integers(0, 2, n), "patient_ids": [f"p{i}" for i in range(n)],
         "features": "v2"}
    monkeypatch.setattr(ab, "time_full_run", lambda *a, workers: (2.0, {}))
    reps, _, b = run(d, 2, 1)
    assert [x["arm"] for x in reps] == ["A", "B", "A", "B"]
    out = report(d, reps, b, 1)
    assert out["label"] == "MEASURED" and out["comparison"].startswith("workflow comparison")
    assert out["speedup_median"] == 2.0 / out["arms"]["B"]["median_s"]
    assert all("load_1min_before" in x for x in reps)
