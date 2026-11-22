"""bench.race_local on invented tissue: both arms run, the report carries the label, and it fits CI time."""
from time import perf_counter

import numpy as np

from bench import race_local as rl

NAMES = ["B cell", "Blood vessel", "CD4 T cell", "CD8 T cell", "Granulocyte", "Macrophage", "Other", "Stroma", "Tumor 1",
         "Tumor 2 (Ki67 Proliferating)", "Tumor 3", "Tumor 4", "Tumor 5", "Tumor 6 / DC", "Tumor 7"]  # Charville's


def invented(patients=12, regions=2, per=150, seed=0):
    """Random cells in the contract's shape (no real data): 12 patients, 2 regions each, Charville's type names."""
    g, n = np.random.default_rng(seed), patients * regions * per
    return {"cell_type": g.integers(0, len(NAMES), n).astype(np.int8),
            "x": g.uniform(0, 600, n).astype(np.float32), "y": g.uniform(0, 600, n).astype(np.float32),
            "region_id": np.repeat(np.arange(patients * regions), per).astype(np.int32),
            "patient_of_region": np.repeat(np.arange(patients), regions).astype(np.int32),
            "time": g.uniform(100, 3000, patients), "event": g.random(patients) < 0.5,
            "strata": np.zeros(patients, np.int32), "type_names": NAMES}


def test_a_tiny_race_runs_both_arms_with_parity_inside_ci_time():
    from core import truth
    serial = truth.neighbours
    t = perf_counter()
    r = rl.race(invented(), studies=2, workers=2)
    assert perf_counter() - t < 60
    assert truth.neighbours is serial  # the pool-built adjacency is used only inside the race
    assert r["evaluated"] == {"lab": 4, "fused": 6}  # the lab skips calibration; Preflight pays for it
    assert r["parity"] == {"x_bitwise": True, "x_compared": 4}
    assert r["s_per_study"]["lab"] > 0 and r["s_per_study"]["fused"] > 0
    assert r["ratio"] == r["s_per_study"]["lab"] / r["s_per_study"]["fused"]
    text = rl.report(r)
    assert text.startswith("race-local: MEASURED on this machine; not the Modal figures")
    assert "bit-identical on 4 shared studies" in text and "s/study" in text


def test_a_feature_mismatch_is_reported_and_fails_the_run(monkeypatch, capsys):
    real = rl.db.parity
    monkeypatch.setattr(rl.db, "parity", lambda arms: {**real(arms), "x_bitwise": False})
    monkeypatch.setattr("core.cohorts.load", lambda cohort: invented())
    assert rl.main(["--studies", "1", "--workers", "1"]) == 1
    assert "MISMATCH" in capsys.readouterr().out
