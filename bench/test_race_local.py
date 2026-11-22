"""bench.race_local on invented tissue: both arms run, the report carries the label, and it fits CI time."""
from time import perf_counter

import numpy as np
import pytest

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
    serial, reference = truth.neighbours, truth.truth
    t = perf_counter()
    r = rl.race(invented(), studies=2, workers=2)
    assert perf_counter() - t < 60
    assert truth.neighbours is serial and truth.truth is reference  # the shortcuts live only inside the race
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


def test_the_cohort_truth_is_computed_once_for_both_arms(monkeypatch):
    from core import truth
    calls, real = [], truth.truth
    monkeypatch.setattr(truth, "truth", lambda c: calls.append(1) or real(c))
    rl.race(invented(), studies=1, workers=1)
    assert len(calls) == 1


def stub_arm(edit=None):
    """run_arm's shape without running it: every planned ID once, a 64-hex feature hash and a finite statistic."""
    def run_arm(arm, cells, plan, workers=8):
        stat = {"p": 0.5, "test_s": 1.0} if arm == "lab" else {"z": np.array([1.0, 2.0]), "score_s": 0.1}
        recs = [{"stream": s, "i": i, "x_sha": "ab" * 32, "reimage_s": 0.1, **stat}
                for s, i in sorted(rl.db.planned_ids(arm, plan["null"]))]
        if edit:
            edit(arm, recs)
        return {"arm": arm, "records": recs, "setup": {"prep_s": 1.0, "adjacency_s": 0.0}, "warm_s": None,
                "studies_wall_s": 1.0}
    return run_arm


def only_one_null(arm, recs):  # the review's probe: one matching null record per arm for 8 requested per stream
    recs[:] = [r for r in recs if (r["stream"], r["i"]) == ("null", 0)]


def duplicated(arm, recs):
    recs[-1] = dict(recs[0])


def bad_hash(arm, recs):
    recs[0]["x_sha"] = "x"


def no_statistic(arm, recs):
    recs[0].pop("p" if arm == "lab" else "z")


def infinite_z(arm, recs):
    if arm == "fused":
        recs[0]["z"] = np.array([np.inf])


@pytest.mark.parametrize("edit", [only_one_null, duplicated, bad_hash, no_statistic, infinite_z])
def test_incomplete_or_malformed_records_are_refused_before_any_timing_or_parity(monkeypatch, capsys, edit):
    from core import truth
    serial, reference = truth.neighbours, truth.truth
    monkeypatch.setattr(rl.db, "run_arm", stub_arm(edit))
    monkeypatch.setattr("core.cohorts.load", lambda cohort: invented())
    assert rl.main(["--studies", "8", "--workers", "1"]) == 1
    out = capsys.readouterr()
    assert "refused" in out.err and "parity" not in out.out
    assert truth.neighbours is serial and truth.truth is reference  # restored after the refusal too


def test_a_complete_stubbed_race_passes_the_same_gate(monkeypatch):
    monkeypatch.setattr(rl.db, "run_arm", stub_arm())
    r = rl.race(invented(), studies=8, workers=1)
    assert r["evaluated"] == {"lab": 16, "fused": 24} and r["parity"] == {"x_bitwise": True, "x_compared": 16}


def test_the_report_names_the_cpu_and_does_not_claim_a_threshold():
    r = rl.race(invented(), studies=1, workers=1)
    assert r["host"]["cpu_model"] and r["host"]["usable_cpus"] >= 1 and r["host"]["workers"] == 1
    text = rl.report(r)
    assert "CPU: " in text and "threshold and calls not computed" in text and "one calibrated threshold" not in text
