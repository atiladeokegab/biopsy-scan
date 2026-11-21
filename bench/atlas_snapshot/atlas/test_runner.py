import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from atlas.artifacts import save_tables, source_metadata
from atlas.runner import fit_omnibus, interval, omnibus_p, run_point, studies
from atlas.simulation import fit_model


@pytest.fixture
def invented(monkeypatch):
    x = np.arange(4.0)
    reference = np.column_stack([x, -x, np.zeros((4, 4))])
    cells = {"time": np.full(4, 5.0), "event": np.array([1, 0, 0, 0]), "strata": np.zeros(4, int),
             "cell_type": np.zeros(8, np.int8), "x": np.tile([0.0, 1.0], 4), "y": np.zeros(8),
             "region_id": np.repeat(np.arange(4), 2), "patient_of_region": np.arange(4), "type_names": ["Tumor"]}

    def reimage(design, seeds):
        return np.stack([reference[np.random.default_rng(int(seed)).integers(0, 4, design["patients"])] for seed in seeds])

    monkeypatch.setitem(sys.modules, "reimage", SimpleNamespace(run=reimage))
    return cells, reference


def test_patient_rows_match_shared_core_draws_and_stage_streams_are_disjoint(invented):
    from core.reimage_ref import draw

    cells, reference = invented
    model = fit_model(cells, reference)
    design = {"cohort": "invented", "patients": 12, "regions_per_patient": 1, "fov_um": None, "eps": 0.0}
    nulls = list(studies(cells, model, design, 5, 41, 202, 3))
    alternatives = list(studies(cells, model, design, 5, 41, 303, 2))
    assert {r["image_seed"] for r in nulls}.isdisjoint(r["image_seed"] for r in alternatives)
    for row in nulls:
        patients = draw(cells, design, row["image_seed"])["patients"]
        assert_allclose(row["X"], reference[patients])
        assert len(set(row[k] for k in ("image_seed", "survival_seed", "censor_seed", "permutation_seed"))) == 4
    other_batching = list(studies(cells, model, design, 5, 41, 202, 1))
    for a, b in zip(nulls, other_batching):
        assert_allclose(a["time"], b["time"])
        assert_allclose(a["z"], b["z"])


def test_power_identification_ties_and_held_out_null_evidence(invented):
    cells, reference = invented
    design = {"cohort": "invented", "patients": 80, "regions_per_patient": 1, "fov_um": None, "eps": 0.0}
    budget = {"calibration": 64, "null": 64, "alternative": 64, "nested": 0, "batch": 8}
    summary, rows, _ = run_point(cells, reference, design, hr=10, budget=budget, seed=31)
    assert summary["power"] > 0.7
    assert summary["fwer"] < 0.2
    assert summary["identification_rate"] == 0  # Two perfectly correlated top candidates.
    assert summary["discovery_tie_rate"] == summary["power"]
    assert summary["wrong_identification_rate"] == 0
    for name in ("power", "identification_rate", "fwer", "omnibus_fwer", "omnibus_power"):
        assert summary[name + "_n"] == 64
        assert summary[name + "_lo"] <= summary[name] <= summary[name + "_hi"]
    stages = {stage: {r["image_seed"] for r in rows if r["stage"] == stage}
              for stage in ("calibration", "null", "alternative")}
    assert stages["calibration"].isdisjoint(stages["null"] | stages["alternative"])
    assert stages["null"].isdisjoint(stages["alternative"])
    assert summary["region_um"] == 1 and summary["source_n"] == 4


def test_reusable_omnibus_is_superuniform_on_exchangeable_tiny_support():
    marginal = np.array([[0, 0], [1, -1], [2, 0]], float)
    support = np.array([[0.1, 1.0], [1.2, 0.2], [2.0, 1.0], [3.0, 0.5], [0.2, 0.1]])
    p = []
    for i, z in enumerate(support):
        calibration = fit_omnibus(np.vstack([marginal, np.delete(support, i, axis=0)]))
        p.append(omnibus_p(calibration, z)[0])
    for alpha in (0.2, 0.4, 0.6, 0.8):
        assert np.mean(np.array(p) <= alpha + 1e-14) <= alpha + 1e-14


def test_binomial_zero_and_all_success_intervals():
    assert_allclose(interval(0, 10), [0, 1 - 0.025 ** 0.1])
    assert_allclose(interval(10, 10), [0.025 ** 0.1, 1])
    with pytest.raises(ValueError):
        interval(0, 0)


def test_nested_comparison_and_parquet_provenance(invented, tmp_path):
    cells, reference = invented
    design = {"cohort": "invented", "patients": 12, "regions_per_patient": 1, "fov_um": None, "eps": 0.0}
    budget = {"calibration": 8, "null": 4, "alternative": 4, "nested": 2, "permutations": 9, "batch": 2}
    summary, rows, _ = run_point(cells, reference, design, budget=budget)
    assert summary["null_nested_disagreement_n"] == summary["alternative_nested_disagreement_n"] == 2
    assert len([r for r in rows if "conditional_p" in r]) == 4
    save_tables([summary], rows, source_metadata(cells), tmp_path)
    assert pd.read_parquet(tmp_path / "summary.parquet").iloc[0].source_hash == summary["source_hash"]
    trial = pd.read_parquet(tmp_path / "studies.parquet")
    assert len(trial) == 16 and trial.image_seed.is_unique


def test_budget_and_calibration_mismatch_fail_clearly(invented):
    cells, reference = invented
    design = {"cohort": "invented", "patients": 12, "regions_per_patient": 1, "eps": 0.0}
    for budget in ({"nested": -1}, {"nested": 0.5}, {"calibration": 3}, {"null": 0}, {"batch": True}):
        with pytest.raises(ValueError):
            run_point(cells, reference, design, budget=budget)
