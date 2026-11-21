import sys
from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from atlas.artifacts import (
    array_hash,
    charts,
    comparison_charts,
    save_point,
    save_tables,
    source_metadata,
)
from atlas.bank import bank_batch, build_bank
from atlas.runner import (
    calibrate,
    fit_omnibus,
    interval,
    omnibus_p,
    paired_interval,
    run_point,
    studies,
)
from atlas.simulation import fit_model


@pytest.fixture
def invented(monkeypatch):
    x = np.arange(4.0)
    reference = np.column_stack([x, -x, np.zeros((4, 4))])
    cells = {"time": np.full(4, 5.0), "event": np.array([1, 0, 0, 0]), "strata": np.zeros(4, int),
             "cell_type": np.zeros(8, np.int8), "x": np.tile([0.0, 1.0], 4), "y": np.zeros(8),
             "region_id": np.repeat(np.arange(4), 2), "patient_of_region": np.arange(4), "type_names": ["Tumor"]}

    def reimage(design, seeds):
        if design["patients"] is None:
            return np.stack([reference for _ in seeds])
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
        assert len({row[k] for k in ("image_seed", "survival_seed", "censor_seed", "permutation_seed")}) == 4
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


def test_paired_difference_bounds_cover_an_enumerable_distribution():
    probabilities = {-1: 0.05, 0: 0.85, 1: 0.10}
    coverage = 0.0
    for values in product((-1, 0, 1), repeat=4):
        lo, hi = paired_interval(np.array(values))
        if lo <= 0.05 <= hi:
            coverage += np.prod([probabilities[v] for v in values])
    assert coverage >= 0.95
    lo, hi = paired_interval(np.zeros(20))
    assert lo < 0 < hi  # Zero observed disagreements does not mean zero MC uncertainty.


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
    for bad in ({"regions_per_patient": 0}, {"fov_um": -1}, {"eps": 2}):
        with pytest.raises(ValueError):
            run_point(cells, reference, {**design, **bad})


def test_bank_samples_a_separate_acquisition_for_each_patient_row():
    X = np.zeros((4, 1, 3))
    X[:, 0, 0] = np.arange(4)
    X[:, 0, 1] = 2 * np.arange(4)
    X[:, 0, 2] = np.nan
    sampled = bank_batch({"X": X}, [2**63 + 19], 200)[0]
    assert len(np.unique(sampled[:, 0])) == 4
    assert_allclose(sampled[:, 1], 2 * sampled[:, 0])
    assert np.isnan(sampled[:, 2]).all()


def test_cached_bank_preserves_noiseless_law_and_direct_paired_checks(invented, tmp_path):
    cells, reference = invented
    design = {"cohort": "invented", "patients": 12, "regions_per_patient": 1, "fov_um": None, "eps": 0.0}
    bank = build_bank(cells, design, 4, 71, tmp_path, batch_size=2)
    assert not bank["cached"] and bank["X"].shape == (4, 4, 6)
    assert_allclose(bank["X"], np.broadcast_to(reference, (4, 4, 6)))
    cached = build_bank(cells, design, 4, 71, tmp_path)
    assert cached["cached"] and cached["hash"] == bank["hash"]
    budget = {"calibration": 8, "null": 4, "alternative": 4, "nested": 0, "direct": 2, "batch": 2}
    summary, rows, _ = run_point(cells, reference, {**design, "acquisition_bank": bank}, budget=budget)
    assert summary["measurement_model"] == "finite-bank approximation"
    assert summary["bank_draws"] == 4 and summary["bank_hash"] == bank["hash"]
    for stage in ("null", "alternative"):
        assert summary[stage + "_direct_rejection_n"] == 2
        assert summary[stage + "_bank_minus_direct_rejection"] == 0
    for row in rows:
        if "direct_maxT" in row:
            assert row["direct_maxT"] == row["maxT"]
    with pytest.raises(ValueError, match="different source"):
        run_point(cells, reference, {**design, "acquisition_bank": {**bank, "source_hash": "wrong"}}, budget=budget)
    for changed in ({"eps": 0.25}, {"fov_um": 1}, {"regions_per_patient": 2}, {"cohort": "different"}):
        with pytest.raises(ValueError, match="imaging design"):
            run_point(cells, reference, {**design, **changed, "acquisition_bank": bank}, budget=budget)


def test_public_calibration_can_be_reused_with_implicit_defaults(invented):
    cells, reference = invented
    design = {"cohort": "invented", "patients": 12, "regions_per_patient": 1}
    budget = {"calibration": 8, "null": 4, "alternative": 4, "nested": 0, "direct": 0, "batch": 2}
    calibration = calibrate(cells, reference, design, budget, 71)
    summary, _, reused = run_point(cells, reference, design, budget=budget, seed=71, calibration=calibration)
    assert reused is calibration and summary["design_id"] == calibration["design_id"]


def test_calibration_rejects_changed_bank_contents_even_with_same_recipe(invented, tmp_path):
    cells, reference = invented
    design = {"cohort": "invented", "patients": 12, "regions_per_patient": 1, "eps": 0.0}
    bank = build_bank(cells, design, 4, 71, tmp_path)
    design["acquisition_bank"] = bank
    budget = {"calibration": 8, "null": 4, "alternative": 4, "nested": 0, "direct": 0, "batch": 2}
    calibration = calibrate(cells, reference, design, budget, 71)
    changed = bank["X"].copy()
    changed[0, 0, 0] += 1
    design["acquisition_bank"] = {**bank, "X": changed}
    with pytest.raises(ValueError, match="content hash"):
        run_point(cells, reference, design, budget=budget, seed=71, calibration=calibration)
    design["acquisition_bank"]["hash"] = array_hash(changed)
    with pytest.raises(ValueError, match="calibration does not match"):
        run_point(cells, reference, design, budget=budget, seed=71, calibration=calibration)


def test_bank_cache_binds_generating_backend_and_code(invented, tmp_path, monkeypatch):
    import atlas.bank as module

    cells, _ = invented
    design = {"cohort": "invented", "patients": 12, "regions_per_patient": 1, "eps": 0.0}
    numpy = build_bank(cells, design, 4, 71, tmp_path)
    cpu = build_bank(cells, {**design, "backend": "cpu"}, 4, 71, tmp_path)
    assert numpy["key"] != cpu["key"] and not cpu["cached"]
    assert cpu["generator"]["backend"] == "cpu"
    monkeypatch.setattr(module, "generator_provenance", lambda *args: {"code_fingerprint": "changed"})
    changed = build_bank(cells, design, 4, 71, tmp_path)
    assert changed["key"] not in (numpy["key"], cpu["key"]) and not changed["cached"]


def test_reference_cache_rebuilds_without_matching_generator_provenance(invented, tmp_path, monkeypatch):
    import atlas.__main__ as cli
    from core.grammar import grammar, grammar_hash

    cells, reference = invented
    metadata = source_metadata(cells)
    path = tmp_path / "reference.npz"
    np.savez(path, X=np.zeros_like(reference), source_hash=metadata["source_hash"],
             grammar_hash=grammar_hash(grammar(cells["type_names"])))
    calls = []

    def fresh(_):
        calls.append(1)
        return reference

    monkeypatch.setattr(cli, "truth", fresh)
    X, _ = cli.reference_matrix(cells, path, metadata)
    assert_allclose(X, reference)
    cli.reference_matrix(cells, path, metadata)
    assert len(calls) == 1
    monkeypatch.setattr(cli, "generator_provenance", lambda *args: {"code_fingerprint": "changed"})
    cli.reference_matrix(cells, path, metadata)
    assert len(calls) == 2


def test_partitioned_evidence_retains_nullable_checks_and_uint64_seeds(invented, tmp_path):
    cells, reference = invented
    budget = {"calibration": 8, "null": 4, "alternative": 4, "nested": 0, "direct": 0, "batch": 2}
    design = {"patients": 12, "regions_per_patient": 1}
    summary, rows, _ = run_point(cells, reference, design, budget=budget)
    rows[0]["image_seed"] = 2**64 - 1
    save_point(summary, rows, tmp_path)
    second = {**summary, "design_id": "second"}
    rows[0]["conditional_p"] = 0.01
    rows[0]["conditional_reject"] = True
    save_point(second, rows, tmp_path)
    evidence = pd.read_parquet(tmp_path / "studies")
    assert len(evidence) == 32
    assert evidence.image_seed.max() == 2**64 - 1
    assert evidence.conditional_p.notna().sum() == 1
    assert evidence.conditional_reject.sum() == 1


def test_grouped_cli_checkpoints_and_resumes_without_repeating_studies(invented, tmp_path, monkeypatch):
    import atlas.__main__ as cli

    cells, reference = invented
    monkeypatch.setattr(cli, "load", lambda _: cells)
    monkeypatch.setattr(cli, "reference_matrix", lambda *args: (reference, "toy-grammar"))
    monkeypatch.setattr(cli, "charts", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "comparison_charts", lambda *args, **kwargs: None)
    args = ["--cohorts", "charville", "--patients", "12", "16", "--regions", "1",
            "--fov-um", "1", "full", "--backend", "numpy", "--workers", "2",
            "--eps", "0.1", "0.1",
            "--bank-draws", "4", "--calibration", "8", "--null", "4", "--alternative", "4",
            "--nested", "2", "--direct", "2", "--permutations", "9", "--batch", "2",
            "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache")]
    cli.main(args)
    out = tmp_path / "out" / "charville"
    before = pd.read_parquet(out / "summary.parquet")
    assert len(before) == 4 and len(list((out / "studies").glob("*.parquet"))) == 4
    assert len(before[before.null_nested_disagreement_n.notna()]) == 3
    real_runner = cli.run_point
    calls = []

    def record_run(*args, **kwargs):
        calls.append(1)
        return real_runner(*args, **kwargs)

    next((out / "studies").glob("*.parquet")).unlink()
    monkeypatch.setattr(cli, "run_point", record_run)
    cli.main(args)
    assert len(calls) == 1 and len(list((out / "studies").glob("*.parquet"))) == 4
    before = pd.read_parquet(out / "summary.parquet")
    monkeypatch.setattr(cli, "run_point", lambda *args, **kwargs: pytest.fail("completed point repeated"))
    cli.main(args)
    pd.testing.assert_frame_equal(before, pd.read_parquet(out / "summary.parquet"))
    with pytest.raises(ValueError, match="different run"):
        cli.main([*args, "--hr", "3"])


def test_standalone_scientific_figures_include_all_region_case(invented, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    cells, reference = invented
    design = {"cohort": "invented", "patients": 12, "regions_per_patient": 1, "fov_um": None, "eps": 0.0}
    budget = {"calibration": 8, "null": 4, "alternative": 4, "nested": 2, "permutations": 9, "batch": 2}
    summary, _, _ = run_point(cells, reference, design, budget=budget)
    rows = [summary, {**summary, "regions_per_patient": None}]
    charts(rows, tmp_path, cohort="invented", eps=0.0)
    comparison_charts(rows, tmp_path, cohort="invented")
    for name in ("power", "calibration", "method-checks"):
        assert (tmp_path / f"invented-{name}.png").read_bytes().startswith(b"\x89PNG")
        assert "<svg" in (tmp_path / f"invented-{name}.svg").read_text()
