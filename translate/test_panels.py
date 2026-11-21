"""Acceptance tests for Translate Step 1 (Req: A-17.1): exhaustive 1- and 2-marker panels in nested CV."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from translate import LockedSplitError, enumerate_panels, run_panel_cv, run_translate


def _make_spatial_cohort(
    *,
    n_patients: int = 40,
    m_markers: int = 8,
    signal: bool = True,
    constant_geometry: bool = False,
    seed: int = 42,
) -> dict:
    """Build synthetic CODEX tissue with R-16 cell types ['Other', 'Tumor (Podo+)', 'Stromal / Fibroblast']."""
    rng = np.random.default_rng(seed)
    n_per_reg = 40
    n_reg = n_patients
    xs, ys, types, markers_list, rids = [], [], [], [], []
    risk_true = np.zeros(n_patients, dtype=np.float64)
    b_col = min(2, m_markers - 1)

    for p in range(n_patients):
        high_risk = p < (n_patients // 2) if signal else bool(rng.integers(0, 2))
        risk_true[p] = 1.0 if high_risk else -1.0
        if constant_geometry:
            # Exact identical cell positions and types per patient -> zero R-16 variance -> C_full == 0.5.
            tx = np.linspace(48.0, 52.0, 10)
            ty = np.full(10, 50.0)
            fx = np.linspace(45.0, 55.0, 20)
            fy = np.full(20, 54.0)
            ox = np.linspace(135.0, 145.0, 10)
            oy = np.full(10, 140.0)
            ct = np.concatenate([
                np.full(10, 1, dtype=np.int16),
                np.full(20, 2, dtype=np.int16),
                np.full(10, 0, dtype=np.int16),
            ])
            xs.append(np.concatenate([tx, fx, ox]))
            ys.append(np.concatenate([ty, fy, oy]))
        elif high_risk:
            tx = rng.normal(50.0, 2.5, 10)
            ty = rng.normal(50.0, 2.5, 10)
            # High-risk: 15 Stromal/Fibroblast cells within 25um of Tumor(Podo+), 15 Other far away.
            fx = rng.normal(50.0, 4.0, 15)
            fy = rng.normal(50.0, 4.0, 15)
            ox = rng.normal(140.0, 4.0, 15)
            oy = rng.normal(140.0, 4.0, 15)
            ct = np.concatenate([
                np.full(10, 1, dtype=np.int16),  # Tumor (Podo+)
                np.full(15, 2, dtype=np.int16),  # Stromal / Fibroblast
                np.full(15, 0, dtype=np.int16),  # Other
            ])
            xs.append(np.concatenate([tx, fx, ox]))
            ys.append(np.concatenate([ty, fy, oy]))
        else:
            tx = rng.normal(50.0, 2.5, 10)
            ty = rng.normal(50.0, 2.5, 10)
            # Low-risk: equal totals (15 Fibroblast, 15 Other), but Other near tumour and Fibroblasts >100um away.
            ox = rng.normal(50.0, 4.0, 15)
            oy = rng.normal(50.0, 4.0, 15)
            fx = rng.normal(140.0, 4.0, 15)
            fy = rng.normal(140.0, 4.0, 15)
            ct = np.concatenate([
                np.full(10, 1, dtype=np.int16),  # Tumor (Podo+)
                np.full(15, 0, dtype=np.int16),  # Other
                np.full(15, 2, dtype=np.int16),  # Stromal / Fibroblast
            ])
            xs.append(np.concatenate([tx, ox, fx]))
            ys.append(np.concatenate([ty, oy, fy]))
        types.append(ct)

        mk = rng.normal(-0.4, 0.25, size=(n_per_reg, m_markers))
        if not constant_geometry:
            mk[ct == 1, 1] = rng.normal(3.6, 0.25, int((ct == 1).sum()))
            mk[ct == 2, b_col] = rng.normal(3.9, 0.25, int((ct == 2).sum()))
        markers_list.append(mk)
        rids.append(np.full(n_per_reg, p, dtype=np.int32))

    if signal and not constant_geometry:
        time = np.where(risk_true > 0, rng.uniform(40, 120, n_patients), rng.uniform(240, 450, n_patients))
        event = np.ones(n_patients, dtype=bool)
    else:
        time = np.arange(1, n_patients + 1, dtype=np.float64) * 10.0
        event = np.array([i % 2 == 0 for i in range(n_patients)], dtype=bool)

    return {
        "cohort": "synthetic",
        "split": "discovery",
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "cell_type": np.concatenate(types),
        "markers": np.concatenate(markers_list, axis=0),
        "marker_names": [f"M{i}" for i in range(m_markers)],
        "type_names": ["Other", "Tumor (Podo+)", "Stromal / Fibroblast"],
        "region_id": np.concatenate(rids),
        "patient_of_region": np.arange(n_reg, dtype=np.int32),
        "time": time,
        "event": event,
        "strata": np.array([i % 2 for i in range(n_patients)], dtype=np.int32),
        "region_names": [f"R{i}" for i in range(n_reg)],
        "patient_names": [f"P{i}" for i in range(n_patients)],
    }


def test_panel_enumeration_820_at_m40_and_253_at_m22():
    p40 = enumerate_panels([f"M{i}" for i in range(40)])
    p22 = enumerate_panels([f"M{i}" for i in range(22)])
    assert len(p40) == 40 + 40 * 39 // 2 == 820
    assert len(p22) == 22 + 22 * 21 // 2 == 253
    assert len(set(p40)) == 820


def test_step1_m40_820_panels_positive_signal_paired_ci_and_hash():
    data = _make_spatial_cohort(n_patients=48, m_markers=40, signal=True, seed=7)
    res1 = run_panel_cv(data, n_outer=4, n_inner=3, n_boot=150, seed=0)
    res2 = run_panel_cv(data, n_outer=4, n_inner=3, n_boot=150, seed=0)
    assert res1["status"] == "MEASURED"
    assert res1["m"] == 40 and res1["n_panels"] == 820 and len(res1["panels"]) == 820
    assert res1["c_full_oof"] > 0.7
    assert res1["panel_model_hash"] == res2["panel_model_hash"]
    n2 = res1["nested_selected_2marker"]
    assert n2["status"] == "MEASURED"
    assert n2["c_full"] == pytest.approx(res1["c_full_oof"])
    assert n2["ci_low"] <= n2["retained_c_index"] <= n2["ci_high"]
    assert n2["retained_c_index"] > 0.7


def test_c_full_at_or_below_chance_returns_unmeasured():
    # Zero spatial & marker variance -> identical predicted R-16 features -> beta == 0 -> C_full == 0.5 -> UNMEASURED.
    data = _make_spatial_cohort(n_patients=24, m_markers=6, signal=False, constant_geometry=True, seed=3)
    data["markers"][:] = 0.0
    out = run_translate(data, out_path=None, n_outer=3, n_inner=2, n_boot=50, seed=0)
    assert out["status"] == "UNMEASURED"
    assert out["step1_panels"]["status"] == "UNMEASURED"
    assert out["step1_panels"]["nested_selected_2marker"]["status"] == "UNMEASURED"
    assert out["step1_panels"]["nested_selected_2marker"]["retained_c_index"] is None
    assert "0.5" in (out["step1_panels"]["nested_selected_2marker"]["reason"] or "")
    assert out["step2_formats"]["status"] == "UNMEASURED"
    assert out["step2_formats"]["single_slide"]["status"] == "UNMEASURED"


def test_unsupported_r16_makes_every_r17_result_unmeasured():
    data = _make_spatial_cohort(n_patients=24, m_markers=5, signal=True, seed=5)
    reason = "UPMC cannot express Slug-defined pEMT in this test"
    out = run_translate(
        data,
        out_path=None,
        r16_rule={"expressible": False, "reason": reason},
        n_outer=3,
        n_inner=2,
        n_boot=50,
    )
    assert out["status"] == "UNMEASURED"
    assert out["reason"] == reason
    assert out["step1_panels"]["status"] == "UNMEASURED"
    assert all(p["status"] == "UNMEASURED" and p["reason"] == reason for p in out["step1_panels"]["panels"])
    assert out["step2_formats"]["single_slide"]["status"] == "UNMEASURED"
    assert out["step2_formats"]["single_slide"]["reason"] == reason


def test_locked_split_or_patients_always_rejected_and_zero_test_leakage():
    data = _make_spatial_cohort(n_patients=24, m_markers=5, signal=True, seed=11)
    with pytest.raises(LockedSplitError):
        run_translate(data, out_path=None, split="locked")
    with pytest.raises(LockedSplitError):
        run_translate(data, out_path=None, split="all")
    bad_data = {**data, "locked_patients": ["P0"]}
    with pytest.raises(LockedSplitError):
        run_translate(bad_data, out_path=None, split="discovery")

    # Verify outer-fold training models stay bit-identical when outer test patients are altered.
    from translate._stats import stratified_kfolds
    from translate.formats import fit_arcsinh_gmm

    folds = stratified_kfolds(data["event"], data["strata"], n_splits=4, seed=0)
    tr_idx, te_idx = folds[0]
    tr_cells = np.isin(data["patient_of_region"][data["region_id"]], tr_idx)
    te_cells = np.isin(data["patient_of_region"][data["region_id"]], te_idx)
    g1 = fit_arcsinh_gmm(data["markers"][tr_cells, 1]).to_dict()
    mutated = data["markers"].copy()
    mutated[te_cells, 1] += 1000.0
    g2 = fit_arcsinh_gmm(mutated[tr_cells, 1]).to_dict()
    assert g1 == g2


def test_committed_upmc_translate_artifact_contract():
    artifact = Path(__file__).resolve().parents[1] / "results" / "upmc-translate.json"
    saved = json.loads(artifact.read_text())
    assert saved["cohort"] == "upmc"
    assert saved["split"] == "discovery"
    assert saved["locked_patients_used"] is False
    assert saved["clinical_claims"] is False
    assert set(saved["labels"].values()) <= {"MEASURED", "UNMEASURED"}


def test_r16_rule_change_actually_changes_reference_and_hash():
    data = _make_spatial_cohort(n_patients=30, m_markers=5, signal=True, seed=21)
    r_25 = run_panel_cv(data, n_outer=3, n_inner=2, n_boot=40, seed=0)
    r_250 = run_panel_cv(
        data,
        n_outer=3,
        n_inner=2,
        n_boot=40,
        seed=0,
        r16_rule={
            "feature": "Tumor (Podo+)|Stromal / Fibroblast|250um|fraction",
            "focal": "Tumor (Podo+)",
            "neighbour": "Stromal / Fibroblast",
            "radius_um": 250,
            "statistic": "fraction",
            "expressible": True,
        },
    )
    assert r_25["status"] == "MEASURED"
    assert r_25["r16_rule_sha256"] != r_250["r16_rule_sha256"]
    assert r_25["c_full_oof"] > 0.72
    # At radius=250um, both high- and low-risk tumour cells see all fibroblasts, erasing contrast.
    assert r_250["status"] == "UNMEASURED" or (r_250["c_full_oof"] or 0.5) < r_25["c_full_oof"] - 0.15


def test_public_entrypoints_fail_closed_on_split_real_provenance_and_bad_indices(monkeypatch, tmp_path):
    from translate import formats, runner
    from core import cohorts

    data = _make_spatial_cohort(n_patients=24, m_markers=4, signal=True, seed=13)
    with pytest.raises(LockedSplitError):
        run_panel_cv({**data, "split": "locked"})
    with pytest.raises(LockedSplitError):
        run_panel_cv(data, split="all")
    with pytest.raises(LockedSplitError):
        formats.simulate_formats({**data, "split": "all"})
    # UPMC-tagged payload without verified lock provenance must fail closed.
    with pytest.raises(LockedSplitError):
        run_panel_cv({**data, "cohort": "upmc"})

    # Missing required markers must report not testable cleanly (never dropped silently).
    missing_res = run_panel_cv(
        {**data, "cohort": "upmc", "lock_verified": True, "locked_regions_count": 10},
        r16_rule={
            "feature": "Tumor (Podo+)|Stromal / Fibroblast|25um|fraction",
            "focal": "Tumor (Podo+)",
            "neighbour": "Stromal / Fibroblast",
            "required_markers": ["PD1", "PDL1", "GranzymeB"],
            "radius_um": 25,
            "statistic": "fraction",
            "expressible": True,
        },
    )
    assert missing_res["status"] == "UNMEASURED"
    assert "not testable" in missing_res["reason"] and "PD1" in missing_res["reason"]

    # Negative/float indices must raise ValueError instead of silently wrapping.
    for key in ("region_id", "patient_of_region", "cell_type"):
        bad = {**data, key: data[key].copy()}
        bad[key][0] = -1
        with pytest.raises(ValueError):
            run_panel_cv(bad)
        with pytest.raises(ValueError):
            formats.simulate_formats(bad)

    # _load_upmc_discovery_markers must fail closed if LOCK_FILE is None (when R-13 API exists).
    monkeypatch.setattr(cohorts, "LOCK_FILE", None, raising=False)
    monkeypatch.setattr(cohorts, "locked_regions", lambda *a, **k: {"REG1"}, raising=False)
    orig_load = cohorts.load_regions
    monkeypatch.setattr(cohorts, "load_regions", lambda cohort, data_dir=None, identity="case", split="discovery": orig_load(cohort, data_dir, identity), raising=False)
    with pytest.raises(LockedSplitError):
        runner._load_upmc_discovery_markers(tmp_path)


def test_within_fold_c_index_offset_invariance_cox_grad_and_honest_ratio_ci():
    from scipy.optimize import check_grad
    from translate._stats import cox_neg_loglik_and_grad, harrell_c_index, paired_retained_c_index

    # 1. Within-(fold, stratum) C-index invariance under arbitrary per-fold offsets.
    time = np.array([10.0, 20.0, 30.0, 15.0, 25.0, 35.0])
    event = np.ones(6, dtype=bool)
    folds = np.array([0, 0, 0, 1, 1, 1])
    strata = np.zeros(6, dtype=np.int64)
    risk = np.array([3.0, 2.0, 1.0, 3.0, 2.0, 1.0])
    c_base = harrell_c_index(time, event, risk, strata=strata, folds=folds)
    c_shifted = harrell_c_index(time, event, risk + np.array([0, 0, 0, 500.0, 500.0, 500.0]), strata=strata, folds=folds)
    assert c_base == pytest.approx(1.0)
    assert c_shifted == pytest.approx(c_base)

    # 2. Exact Cox gradient at large |eta| (unclipped log-sum-exp consistency).
    Zs = np.array([[4.0], [1.0], [-3.0]])
    grp_ends = np.array([0, 1, 2])
    d_k = np.array([1.0, 1.0, 1.0])
    s_k = Zs.copy()
    groups = [(Zs, grp_ends, d_k, s_k)]
    err = check_grad(
        lambda b: cox_neg_loglik_and_grad(b, groups, 1.0 / 3.0, 0.05)[0],
        lambda b: cox_neg_loglik_and_grad(b, groups, 1.0 / 3.0, 0.05)[1],
        np.array([15.0]),
    )
    assert err < 1e-6

    # 3. Unconditional bootstrap marks ratio CI UNMEASURED when C_full <= 0.5 occurs in any draw.
    t8 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    e8 = np.ones(8, dtype=bool)
    rf_weak = np.array([0.8, 0.1, 0.6, 0.2, 0.7, 0.3, 0.5, 0.4])
    rp_same = rf_weak.copy()
    stats_res = paired_retained_c_index(t8, e8, rp_same, rf_weak, n_boot=100, seed=0)
    assert stats_res["n_boot_nonpositive"] > 0
    assert stats_res["ci_status"] == "UNMEASURED"
    assert stats_res["ci_low"] is None and stats_res["ci_high"] is None
