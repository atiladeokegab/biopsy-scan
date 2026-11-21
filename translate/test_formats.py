"""Acceptance tests for Translate Step 2 (Req: A-17.2): arcsinh-GMM, single-slide, serial 0-100um, tiles."""
from __future__ import annotations

import numpy as np
import pytest

from translate import fit_arcsinh_gmm, run_translate, simulate_formats


def _make_proximity_cohort(*, n_patients: int = 36, seed: int = 99) -> dict:
    """Create a proximity cohort matching R-16 ['Other', 'Tumor (Podo+)', 'Stromal / Fibroblast']."""
    rng = np.random.default_rng(seed)
    xs, ys, types, markers_list, rids = [], [], [], [], []
    risk = np.zeros(n_patients, dtype=np.float64)

    for p in range(n_patients):
        high_risk = p < (n_patients // 2)
        risk[p] = 1.0 if high_risk else -1.0
        # 12 Tumor (Podo+) cells at (100, 100)
        tx = np.full(12, 100.0) + rng.normal(0.0, 2.0, 12)
        ty = np.full(12, 100.0) + rng.normal(0.0, 2.0, 12)
        if high_risk:
            # High-risk: 18 Stromal/Fibroblast cells inside 25um of tumour + 18 Other far away at (250, 250).
            fx = rng.normal(100.0, 4.0, 18)
            fy = rng.normal(100.0, 4.0, 18)
            ox = rng.normal(250.0, 6.0, 18)
            oy = rng.normal(250.0, 6.0, 18)
            ct = np.concatenate([np.full(12, 1, dtype=np.int16), np.full(18, 2, dtype=np.int16), np.full(18, 0, dtype=np.int16)])
            xs.append(np.concatenate([tx, fx, ox]))
            ys.append(np.concatenate([ty, fy, oy]))
        else:
            # Low-risk: 18 Other inside 25um of tumour + 18 Stromal/Fibroblast far away at (250, 250).
            ox = rng.normal(100.0, 4.0, 18)
            oy = rng.normal(100.0, 4.0, 18)
            fx = rng.normal(250.0, 6.0, 18)
            fy = rng.normal(250.0, 6.0, 18)
            ct = np.concatenate([np.full(12, 1, dtype=np.int16), np.full(18, 0, dtype=np.int16), np.full(18, 2, dtype=np.int16)])
            xs.append(np.concatenate([tx, ox, fx]))
            ys.append(np.concatenate([ty, oy, fy]))

        types.append(ct)
        mk = rng.normal(-0.5, 0.2, size=(48, 3))
        mk[ct == 1, 0] = rng.normal(4.0, 0.3, int((ct == 1).sum()))  # Podoplanin+ on Tumor(Podo+)
        mk[ct == 2, 1] = rng.normal(4.5, 0.3, int((ct == 2).sum()))  # aSMA+ on Stromal/Fibroblast
        mk[:, 2] = rng.normal(0.0, 0.2, 48)
        markers_list.append(mk)
        rids.append(np.full(48, p, dtype=np.int32))

    time = np.where(risk > 0, rng.uniform(30, 110, n_patients), rng.uniform(220, 420, n_patients))
    event = np.ones(n_patients, dtype=bool)

    return {
        "cohort": "synthetic",
        "split": "discovery",
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "cell_type": np.concatenate(types),
        "markers": np.concatenate(markers_list, axis=0),
        "marker_names": ["Podoplanin", "aSMA", "Noise"],
        "type_names": ["Other", "Tumor (Podo+)", "Stromal / Fibroblast"],
        "region_id": np.concatenate(rids),
        "patient_of_region": np.arange(n_patients, dtype=np.int32),
        "time": time,
        "event": event,
        "strata": np.array([i % 2 for i in range(n_patients)], dtype=np.int32),
        "region_names": [f"R{i}" for i in range(n_patients)],
        "patient_names": [f"P{i}" for i in range(n_patients)],
    }


def test_arcsinh_gmm_separates_bimodal_and_negative_centered_intensities():
    rng = np.random.default_rng(1)
    train_vals = np.concatenate([rng.normal(-0.5, 0.2, 300), rng.normal(4.0, 0.4, 150)])
    gate = fit_arcsinh_gmm(train_vals)
    assert gate.cofactor > 0.0
    low_calls = gate.predict_positive(np.array([-1.2, -0.5, 0.0]))
    high_calls = gate.predict_positive(np.array([3.5, 4.5, 8.0]))
    assert not low_calls.any()
    assert high_calls.all()


def test_zero_misregistration_equals_single_slide_and_formats_report_paired_cis():
    data = _make_proximity_cohort(n_patients=36, seed=99)
    out = run_translate(
        data,
        out_path=None,
        n_outer=4,
        n_inner=2,
        n_boot=120,
        n_draws=6,
        misreg_grid=(0.0, 25.0, 50.0, 75.0, 100.0),
        tile_grid=(100.0, 150.0, 200.0),
        seed=0,
    )
    assert out["status"] == "MEASURED"
    s2 = out["step2_formats"]
    single = s2["single_slide"]
    serial0 = s2["serial_section"]["0"]
    assert single["status"] == "MEASURED"
    assert serial0["c_panel"] == single["c_panel"]
    assert serial0["retained_c_index"] == single["retained_c_index"]
    assert serial0["ci_low"] == single["ci_low"]
    assert serial0["ci_high"] == single["ci_high"]
    assert single["c_panel"] > 0.7 and single["retained_c_index"] > 0.6

    for d_key in ("0", "25", "50", "75", "100"):
        row = s2["serial_section"][d_key]
        assert row["status"] == "MEASURED"
        if row["ci_status"] == "MEASURED":
            assert row["ci_low"] <= row["retained_c_index"] <= row["ci_high"]
        if d_key != "0":
            assert row["eval_mode"] == "per_draw_expected"
            assert row["n_draws"] == 6
    # Large 100um displacement disrupts the 25um adjacency relative to aligned single slide.
    assert s2["serial_section"]["100"]["c_panel"] <= single["c_panel"] + 0.05

    for t_key in ("100", "150", "200"):
        row = s2["tiles"][t_key]
        assert row["status"] == "MEASURED"
        if row["ci_status"] == "MEASURED":
            assert row["ci_low"] <= row["retained_c_index"] <= row["ci_high"]


def test_step2_c_full_below_chance_returns_unmeasured():
    data = _make_proximity_cohort(n_patients=24, seed=12)
    # Zero R-16 signal via identical survival / constant radius=250um so C_full == 0.5.
    data["time"] = np.arange(1, 25, dtype=np.float64) * 10.0
    data["event"] = np.array([i % 2 == 0 for i in range(24)], dtype=bool)
    s2 = simulate_formats(
        data,
        panel=(0, 1),
        n_outer=3,
        n_boot=50,
        seed=0,
        r16_rule={
            "feature": "Tumor (Podo+)|Stromal / Fibroblast|500um|fraction",
            "focal": "Tumor (Podo+)",
            "neighbour": "Stromal / Fibroblast",
            "radius_um": 500,
            "statistic": "fraction",
            "expressible": True,
        },
    )
    assert s2["status"] == "UNMEASURED"
    assert s2["single_slide"]["status"] == "UNMEASURED"
    assert all(v["status"] == "UNMEASURED" for v in s2["serial_section"].values())
    assert all(v["status"] == "UNMEASURED" for v in s2["tiles"].values())
