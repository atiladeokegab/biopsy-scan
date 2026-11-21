"""Core grammar v2, channel loading/transform, and cKDTree truth checks (Req: A-12.1@1)."""
from __future__ import annotations

import zipfile

import numpy as np
import pytest

from core import cohorts
from core.grammar import (
    EXPRESSION_CHANNELS,
    EXPRESSION_TRANSFORM,
    feature_names,
    grammar,
    grammar_hash,
    grammar_v2,
    n_features,
)
from core.reimage_ref import reimage_v2
from core.truth import truth, truth_v2

TYPES = ["B cell", "Stroma", "Tumor", "Tumor (Ki67+)"]


def _write_charville_fixture(tmp_path, *, expr_rows_fn, feat_rows_fn):
    (tmp_path / "charville_labels.csv").write_text(
        "region_id,primary_outcome,recurrence,alive_or_deceased,type_of_first_recurrence,surgery_date,"
        "first_recurrence_date,last_contact_date,length_of_disease_free_survival,grade_differentiation,"
        "acquisition_id_visualizer,sample_label_visualizer\n"
        "Charville_c001_v001_r001_reg001,0,0,Dead,X,1/1/2010,NONE,1/11/2010,1,2,a,100_1\n"
        "Charville_c001_v001_r001_reg002,0,0,Alive,X,1/1/2010,NONE,1/21/2010,1,2,a,101_1\n"
    )
    expr_cols = [c for c in EXPRESSION_CHANNELS if c != "SIZE"]
    with zipfile.ZipFile(tmp_path / "charville_raw_data.zip", "w") as z:
        for idx, reg in enumerate(["Charville_c001_v001_r001_reg001", "Charville_c001_v001_r001_reg002"]):
            z.writestr(f"raw_data/{reg}.cell_data.csv", "CELL_ID,X,Y\n10,0,0\n20,50,0\n")
            z.writestr(f"raw_data/{reg}.cell_types.csv", "CELL_ID,CELL_TYPE\n20,Tumor\n10,Stroma\n")
            hdr = "CELL_ID," + ",".join(expr_cols)
            r_rows = expr_rows_fn(idx, expr_cols)
            z.writestr(f"raw_data/{reg}.expression.csv", hdr + "\n" + "\n".join(r_rows) + "\n")
            f_rows = feat_rows_fn(idx)
            z.writestr(f"raw_data/{reg}.cell_features.csv", "CELL_ID,SIZE\n" + "\n".join(f_rows) + "\n")


def test_grammar_v2_ordering_and_v1_frozen():
    g1 = grammar(TYPES)
    g2 = grammar_v2(TYPES)
    assert tuple(g2["channels"]) == EXPRESSION_CHANNELS and len(g2["channels"]) == 10
    assert g2["transform"] == EXPRESSION_TRANSFORM
    assert n_features(g1) == 48 and n_features(g2) == 48 + 2 * 4 * 3 * 10 == 288
    names = feature_names(g2)
    assert names[:48] == feature_names(g1)
    assert names[48] == "Tumor|B cell|25um|PD1"
    assert names[58] == "Tumor|B cell|50um|PD1"
    assert names[-1] == "Tumor (Ki67+)|Tumor (Ki67+)|100um|SIZE"
    assert grammar_hash(g1) != grammar_hash(g2)


def test_zero_neighbours_gets_zero_and_missing_focal_gets_nan():
    # Two patients: P0 has Tumor cell at origin with 1 Stroma neighbour at 10um; P1 has only Stroma (no Tumor).
    cells = {
        "cell_type": np.array([2, 1, 1], dtype=np.int8),
        "x": np.array([0.0, 10.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "channels": np.array([[1.0] * 10, [3.0] * 10, [5.0] * 10], dtype=np.float32),
        "region_id": np.array([0, 0, 1], dtype=np.int32),
        "patient_of_region": np.array([0, 1], dtype=np.int32),
        "type_names": TYPES,
    }
    x1 = truth(cells)
    x12 = truth_v2(cells)
    assert x12.shape == (2, 288)
    np.testing.assert_array_equal(x12[:, :48], x1)
    # P0 has Tumor (t=0): B cell neighbours=0 -> 0.0; Stroma neighbours at 25um=1 -> channel mean 3.0
    assert np.allclose(x12[0, 48:58], 0.0)
    assert np.allclose(x12[0, 48 + 30:48 + 40], 3.0)
    # P0 has no Tumor (Ki67+) focal cells -> NaN; P1 has zero Tumor focal cells -> all NaN
    assert np.isnan(x12[0, 24:48]).all() and np.isnan(x12[0, 48 + 120:]).all()
    assert np.isnan(x12[1]).all()
    r12 = reimage_v2(cells, {"patients": None, "regions_per_patient": None, "fov_um": None, "eps": 0.0}, seed=0)
    np.testing.assert_allclose(r12, x12, rtol=0, atol=1e-12, equal_nan=True)


def test_zero_variance_channel_becomes_zero_and_missing_value_fails_loudly(tmp_path):
    # First fixture: channel 0 (PD1) is constant 1.25 across all cells -> zero variance -> z = 0.0
    _write_charville_fixture(
        tmp_path,
        expr_rows_fn=lambda idx, cols: [
            "20," + ",".join(["1.25"] + [str(-0.5 + idx + j) for j in range(1, len(cols))]),
            "10," + ",".join(["1.25"] + [str(0.5 + idx + j) for j in range(1, len(cols))]),
        ],
        feat_rows_fn=lambda idx: [f"20,{0.2 + 0.1 * idx}", f"10,{0.8 + 0.1 * idx}"],
    )
    cells = cohorts.load("charville", data_dir=tmp_path, include_channels=True)
    assert np.array_equal(cells["channels"][:, 0], np.zeros(4, dtype=np.float32))
    assert np.allclose(cells["channels"][:, 1:].mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(cells["channels"][:, 1:].std(axis=0), 1.0, atol=1e-5)

    # Second fixture in fresh dir: NaN in PD1 must fail loudly
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    _write_charville_fixture(
        bad_dir,
        expr_rows_fn=lambda idx, cols: [
            "20," + ",".join(["" if idx == 0 else "0.5"] + ["1.0"] * (len(cols) - 1)),
            "10," + ",".join(["0.5"] * len(cols)),
        ],
        feat_rows_fn=lambda idx: ["20,0.4", "10,0.6"],
    )
    with pytest.raises(ValueError, match="non-finite"):
        cohorts.load("charville", data_dir=bad_dir, include_channels=True)


def test_channel_cache_validates_corrupted_npz_and_preserves_float64_precision(tmp_path):
    # Tiny differences separated only at float64 precision (collapse in float32) must normalize cleanly.
    _write_charville_fixture(
        tmp_path,
        expr_rows_fn=lambda idx, cols: [
            "20," + ",".join(["1.00000001"] + [str(0.5 + idx + j) for j in range(1, len(cols))]),
            "10," + ",".join(["1.00000002"] + [str(-0.5 + idx + j) for j in range(1, len(cols))]),
        ],
        feat_rows_fn=lambda idx: [f"20,{0.2 + 0.1 * idx}", f"10,{0.8 + 0.1 * idx}"],
    )
    cells = cohorts.load("charville", data_dir=tmp_path, include_channels=True)
    cache_path = tmp_path / "charville_channels.npz"
    cached = np.load(cache_path)
    assert cached["channels_raw"].dtype == np.float64
    assert np.allclose(cells["channels"][:, 0], [1.0, -1.0, 1.0, -1.0], atol=1e-5)

    # Misaligned cached region_ptr must fail loudly on subsequent load.
    bad_ptr_npz = dict(cached)
    cached.close()
    bad_ptr_npz["region_ptr"] = bad_ptr_npz["region_ptr"] + 1
    np.savez(cache_path, **bad_ptr_npz)
    with pytest.raises(ValueError, match="region pointers|align"):
        cohorts.load("charville", data_dir=tmp_path, include_channels=True)

    # Corrupted NaN inside an otherwise structurally aligned cache must fail loudly on cache hit.
    nan_npz = dict(bad_ptr_npz)
    nan_npz["region_ptr"] = nan_npz["region_ptr"] - 1
    bad_raw = nan_npz["channels_raw"].copy()
    bad_raw[0, 0] = np.nan
    nan_npz["channels_raw"] = bad_raw
    np.savez(cache_path, **nan_npz)
    with pytest.raises(ValueError, match="non-finite"):
        cohorts.load("charville", data_dir=tmp_path, include_channels=True)

    # Legacy float32 cache is rebuilt cleanly to float64 without failing.
    legacy = dict(bad_ptr_npz)
    legacy["region_ptr"] = legacy["region_ptr"] - 1
    legacy["channels_raw"] = legacy["channels_raw"].astype(np.float32)
    np.savez(cache_path, **legacy)
    rebuilt = cohorts.load("charville", data_dir=tmp_path, include_channels=True)
    with np.load(cache_path) as reloaded:
        assert reloaded["channels_raw"].dtype == np.float64
    assert np.allclose(rebuilt["channels"][:, 0], [1.0, -1.0, 1.0, -1.0], atol=1e-5)
