"""Parity, invariance, and cohort-channel tests for grammar v2 expression features (Req: A-12.1)."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
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
from core.reimage_ref import neighbours, reimage, reimage_v2, reimage_region_v2
from core.truth import truth, truth_v2
from reimage import cpu, run_v2

TYPES = ["B cell", "Stroma", "Tumor", "Tumor (Ki67+)"]
K, FOCAL = 4, [2, 3]


def _make_v2_cells(seed=0):
    rng = np.random.default_rng(seed)
    xs, ys, ts, chs, rid = [], [], [], [], []
    for r in range(5):
        n = 90
        xy = rng.random((n, 2)) * 220.0
        xy[1] = xy[0]
        ty = rng.integers(0, K, n).astype(np.int16)
        if r == 4:
            ty[ty == 3] = 2  # patient 2 lacks Tumor (Ki67+)
        ch = rng.normal(0.0, 1.0, size=(n, len(EXPRESSION_CHANNELS))).astype(np.float32)
        xs.append(xy[:, 0]); ys.append(xy[:, 1]); ts.append(ty); chs.append(ch)
        rid.append(np.full(n, r, dtype=np.int32))
    return {
        "cell_type": np.concatenate(ts).astype(np.int8),
        "x": np.concatenate(xs).astype(np.float32),
        "y": np.concatenate(ys).astype(np.float32),
        "channels": np.concatenate(chs, axis=0),
        "region_id": np.concatenate(rid),
        "patient_of_region": np.array([0, 0, 1, 1, 2], dtype=np.int32),
        "type_names": TYPES,
        "channel_names": list(EXPRESSION_CHANNELS),
    }


def test_grammar_v1_hash_frozen_and_v2_adds_10_channels():
    g1 = grammar(TYPES)
    g2 = grammar_v2(TYPES)
    assert g1["version"] == 1 and n_features(g1) == 2 * 4 * 3 * 2 == 48
    assert g2["version"] == 2
    assert tuple(g2["channels"]) == EXPRESSION_CHANNELS and len(g2["channels"]) == 10
    assert g2["transform"] == EXPRESSION_TRANSFORM
    assert n_features(g2) == 48 + 2 * 4 * 3 * 10 == 288
    assert len(feature_names(g2)) == 288
    assert feature_names(g2)[:48] == feature_names(g1)
    assert feature_names(g2)[48] == "Tumor|B cell|25um|PD1"
    assert grammar_hash(g1) != grammar_hash(g2)


def test_truth_v2_equals_reimage_v2_no_crop_eps0_and_preserves_v1_slice():
    cells = _make_v2_cells(seed=5)
    x1 = truth(cells)
    x12 = truth_v2(cells)
    assert x12.shape == (3, 288)
    np.testing.assert_array_equal(x12[:, :48], x1)
    assert np.isnan(x12[2, 24:48]).all()
    assert np.isnan(x12[2, 48 + 120:]).all()

    r12 = reimage_v2(cells, {"patients": None, "regions_per_patient": None, "fov_um": None, "eps": 0.0}, seed=11)
    np.testing.assert_allclose(r12, x12, rtol=0, atol=1e-12, equal_nan=True)


def test_miscall_relabels_neighbours_without_altering_cell_expression():
    """Crop + mis-call change which neighbours contribute, never a cell's own expression vector."""
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    ty = np.array([2, 0, 1], dtype=np.int16)  # cell 0 is Tumor; 1 is B cell; 2 is Stroma
    ch = np.zeros((3, len(EXPRESSION_CHANNELS)), dtype=np.float32)
    ch[1, :] = 2.0
    ch[2, :] = 6.0
    nbr = neighbours(xy)
    # Confusion swaps cell 1 (B cell -> Stroma), keeps cell 0 & 2 unchanged via deterministic u
    M = np.eye(K)
    M[0, 0], M[0, 1] = 0.0, 1.0  # all B cells miscalled as Stroma
    u = np.array([0.0, 0.5, 0.0])
    sc, se, m = reimage_region_v2(nbr, xy, ty, ch, FOCAL, K, crop=None, M=M, u=u)
    assert m[0] == 1
    # Both neighbours now observed as Stroma (type 1); their original expressions (2.0, 6.0) average to 4.0
    assert sc[0, 0, 0, 0] == 0.0
    assert np.allclose(se[0, 1, 0], 4.0)
    assert np.allclose(se[0, 0, 0], 0.0)


def test_cpu_v2_matches_numpy_reference():
    cells = _make_v2_cells(seed=9)
    design = {"patients": 4, "regions_per_patient": 1, "fov_um": 120.0, "eps": 0.25, "seed": 2}
    ref = run_v2({**design, "cells": cells, "backend": "numpy"}, [3, 7])
    got = run_v2({**design, "cells": cells, "backend": "cpu"}, [3, 7])
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12, equal_nan=True)
    # Also verify v1 portion matches reimage.run directly
    v1 = reimage(cells, design, 3)
    np.testing.assert_array_equal(ref[0, :, :48], v1)


def test_load_channels_aligns_cell_ids_and_zscores_cohort_before_reimaging(tmp_path):
    (tmp_path / "charville_labels.csv").write_text(
        "region_id,primary_outcome,recurrence,alive_or_deceased,type_of_first_recurrence,surgery_date,"
        "first_recurrence_date,last_contact_date,length_of_disease_free_survival,grade_differentiation,"
        "acquisition_id_visualizer,sample_label_visualizer\n"
        "Charville_c001_v001_r001_reg001,0,0,Dead,X,1/1/2010,NONE,1/11/2010,1,2,a,100_1\n"
        "Charville_c001_v001_r001_reg002,0,0,Alive,X,1/1/2010,NONE,1/21/2010,1,2,a,101_1\n"
    )
    expr_cols = [c for c in EXPRESSION_CHANNELS if c != "SIZE"]
    with zipfile.ZipFile(tmp_path / "charville_raw_data.zip", "w") as z:
        for reg, offset in [("Charville_c001_v001_r001_reg001", 0.0), ("Charville_c001_v001_r001_reg002", 2.0)]:
            z.writestr(f"raw_data/{reg}.cell_data.csv", "CELL_ID,X,Y\n10,0,0\n20,50,0\n")
            z.writestr(f"raw_data/{reg}.cell_types.csv", "CELL_ID,CELL_TYPE\n20,Tumor\n10,Stroma\n")
            hdr = "CELL_ID," + ",".join(expr_cols)
            r1 = "20," + ",".join(str(-1.5 + offset + i * 0.1) for i in range(len(expr_cols)))
            r2 = "10," + ",".join(str(0.5 + offset + i * 0.1) for i in range(len(expr_cols)))
            z.writestr(f"raw_data/{reg}.expression.csv", f"{hdr}\n{r1}\n{r2}\n")
            z.writestr(f"raw_data/{reg}.cell_features.csv", f"CELL_ID,SIZE\n20,{0.2+0.1*offset}\n10,{0.6+0.1*offset}\n")

    cells = cohorts.load("charville", data_dir=tmp_path, include_channels=True)
    assert cells["channels"].dtype == np.float32 and cells["channels"].shape == (4, 10)
    assert cells["channel_names"] == list(EXPRESSION_CHANNELS)
    assert np.allclose(cells["channels"].mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(cells["channels"].std(axis=0), 1.0, atol=1e-5)


def test_gpu_v2_matches_reference_under_cuda_simulator():
    script = """
import numpy as np
from core.grammar import EXPRESSION_CHANNELS
from reimage import run_v2

rng = np.random.default_rng(42)
n_reg, n_cell, K = 4, 30, 4
xs, ys, ts, chs, rid = [], [], [], [], []
for r in range(n_reg):
    xs.append(rng.random(n_cell).astype(np.float32) * 140.0)
    ys.append(rng.random(n_cell).astype(np.float32) * 140.0)
    ts.append(rng.integers(0, K, n_cell, dtype=np.int8))
    chs.append(rng.normal(0.0, 1.0, size=(n_cell, len(EXPRESSION_CHANNELS))).astype(np.float32))
    rid.append(np.full(n_cell, r, dtype=np.int32))
cells = {
    "cell_type": np.concatenate(ts),
    "x": np.concatenate(xs),
    "y": np.concatenate(ys),
    "channels": np.concatenate(chs, axis=0),
    "region_id": np.concatenate(rid),
    "patient_of_region": np.array([0, 0, 1, 1], dtype=np.int32),
    "type_names": ["B cell", "Stroma", "Tumor", "Tumor (Ki67+)"],
}
design = {"cells": cells, "patients": None, "regions_per_patient": 1, "fov_um": 90.0, "eps": 0.2, "seed": 5}
ref = run_v2({**design, "backend": "numpy"}, [3, 9])
gpu_out = run_v2({**design, "backend": "gpu"}, [3, 9])
np.testing.assert_allclose(gpu_out, ref, rtol=0, atol=1e-11, equal_nan=True)
print("GPU_V2_PARITY_OK")
"""
    root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": root}
    proc = subprocess.run([sys.executable, "-c", script], env=env, cwd=root, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "GPU_V2_PARITY_OK" in proc.stdout
