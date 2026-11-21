"""Fleet sharding bit-identity check with v2 expression features active (Req: A-12.2)."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_v2_fleet_two_shards_bit_identical_to_single_device():
    script = """
import numpy as np
from core.grammar import EXPRESSION_CHANNELS
from reimage import run_v2

rng = np.random.default_rng(17)
n_reg, n_cell, K = 4, 24, 4
xs, ys, ts, chs, rid = [], [], [], [], []
for r in range(n_reg):
    xs.append(rng.random(n_cell).astype(np.float32) * 120.0)
    ys.append(rng.random(n_cell).astype(np.float32) * 120.0)
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
design = {"cells": cells, "backend": "gpu", "patients": None, "regions_per_patient": 1, "fov_um": 80.0, "eps": 0.2, "seed": 5}
seeds = np.array([10, 20, 30, 40, 50], dtype=np.int64)

single = run_v2(design, seeds)
sharded = run_v2(design, seeds, devices=2, fake_shards=True)
np.testing.assert_array_equal(sharded, single)
print("V2_FLEET_BIT_IDENTICAL_OK")
"""
    root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": root}
    proc = subprocess.run([sys.executable, "-c", script], env=env, cwd=root, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "V2_FLEET_BIT_IDENTICAL_OK" in proc.stdout
