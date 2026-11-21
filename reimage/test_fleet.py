"""Multi-GPU sharding tests, including bit-identical 2-shard check under the CUDA simulator."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from core.truth import neighbours
from reimage import run
from reimage.fleet import run_sharded, shard_bounds


def test_shard_bounds_splits_contiguously():
    assert shard_bounds(8, 1) == [(0, 0, 8)]
    assert shard_bounds(8, 2) == [(0, 0, 4), (1, 4, 8)]
    assert shard_bounds(7, 3) == [(0, 0, 3), (1, 3, 5), (2, 5, 7)]
    assert shard_bounds(3, 8) == [(0, 0, 1), (1, 1, 2), (2, 2, 3)]


def test_two_fake_shards_on_cuda_simulator_are_bit_identical():
    """Run in a fresh process with NUMBA_ENABLE_CUDASIM=1 so Numba initializes the simulator cleanly."""
    script = """
import numpy as np
from core.truth import neighbours
from bench.ladder import synthetic_pool, time_gpu_sharded
from reimage import run
from reimage.fleet import run_sharded

xy = np.array([[0, 0], [0, 0], [25, 0], [50, 0], [100, 0], [110, 0]], dtype=np.float64)
types = np.array([0, 1, 2, 1, 0, 2], dtype=np.int16)
cells = dict(
    x=np.tile(xy[:, 0], 4).astype(np.float32),
    y=np.tile(xy[:, 1], 4).astype(np.float32),
    cell_type=np.tile(types, 4).astype(np.int8),
    region_id=np.repeat(np.arange(4), len(types)).astype(np.int32),
    patient_of_region=np.array([0, 0, 1, 1], dtype=np.int32),
    type_names=["Tumor A", "Other B", "Other C"],
)
design = dict(cells=cells, backend="gpu", patients=None, regions_per_patient=1, fov_um=40.0, eps=0.25, seed=7)
seeds = np.array([11, 22, 33, 44, 55], dtype=np.int64)

single = run(design, seeds)
sharded = run_sharded(design, seeds, devices=2, fake_shards=True)
assert single.shape == (5, 2, 18)
np.testing.assert_array_equal(sharded, single)

# Also verify bench.ladder.time_gpu_sharded with 2 fake shards matches single-shard outputs bit for bit.
pool = synthetic_pool(4, 40, 120.0, seed=3)
for reg in pool:
    reg["nbr"] = neighbours(reg["xy"])
ld = {"k": 2, "fov_um": 60.0, "eps": 0.1}
_, _, one_outs = time_gpu_sharded(pool, ld, 4, devices=1, fake_shards=True)
t2, _, two_outs = time_gpu_sharded(pool, ld, 4, devices=2, fake_shards=True)
assert t2["devices"] == 2 and len(one_outs) == 4 and len(two_outs) == 4
for a_study, b_study in zip(one_outs, two_outs):
    for (s0, m0), (s1, m1) in zip(a_study, b_study):
        np.testing.assert_array_equal(s1, s0)
        np.testing.assert_array_equal(m1, m0)
print("BIT_IDENTICAL_OK")
"""
    root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, "NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": root}
    proc = subprocess.run([sys.executable, "-c", script], env=env, cwd=root, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "BIT_IDENTICAL_OK" in proc.stdout


def test_invalid_device_requests_fail_honestly():
    cells = {
        "x": np.array([0.0], dtype=np.float32),
        "y": np.array([0.0], dtype=np.float32),
        "cell_type": np.array([0], dtype=np.int8),
        "region_id": np.array([0], dtype=np.int32),
        "patient_of_region": np.array([0], dtype=np.int32),
        "type_names": ["Tumor A"],
    }
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="devices"):
            shard_bounds(4, bad)
        with pytest.raises(ValueError, match="devices"):
            run({"cells": cells, "backend": "numpy"}, [0, 1], devices=bad)
    with pytest.raises(RuntimeError, match="NUMBA_ENABLE_CUDASIM"):
        run_sharded({"cells": cells, "backend": "gpu"}, [0, 1], devices=2, fake_shards=True)
