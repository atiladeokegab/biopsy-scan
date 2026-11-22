"""reimage.gpu_fused against the CPU fused arm on invented tissue: X bit for bit, z within 1e-9."""
import os
import subprocess
import sys

import numpy as np

DESIGNS = (dict(patients=12, regions_per_patient=1, fov_um=60.0, eps=0.2),  # region choice, crops, mis-calls
           dict(patients=None, regions_per_patient=None, fov_um=None, eps=0.0))  # the truth layer: no draws at all
SEEDS = (3, 2**63 + 11, 20261121, 7, 8)


def _cells():
    """tests/conftest.py's tiny_cells: 216 cells, three patients, six regions, three types."""
    grid, spacing = np.arange(36), np.repeat(np.arange(10, 40, 5), 36)
    return {"cell_type": ((np.tile(grid % 3, 6) + np.repeat(np.arange(6), 36)) % 3).astype(np.int8),
            "x": (np.tile(grid % 6, 6) * spacing).astype(np.float32),
            "y": (np.tile(grid // 6, 6) * spacing).astype(np.float32),
            "region_id": np.repeat(np.arange(6), 36).astype(np.int32),
            "patient_of_region": np.repeat(np.arange(3), 2).astype(np.int32)}


def check(backend: str) -> None:
    from core.reimage_ref import draw, reimage
    from reimage import fused, gpu_fused as gf
    from stats import screen_batch

    np.seterr(over="ignore")  # the simulator's uint64 scalars wrap, as the compiled limbs do
    for design in DESIGNS:
        cells = _cells()
        st = gf.static(cells, design)
        pl = gf.merge([gf.plan(st, SEEDS[:2]), gf.plan(st, SEEDS[2:])])
        S, P = pl["patients"].shape
        rng = np.random.default_rng(1)
        time, event, strata = rng.exponential(1.0, (S, P)), (rng.random((S, P)) < 0.7).astype(float), rng.integers(0, 2, (S, P))
        out = gf.Fused(cells, design, backend=backend, st=st).evaluate(pl, time, event, strata, keep_X=True)
        for s, seed in enumerate(SEEDS):
            assert np.array_equal(pl["patients"][s], draw(cells, design, seed)["patients"])
            X = reimage(cells, design, seed, region_kernel=fused.reimage_region)
            assert out["X"][s].tobytes() == X.tobytes(), (backend, design, seed)
            z = screen_batch(X, time[s][None], event[s][None], strata=strata[s], backend="numpy")["z"][0]
            np.testing.assert_allclose(out["z"][s], z, rtol=1e-9, atol=1e-9)
            assert abs(out["maxT"][s] - np.abs(z).max()) <= 1e-9
        assert np.isnan(out["X"]).any() == (design["fov_um"] is not None)  # a crop can hide every focal cell of a type


def test_cpu_backend_matches_the_fused_arm_bit_for_bit():
    check("cpu")


def test_cuda_simulator_matches_the_fused_arm_bit_for_bit():
    """A fresh process with NUMBA_ENABLE_CUDASIM=1, so Numba starts the simulator cleanly."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": root}
    run = subprocess.run([sys.executable, "-c", "from reimage.test_gpu_fused import check; check('gpu')"],
                         cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]


def test_real_device_matches_the_fused_arm_bit_for_bit():
    import pytest
    from numba import cuda
    if not cuda.is_available() or cuda.config.ENABLE_CUDASIM:
        pytest.skip("no CUDA device")
    check("gpu")


def test_gpu_backend_refuses_without_a_device():
    import pytest
    from numba import cuda
    from reimage import gpu_fused as gf
    if cuda.is_available():
        pytest.skip("a CUDA device or the simulator is present")
    with pytest.raises(RuntimeError, match="CUDA"):
        gf.Fused(_cells(), DESIGNS[0])
