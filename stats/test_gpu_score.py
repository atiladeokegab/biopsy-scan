"""A-15.2: GPU U, V, z equal the float64 CPU path; legacy zero-covariate regression.

The GPU parity tests run only where cupy sees a CUDA device and skip cleanly
elsewhere. They are UNPROVEN until run on a GPU box.
"""

import numpy as np
import pytest

from stats import permutation_maxT, reference, screen
from stats.screen import get_backend

GPU = get_backend("auto") is not np
GPU_TOL = 1e-9  # float64 on both sides; only summation order differs


def _cohort(seed=1, n=120, p=300):
    rng = np.random.default_rng(seed)
    C = rng.integers(0, 2, (n, 1)).astype(float)
    X = rng.normal(size=(n, p)) + C
    X[rng.random((n, p)) < 0.05] = np.nan
    time = np.round(rng.exponential(np.exp(-0.5 * C[:, 0])), 2)
    return X, time, (rng.random(n) < 0.6).astype(float), C, rng.integers(0, 2, n)


def test_backend_falls_back_to_numpy_without_a_device():
    if GPU:
        pytest.skip("a CUDA device is present")
    assert get_backend("auto") is np
    with pytest.raises(RuntimeError):
        get_backend("cupy")


@pytest.mark.skipif(not GPU, reason="no cupy with a CUDA device; GPU parity UNMEASURED here")
def test_gpu_equals_cpu_including_the_bootstrap():
    args = _cohort()
    cpu = screen(*args, n_perm=16, backend="numpy")
    gpu = screen(*args, n_perm=16, backend="cupy")
    for key in ("U", "V", "z", "null_maxT"):
        np.testing.assert_allclose(gpu[key], cpu[key], rtol=GPU_TOL, atol=GPU_TOL)
    assert gpu["metadata"]["backend"] == "cupy"


@pytest.mark.skipif(not GPU, reason="no cupy with a CUDA device; GPU parity UNMEASURED here")
def test_gpu_permutation_product_equals_cpu():
    X, time, event, _, strata = _cohort()
    X = np.nan_to_num(X)
    cpu = permutation_maxT(X, time, event, strata, n_perm=200, backend="numpy")
    gpu = permutation_maxT(X, time, event, strata, n_perm=200, backend="cupy")
    np.testing.assert_allclose(gpu["null_maxT"], cpu["null_maxT"], rtol=GPU_TOL)


def test_legacy_permutation_variance_path_stays_equal_to_reference():
    X, time, event, _, strata = _cohort(p=20)
    X = np.nan_to_num(X)
    legacy = permutation_maxT(X, time, event, strata, n_perm=1, backend="numpy")
    expected = reference.score(X, time, event, strata)
    for key in ("U", "V", "z"):
        np.testing.assert_allclose(legacy[key], expected[key], rtol=1e-12)
