"""A-15.2: GPU U, V, z equal the float64 CPU path; legacy zero-covariate regression.

backend="gpu" (the numba.cuda kernel) is checked under the CUDA simulator in a
fresh process, as reimage/ does; the optional cupy backend's checks skip without
a device.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stats import permutation_maxT, reference, screen
from stats.screen import get_backend

GPU = get_backend("auto") is not np  # cupy with a device
GPU_TOL = 1e-9  # float64 on both sides; only summation order differs


def _cohort(seed=1, n=120, p=300):
    rng = np.random.default_rng(seed)
    C = rng.integers(0, 2, (n, 1)).astype(float)
    X = rng.normal(size=(n, p)) + C
    X[rng.random((n, p)) < 0.05] = np.nan
    time = np.round(rng.exponential(np.exp(-0.5 * C[:, 0])), 2)
    return X, time, (rng.random(n) < 0.6).astype(float), C, rng.integers(0, 2, n)


ROOT = str(Path(__file__).resolve().parents[1])
SIM_PARITY = """
import numpy as np
from stats import screen, screen_batch
rng = np.random.default_rng(1)
n, p = 24, 5
C = rng.integers(0, 2, (n, 2)).astype(float)
C[:, 1] = rng.normal(size=n)
X = rng.normal(size=(n, p)) + C[:, :1]
X[rng.random((n, p)) < 0.1] = np.nan
X[:, 4] = 1.0
t = np.round(rng.exponential(size=n), 1)  # ties
e = (rng.random(n) < 0.6).astype(float)
s = rng.integers(0, 2, n)
for clinical in (C, None):
    cpu = screen(X, t, e, clinical, s, n_perm=3, backend="numpy")
    gpu = screen(X, t, e, clinical, s, n_perm=3, backend="gpu")
    for key in ("U", "V", "z", "null_maxT"):
        np.testing.assert_allclose(gpu[key], cpu[key], rtol=1e-9, atol=1e-9, err_msg=key)
    assert gpu["metadata"]["backend"] == "gpu (numba.cuda)"
    T, E = np.round(rng.exponential(size=(3, n)), 1), (rng.random((3, n)) < 0.6).astype(float)
    cpu = screen_batch(X, T, E, clinical, s, backend="numpy")
    gpu = screen_batch(X, T, E, clinical, s, backend="gpu")
    for key in ("U", "V", "z", "maxT"):
        np.testing.assert_allclose(gpu[key], cpu[key], rtol=1e-9, atol=1e-9, err_msg="batch " + key)
print("GPU_SCREEN_PARITY_OK")
"""


SIM_STUDIES = """
import numpy as np
from stats.screen import screen_studies
rng = np.random.default_rng(2)
S, n, p = 5, 20, 6
X = rng.normal(size=(S, n, p)) * 50 + 1e3   # offset and scale: centring must keep parity
X[rng.random((S, n, p)) < 0.15] = np.nan
X[:, :, 3] = 7.0                            # constant: z exactly 0
X[1, :, 4] = np.nan                         # no complete cases in one study
X[2, :, 5] = np.where(np.arange(n) < 10, 1.0, 2.0)  # constant within each stratum below
s = np.tile((np.arange(n) >= 10).astype(int), (S, 1))
s[3] = rng.integers(0, 3, n)                # a study with three strata
t = np.round(rng.exponential(size=(S, n)), 1)  # ties
e = (rng.random((S, n)) < 0.6).astype(float)
e[4] = 0                                    # a study with no events
cpu = screen_studies(X, t, e, s, backend="numpy")
gpu = screen_studies(X, t, e, s, backend="gpu", block=2)
np.testing.assert_allclose(gpu["z"], cpu["z"], rtol=1e-9, atol=1e-9)
np.testing.assert_allclose(gpu["maxT"], cpu["maxT"], rtol=1e-9, atol=1e-9)
assert (cpu["z"][:, 3] == 0).all() and (gpu["z"][:, 3] == 0).all() and (gpu["z"][4] == 0).all()
assert gpu["z"][2, 5] == 0 and np.abs(cpu["z"]).max() > 0
print("GPU_STUDIES_PARITY_OK")
"""


def _run(script, sim):
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    env.update({"PYTHONPATH": ROOT, **({"NUMBA_ENABLE_CUDASIM": "1"} if sim else {})})
    return subprocess.run([sys.executable, "-c", script], env=env, cwd=ROOT,
                          capture_output=True, text=True, check=False)


def test_numba_gpu_kernel_equals_cpu_under_cuda_simulator():
    """Adjusted and zero-covariate U, V, z, bootstrap maxT and screen_batch, within 1e-9."""
    proc = _run(SIM_PARITY, sim=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "GPU_SCREEN_PARITY_OK" in proc.stdout


def test_per_study_kernel_equals_screen_batch_under_cuda_simulator():
    """screen_studies on the GPU (own X per study, blocks of 2) equals screen_batch per study."""
    proc = _run(SIM_STUDIES, sim=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "GPU_STUDIES_PARITY_OK" in proc.stdout


def test_gpu_backend_raises_without_cuda_instead_of_falling_back():
    proc = _run("from numba import cuda; print(cuda.is_available())", sim=False)
    if "True" in proc.stdout:
        pytest.skip("a CUDA device is present")
    proc = _run("import numpy as np; from stats import screen; "
                "screen(np.ones((3, 1)), [1, 2, 3], [1, 0, 1], backend='gpu')", sim=False)
    assert proc.returncode != 0 and "needs a CUDA device" in proc.stderr


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
