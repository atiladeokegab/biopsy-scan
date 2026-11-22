import numpy as np
import pytest

from bench.charville_v2_gpu import permutation_screen, same_outputs
from stats import reference


@pytest.mark.parametrize('missing', [False, True])
def test_batched_legacy_matches_reference_with_ties_and_complete_cases(missing):
    rng = np.random.default_rng(201)
    X = rng.normal(size=(43, 12))
    X[:, -1] = 0.5
    if missing:
        X[::3, 2:5] = np.nan
        X[::5, 5:8] = np.nan
        X[:, -2] = np.nan
    time = rng.integers(0, 9, size=len(X)).astype(float)
    event = rng.integers(0, 2, size=len(X))
    strata = rng.integers(0, 3, size=len(X))
    event[strata == 2] = 0
    args = X, time, event, strata
    ref = reference.permutation_test(*args, n_resamples=97, seed=31)
    ref['null_maxT'] = reference.maxT(ref['null_z'])
    for size in (1, 16, 256):
        batch = permutation_screen(*args, n_resamples=97, seed=31, block=size)
        assert batch['maxT'] == ref['maxT']
        assert max(same_outputs(batch, ref).values()) < 1e-12
