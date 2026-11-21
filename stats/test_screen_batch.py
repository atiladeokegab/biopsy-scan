"""screen_batch: each study's row equals screen(n_perm=0) on that study alone."""

import numpy as np
import pytest

from stats import screen, screen_batch


def cohort(rng, n=40, p=6):
    C = np.column_stack([rng.integers(0, 2, n), rng.normal(size=n)]).astype(float)
    X = rng.normal(size=(n, p)) + C[:, :1]
    X[rng.random((n, p)) < 0.1] = np.nan
    X[:, -1] = 2.0  # constant: z = 0
    C[3, 1] = np.nan  # one patient drops from the analysis base
    return X, C, rng.integers(0, 3, n)


@pytest.mark.parametrize("adjusted", [True, False])
def test_rows_equal_single_study_screen(adjusted):
    rng = np.random.default_rng(4)
    X, C, s = cohort(rng)
    S = 5
    time = np.round(rng.exponential(size=(S, len(s))), 1)  # ties
    event = (rng.random((S, len(s))) < 0.6).astype(float)
    clinical = C if adjusted else None
    out = screen_batch(X, time, event, clinical, s, backend="numpy", block=2)
    for k in range(S):
        one = screen(X, time[k], event[k], clinical, s, n_perm=0, backend="numpy")
        for key in ("U", "V", "z"):
            np.testing.assert_allclose(out[key][k], one[key], rtol=1e-8, atol=1e-10, err_msg=key)
        assert out["maxT"][k] == pytest.approx(one["maxT"], rel=1e-8)
    np.testing.assert_array_equal(out["n"], one["n"])
    assert out["metadata"]["studies"] == S


def test_rejects_missing_outcomes():
    X = np.ones((3, 1))
    with pytest.raises(ValueError):
        screen_batch(X, [[1.0, np.nan, 2.0]], [[1, 0, 1]])
