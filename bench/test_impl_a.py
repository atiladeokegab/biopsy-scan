"""A-14.1: the CPU baseline's per-candidate PHReg fits agree across worker counts and with a direct call."""
import os

import numpy as np
from statsmodels.duration.hazard_regression import PHReg

from bench import impl_a


def _fixture(seed=0, P=60, J=7):
    """SYNTHETIC: 60 patients, 2 strata, one clinical covariate, 7 candidates with NaN, a constant and a tie."""
    rng = np.random.default_rng(seed)
    C = rng.integers(0, 2, (P, 1)).astype(float)
    X = rng.normal(size=(P, J))
    X[rng.random((P, J)) < 0.1] = np.nan
    X[:, 5] = 1.0  # constant: no fit possible, reported as NaN
    time = np.round(rng.exponential(np.exp(-0.5 * C[:, 0] - 0.4 * np.nan_to_num(X[:, 0]))) * 10) + 1  # ties
    event = (rng.random(P) < 0.7).astype(float)
    strata = rng.integers(0, 2, P)
    return time, event, strata, C, X


def _direct(time, event, strata, C, x):
    ok = np.isfinite(x) & np.isfinite(C).all(1)
    r = PHReg(time[ok], np.column_stack([C[ok], x[ok]]), status=event[ok], strata=strata[ok], ties="breslow").fit()
    return r.params[-1], r.bse[-1], r.pvalues[-1]


def test_workers_all_equals_workers_one_equals_direct_phreg():
    time, event, strata, C, X = _fixture()
    one = impl_a.fit_all(time, event, strata, C, X, workers=1)
    every = impl_a.fit_all(time, event, strata, C, X, workers=os.cpu_count())
    for k in ("coef", "se", "p", "n"):
        np.testing.assert_array_equal(one[k], every[k])
    for j in range(X.shape[1]):
        if j == 5:
            assert np.isnan([one["coef"][j], one["se"][j], one["p"][j]]).all()
            continue
        np.testing.assert_array_equal([one["coef"][j], one["se"][j], one["p"][j]], _direct(time, event, strata, C, X[:, j]))
    assert one["n"][0] == np.isfinite(X[:, 0]).sum()


def test_zero_covariate_mode_fits_the_candidate_alone():
    time, event, strata, _, X = _fixture(seed=1)
    C = np.zeros((len(time), 0))
    got = impl_a.fit_all(time, event, strata, C, X, workers=2)
    ok = np.isfinite(X[:, 2])
    r = PHReg(time[ok], X[ok, 2:3], status=event[ok], strata=strata[ok], ties="breslow").fit()
    np.testing.assert_array_equal([got["coef"][2], got["p"][2]], [r.params[0], r.pvalues[0]])
