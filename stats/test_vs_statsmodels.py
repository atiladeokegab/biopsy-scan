"""A-15.1: the screen's reference statistic against statsmodels PHReg as the oracle."""

import warnings

import numpy as np
import pytest
from statsmodels.duration.hazard_regression import PHReg

from stats import reference, screen
from stats.screen import _newton, _risk

TOL = 1e-8  # float64; the oracle forms the same sums in a different order


def oracle(Xj, time, event, C, strata):
    """PHReg on [C, X_j], Breslow ties, at (beta_C-hat from PHReg on C alone, 0)."""
    rows = np.isfinite(Xj) & np.isfinite(time) & np.isfinite(C).all(1)
    t, e, c, x, s = time[rows], event[rows], C[rows], Xj[rows], strata[rows]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beta = (PHReg(t, c, status=e, strata=s, ties="breslow").fit(disp=False).params
                if c.shape[1] else np.zeros(0))
        model = PHReg(t, np.column_stack([c, x]), status=e, strata=s, ties="breslow")
    params = np.append(beta, 0.0)
    U = model.score(params)[-1]
    info = -model.hessian(params)
    q = c.shape[1]
    V = info[q, q] - (info[q, :q] @ np.linalg.solve(info[:q, :q], info[:q, q]) if q else 0.0)
    return U, V, U / np.sqrt(V)


def test_review_fixture_cox_information_not_permutation_variance():
    X, t, e = np.array([[0.0], [1], [2]]), np.array([1.0, 2, 3]), np.array([1, 1, 0])
    result = screen(X, t, e, n_perm=0)
    np.testing.assert_allclose(result["U"], [-1.5], atol=1e-12)
    np.testing.assert_allclose(result["V"], [11 / 12], atol=1e-12)
    np.testing.assert_allclose(result["z"], [-1.5 / np.sqrt(11 / 12)], atol=1e-12)
    # The legacy statistic is a different one: same U, permutation variance 7/6.
    np.testing.assert_allclose(reference.score(X, t, e, np.zeros(3))["V"], [7 / 6])


def test_review_fixture_missing_value_uses_complete_cases():
    X, t, e = np.array([[0.0], [np.nan], [2]]), np.array([1.0, 2, 3]), np.array([1, 1, 0])
    result = screen(X, t, e, n_perm=0)
    np.testing.assert_allclose(result["U"], [-1.0], atol=1e-12)  # not -5/3 from full residuals
    assert result["n"].tolist() == [2]


@pytest.fixture
def cohort():
    rng = np.random.default_rng(7)
    n, p = 80, 16
    C = np.column_stack([rng.integers(0, 2, n), rng.normal(size=n)]).astype(float)
    X = rng.normal(size=(n, p)) + C[:, :1] * rng.normal(size=p)
    X[:, :4] = rng.integers(0, 3, (n, 4))  # yes/no-like candidates with heavy ties
    X[rng.random((n, p)) < 0.15] = np.nan
    X[:, 5] = np.nan
    X[:40, 5] = rng.normal(size=40)  # a mask that drops a whole stratum's worth
    time = np.round(rng.exponential(np.exp(-0.6 * C[:, 0])), 1) + 0.1  # many tied times
    event = (rng.random(n) < 0.7).astype(float)
    strata = rng.integers(0, 3, n)
    C[3, 1] = np.nan  # missing clinical value: excluded from every analysis set
    return X, time, event, C, strata


@pytest.mark.parametrize("adjusted", [True, False])
def test_every_candidate_matches_phreg_oracle(cohort, adjusted):
    X, time, event, C, strata = cohort
    if not adjusted:
        C = np.zeros((len(time), 0))
    result = screen(X, time, event, C if adjusted else None, strata, n_perm=0, backend="numpy")
    for j in range(X.shape[1]):
        U, V, z = oracle(X[:, j], time, event, C, strata)
        np.testing.assert_allclose([result["U"][j], result["V"][j], result["z"][j]], [U, V, z],
                                   rtol=TOL, atol=TOL, err_msg=f"candidate {j}")
    assert result["metadata"]["masks"] > 2
    assert result["metadata"]["clinical"]["mode"] == ("adjusted" if adjusted else "zero-covariate")


def test_batched_newton_matches_phreg_fit(cohort):
    _, time, event, C, strata = cohort
    rows = np.isfinite(C).all(1)
    t, e, c, s = time[rows], event[rows], C[rows], strata[rows]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = PHReg(t, c, status=e, strata=s, ties="breslow").fit(disp=False).params
    beta = _newton(np, c, t[None], e[None], s[:, None] == s[None])
    np.testing.assert_allclose(beta[0], expected, rtol=1e-7, atol=1e-9)
    M, m, _ = _risk(np, t[None], e[None], np.exp(beta @ c.T), s[:, None] == s[None])
    np.testing.assert_allclose(m @ c, 0, atol=1e-8)  # fitted score is zero
