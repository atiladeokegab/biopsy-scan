"""A-15.5: the clinical-null parametric bootstrap, batched, against a per-candidate PHReg loop."""

import numpy as np

from stats import screen
from stats.screen import _inputs, _null_draws
from stats.test_vs_statsmodels import oracle


def _cohort(seed=3, n=50, p=8):
    rng = np.random.default_rng(seed)
    C = np.column_stack([rng.integers(0, 2, n), rng.normal(size=n)]).astype(float)
    X = rng.normal(size=(n, p)) + 0.8 * C[:, :1]
    X[rng.random((n, p)) < 0.1] = np.nan
    time = rng.exponential(np.exp(-0.7 * C[:, 0] - 0.3 * C[:, 1]))
    censor = rng.exponential(1.5 * np.exp(-0.5 * C[:, 1]))
    return X, np.minimum(time, censor), (time <= censor).astype(float), C, rng.integers(0, 2, n)


def test_resample_maxT_equals_per_candidate_phreg_loop():
    X, time, event, C, strata = _cohort()
    B, block = 4, 2
    result = screen(X, time, event, C, strata, n_perm=B, seed=11, block=block, backend="numpy")
    _, _, _, Cd, codes, base = _inputs(X, time, event, C, strata)
    expected = []
    for tb, eb in _null_draws(time, event, Cd, codes, base, B, 11, block):
        for t, e in zip(tb, eb):  # refit and rescore every candidate, one at a time
            z = [oracle(X[base, j], t, e, C[base], strata[base])[2] for j in range(X.shape[1])]
            expected.append(np.max(np.abs(z)))
    np.testing.assert_allclose(result["null_maxT"], expected, rtol=1e-7)


def test_draws_follow_the_fitted_null_and_record_the_method():
    X, time, event, C, strata = _cohort(n=200)
    _, _, _, Cd, codes, base = _inputs(X, time, event, C, strata)
    tb, eb = next(_null_draws(time, event, Cd, codes, base, 400, 0, 400))
    assert abs(eb.mean() - event.mean()) < 0.06  # event yield reproduced
    assert set(np.unique(tb)) <= set(time)  # Breslow steps sit on observed times
    meta = screen(X, time, event, C, strata, n_perm=9, backend="numpy")["metadata"]
    assert meta["null"]["B"] == 9 and meta["null"]["name"] == "clinical-null parametric bootstrap"
    assert any("proportional hazards" in a for a in meta["null"]["assumptions"])
    assert meta["clinical"]["shape"] == [200, 2] and len(meta["clinical"]["sha256"]) == 64


def test_p_value_counts_the_observed_row():
    X, time, event, C, strata = _cohort()
    result = screen(X, time, event, C, strata, n_perm=19, backend="numpy")
    k = np.sum(result["null_maxT"] >= result["maxT"])
    assert result["p"] == (1 + k) / 20
    assert np.all(result["adjusted_p"] >= result["p"] - 1e-12)
