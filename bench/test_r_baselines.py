import numpy as np

from bench import r_baselines as rb
from stats import reference


def _case(seed=7, n=60, p=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[rng.random(n) < 0.15, :4] = np.nan  # one missingness mask over the first 4 columns
    X[:5, 6] = np.nan  # a second mask
    X[:, 8] = 2.0  # constant: z stays 0
    time = np.round(rng.exponential(size=n), 1)  # ties
    return X, time, (rng.random(n) < 0.5).astype(float), rng.integers(0, 3, n)


def test_batched_shuffles_equal_reference_permutation_test():
    X, time, event, strata = _case()
    P = rb.draws(strata, 200, seed=3)
    got = rb.ours(X, time, event, strata, P, block=64)
    ref = reference.permutation_test(X, time, event, strata, n_resamples=200, seed=3)
    np.testing.assert_allclose(got["z"], ref["z"], atol=1e-12)
    np.testing.assert_allclose(got["null_maxT"], reference.maxT(ref["null_z"]), atol=1e-12)
    assert got["p"] == ref["p"] and got["maxT"] == ref["maxT"]


def test_permgpu_transcription_on_complete_columns_is_the_unstratified_cox_score_chi2():
    X, time, event, _ = _case()
    full = np.isfinite(X).all(0) & (X.std(0) > 0)
    got = rb.permgpu_stat(X[:, full], time, event)
    # unstratified Breslow score and information at beta = 0, one column at a time
    for k, x in enumerate(X[:, full].T):
        u = i = 0.0
        for d in np.flatnonzero(event == 1):
            at = time >= time[d]
            m = x[at].mean()
            u += x[d] - m
            i += np.mean(x[at] ** 2) - m * m
        assert np.isclose(got[k], u * u / i, rtol=1e-12)


def test_ratio_and_spread():
    s = rb.spread([2.0, 4.0, 3.0])
    assert s["median_s"] == 3.0 and abs(s["spread_pct"] - 200 / 3) < 1e-12
    assert rb.ratio(rb.spread([10.0, 10.0, 10.0]), s) == {"median": 10 / 3, "range": [2.5, 5.0]}
