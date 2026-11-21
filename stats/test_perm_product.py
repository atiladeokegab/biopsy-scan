"""A-15.3: zero-covariate maxT as X'[P_1 r .. P_B r], against brute-force loops.

This checks implementation agreement only, not false-positive control under
clinical adjustment (that is A-15.6).
"""

import numpy as np

from stats import permutation_maxT, reference


def _case(seed=5, n=40, p=30):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0  # constant: z stays 0
    time = np.round(rng.exponential(size=n), 1)
    return X, time, (rng.random(n) < 0.6).astype(int), rng.integers(0, 3, n)


def test_streamed_product_equals_reference_loop():
    X, time, event, strata = _case()
    product = permutation_maxT(X, time, event, strata, n_perm=300, seed=2, block=64, backend="numpy")
    loop = reference.permutation_test(X, time, event, strata, n_resamples=300, seed=2)
    np.testing.assert_allclose(product["null_maxT"], reference.maxT(loop["null_z"]), rtol=1e-10)
    np.testing.assert_allclose(product["z"], loop["z"], rtol=1e-12)
    assert product["p"] == loop["p"]


def test_streamed_product_equals_direct_shuffle_loop():
    X, time, event, strata = _case(p=7)
    product = permutation_maxT(X, time, event, strata, n_perm=25, seed=9, block=4, backend="numpy")
    _, _, _, groups = reference._inputs(X, time, event, strata)
    rng = np.random.default_rng(9)
    expected = []
    for _ in range(25):  # shuffle outcome pairs, rescore each candidate from scratch
        order = np.arange(len(time))
        for ix in groups:
            order[ix] = rng.permutation(ix)
        z = [reference.score(X[:, [j]], time[order], event[order], strata)["z"][0] for j in range(7)]
        expected.append(np.max(np.abs(z)))
    np.testing.assert_allclose(product["null_maxT"], expected, rtol=1e-10)
