"""Regressions for Prometheus's #80 review: each case failed before its fix."""

import sys

import numpy as np
import pytest

from stats import permutation_maxT, screen

S = sys.modules["stats.screen"]


def sparse_cohort():
    C = np.tile([0.0, 1.0], 5)[:, None]
    t = np.array([1, 1, 3, 3, 3, 3, 3, 3, 3, 3], dtype=float)
    e = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    return np.random.default_rng(8).normal(size=(10, 2)), t, e, C


def test_zero_event_bootstrap_draws_contribute_zero_instead_of_crashing():
    X, t, e, C = sparse_cohort()
    out = screen(X, t, e, C, n_perm=19, seed=0, backend="numpy")
    assert np.isfinite(out["null_maxT"]).all() and np.isfinite(out["z"]).all()
    assert len(out["null_maxT"]) == 19
    assert out["metadata"]["null"]["nonconverged"] >= 0


def test_zero_event_rows_give_zero_score_and_information():
    X, t, e, C = sparse_cohort()
    same = np.ones((10, 10), dtype=bool)
    tb, eb = np.stack([t, t]), np.stack([e, np.zeros(10)])
    beta, ok = S._newton(np, C - C.mean(0), tb, eb, same)
    assert ok.all() and beta[1, 0] == 0.0
    U, V, z, bad = S._refit_scores(np, False, [(np.arange(2), np.arange(10), X - X.mean(0), C - C.mean(0), same)],
                              tb, eb, np.arange(10))
    assert (U[1] == 0).all() and (V[1] == 0).all() and (z[1] == 0).all()


def test_z_and_adjusted_p_do_not_depend_on_feature_units():
    X = np.array([1, 0, 2, -1, 3, 4], dtype=float)[:, None]
    X = np.column_stack([X, X[::-1]])
    t, e = np.arange(1.0, 7.0), np.array([1, 0, 0, 0, 0, 0], dtype=float)
    a = screen(X, t, e, n_perm=19, backend="numpy")
    b = screen(X * 1e-7, t, e, n_perm=19, backend="numpy")
    assert a["z"][0] == pytest.approx(-0.292770, abs=1e-6)
    np.testing.assert_allclose(b["z"], a["z"], rtol=1e-9)
    np.testing.assert_array_equal(b["adjusted_p"], a["adjusted_p"])


def test_stratum_offsets_do_not_change_z():
    rng = np.random.default_rng(3)
    n = 40
    s = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 3))
    t, e = rng.exponential(size=n), (rng.random(n) < 0.7).astype(float)
    a = screen(X, t, e, None, s, n_perm=0, backend="numpy")
    b = screen(X + 1e8 * s[:, None], t, e, None, s, n_perm=0, backend="numpy")
    np.testing.assert_allclose(b["z"], a["z"], rtol=1e-9)


def test_resample_identity_does_not_depend_on_block_size():
    X, t, e, C = sparse_cohort()
    rng = np.random.default_rng(1)
    t = rng.exponential(size=10)
    e = (rng.random(10) < 0.7).astype(float)
    one = screen(X, t, e, C, n_perm=3, seed=5, block=1, backend="numpy")["null_maxT"]
    three = screen(X, t, e, C, n_perm=3, seed=5, block=3, backend="numpy")["null_maxT"]
    np.testing.assert_array_equal(one, three)


def test_candidate_streaming_bounds_temporaries_and_keeps_results(monkeypatch):
    rng = np.random.default_rng(2)
    n, p = 30, 9
    C = rng.normal(size=(n, 1))
    X = rng.normal(size=(n, p))
    t, e = rng.exponential(size=n), (rng.random(n) < 0.7).astype(float)
    full = screen(X, t, e, C, n_perm=5, block=5, backend="numpy")
    shapes = []
    real = S._stats
    monkeypatch.setattr(S, "_stats", lambda xp, Xc, *a: shapes.append(Xc.shape) or real(xp, Xc, *a))
    monkeypatch.setattr(S, "CHUNK_ELEMS", 5 * n * 2)
    small = screen(X, t, e, C, n_perm=5, block=5, backend="numpy")
    assert max(s[1] for s in shapes) == 2
    for key in ("z", "U", "V", "null_maxT", "adjusted_p"):
        np.testing.assert_allclose(small[key], full[key], rtol=1e-12, err_msg=key)


def test_permutation_maxT_rejects_the_gpu_backend():
    X = np.random.default_rng(0).normal(size=(6, 2))
    with pytest.raises(ValueError, match="gpu"):
        permutation_maxT(X, np.arange(1.0, 7.0), np.ones(6), None, n_perm=3, backend="gpu")


def test_observed_monotone_clinical_fit_is_rejected_not_used():
    C = np.array([1, 1, 1, 0, 0, 0], dtype=float)[:, None]  # every death has C = 1: no finite beta_C
    t, e = np.arange(1.0, 7.0), np.array([1, 1, 0, 0, 0, 0], dtype=float)
    with pytest.raises(ValueError, match="no finite maximum"):
        screen(np.random.default_rng(0).normal(size=(6, 2)), t, e, C, n_perm=0, backend="numpy")
