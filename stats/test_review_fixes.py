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
    B = 19
    out = screen(X, t, e, C, n_perm=B, seed=0, backend="numpy")
    null = out["null_maxT"]
    assert len(null) == B and np.isfinite(out["z"]).all()  # nothing dropped or redrawn
    _, _, _, Cd, codes, base = S._inputs(X, t, e, C, None)
    events = np.concatenate([eb.sum(1) for _, eb in S._null_draws(t, e, Cd, codes, base, B, 0, 64)])
    assert (events == 0).any() and np.all(null[events == 0] == 0)  # zero information: U = V = z = 0, kept
    failed = int(np.isnan(null).sum())  # monotone refits: represented as NaN, not their 50-step z
    assert out["metadata"]["null"]["nonconverged"] == failed
    exceed = np.sum(null[~np.isnan(null)] >= out["maxT"] - 1e-12) + failed
    assert out["p"] == pytest.approx((1 + exceed) / (B + 1))  # a failed fit counts against the candidate


def test_newton_flags_monotone_likelihood_and_keeps_zero_event_rows():
    C = np.tile([-0.5, 0.5], 5)[:, None]
    t = np.tile([1.0, 2.0] + [3.0] * 8, (3, 1))
    e = np.zeros((3, 10))
    e[0, [0, 1]] = 1  # one death per level: identifiable
    e[1, [0, 2]] = 1  # deaths only at C = -0.5: beta_C has no finite maximum
    beta, ok = S._newton(np, C, t, e, np.ones((10, 10), dtype=bool))
    assert ok.tolist() == [True, False, True] and beta[2, 0] == 0.0


def test_zero_event_rows_give_zero_score_and_information():
    X, t, e, C = sparse_cohort()
    same = np.ones((10, 10), dtype=bool)
    tb, eb = np.stack([t, t]), np.stack([e, np.zeros(10)])
    beta, ok = S._newton(np, C - C.mean(0), tb, eb, same)
    assert ok.all() and beta[1, 0] == 0.0
    group = [(np.arange(2), np.arange(10), X - X.mean(0), C - C.mean(0), same)]
    (_, U, V, z), = S._group_scores(np, False, group, tb, eb, np.arange(10))
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
    np.testing.assert_allclose(b["z"], a["z"], rtol=1e-7)  # the shifted input itself keeps ~8 digits


def test_resample_identity_does_not_depend_on_block_size():
    X, t, e, C = sparse_cohort()
    rng = np.random.default_rng(1)
    t = rng.exponential(size=10)
    e = (rng.random(10) < 0.7).astype(float)
    _, _, _, Cd, codes, base = S._inputs(X, t, e, C, None)
    draws = {b: [np.concatenate(d) for d in zip(*S._null_draws(t, e, Cd, codes, base, 3, 5, b))] for b in (1, 2, 3)}
    for b in (2, 3):  # the resamples themselves are bit-identical
        np.testing.assert_array_equal(draws[b][0], draws[1][0])
        np.testing.assert_array_equal(draws[b][1], draws[1][1])
    one = screen(X, t, e, C, n_perm=3, seed=5, block=1, backend="numpy")["null_maxT"]
    three = screen(X, t, e, C, n_perm=3, seed=5, block=3, backend="numpy")["null_maxT"]
    np.testing.assert_allclose(one, three, rtol=1e-12)  # batched Newton/BLAS order differs only in roundoff


def test_candidate_streaming_bounds_temporaries_and_keeps_results(monkeypatch):
    rng = np.random.default_rng(2)
    n, p = 30, 9
    C = rng.normal(size=(n, 1))
    X = rng.normal(size=(n, p))
    t, e = rng.exponential(size=n), (rng.random(n) < 0.7).astype(float)
    full = screen(X, t, e, C, n_perm=5, block=5, backend="numpy")
    shapes = []
    real = S._stats
    monkeypatch.setattr(S, "_stats", lambda xp, Xc, Cc, M, *a: shapes.append((len(M), *Xc.shape)) or real(xp, Xc, Cc, M, *a))
    monkeypatch.setattr(S, "CHUNK_ELEMS", 5 * n * 2, raising=False)
    small = screen(X, t, e, C, n_perm=5, block=5, backend="numpy")
    assert max(b * m * k for b, m, k in shapes) <= 5 * n * 2  # every [resamples, n, chunk] moment array
    assert max(k for b, _, k in shapes if b == 5) == 2
    for key in ("z", "U", "V", "null_maxT", "adjusted_p"):
        np.testing.assert_allclose(small[key], full[key], rtol=1e-12, err_msg=key)
    null = small["null_maxT"]
    brute = (1 + (null[:, None] >= np.abs(small["z"])[None] - 1e-12).sum(0)) / 6
    np.testing.assert_array_equal(small["adjusted_p"], brute)


def test_permutation_maxT_rejects_the_gpu_backend(monkeypatch):
    import stats.gpu_score as g

    monkeypatch.setattr(g, "require_cuda", lambda: None)  # as under NUMBA_ENABLE_CUDASIM: no silent CPU run
    X = np.random.default_rng(0).normal(size=(6, 2))
    with pytest.raises(ValueError, match="gpu"):
        permutation_maxT(X, np.arange(1.0, 7.0), np.ones(6), np.zeros(6), n_perm=3, backend="gpu")


def test_observed_monotone_clinical_fit_is_rejected_not_used():
    C = np.array([1, 1, 1, 0, 0, 0], dtype=float)[:, None]  # every death has C = 1: no finite beta_C
    t, e = np.arange(1.0, 7.0), np.array([1, 1, 0, 0, 0, 0], dtype=float)
    with pytest.raises(ValueError, match="no finite maximum"):
        screen(np.random.default_rng(0).normal(size=(6, 2)), t, e, C, n_perm=0, backend="numpy")


def test_column_constant_within_strata_scores_exactly_zero():
    rng = np.random.default_rng(6)
    s = rng.integers(0, 3, 30)
    X = np.column_stack([0.1 * s + 0.3, rng.normal(size=30)])  # column 0 is a pure stratum offset
    t, e = rng.exponential(size=30), (rng.random(30) < 0.7).astype(float)
    out = screen(X, t, e, None, s, n_perm=0, backend="numpy")
    assert out["z"][0] == 0.0 and out["V"][0] == 0.0 and out["z"][1] != 0.0
