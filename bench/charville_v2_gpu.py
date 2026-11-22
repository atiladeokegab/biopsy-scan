"""Batch the legacy Charville v2 permutation test without changing its null."""
from __future__ import annotations

from collections import Counter
from time import perf_counter

import numpy as np

from stats import reference

BATCH = 256
TOLERANCE = {"rtol": 1e-9, "atol": 1e-9}


def residual_block(time, event, groups):
    """Stratified Breslow residuals for a block of already permuted endpoints."""
    out = np.zeros_like(time)
    for ix in groups:
        n = len(ix)
        order = np.argsort(time[:, ix], axis=1, kind="stable")
        t = np.take_along_axis(time[:, ix], order, axis=1)
        e = np.take_along_axis(event[:, ix], order, axis=1)
        start = np.column_stack((np.ones(len(t), dtype=bool), t[:, 1:] != t[:, :-1]))
        end = np.column_stack((start[:, 1:], np.ones(len(t), dtype=bool)))
        positions = np.arange(n)
        first = np.maximum.accumulate(np.where(start, positions, 0), axis=1)
        last = np.minimum.accumulate(np.where(end, positions, n)[:, ::-1], axis=1)[:, ::-1]
        prefix = np.column_stack((np.zeros(len(t)), np.cumsum(e, axis=1)))
        failures = np.take_along_axis(prefix, last + 1, axis=1) - np.take_along_axis(prefix, first, axis=1)
        hazard = np.cumsum(np.where(start, failures / (n - first), 0.0), axis=1)
        residual = np.empty_like(t)
        np.put_along_axis(residual, order, e - hazard, axis=1)
        out[:, ix] = residual - residual.mean(axis=1, keepdims=True)
    return out


def column_groups(X, groups):
    masks, inverse = np.unique(np.isfinite(X).T, axis=0, return_inverse=True)
    for i, keep in enumerate(masks):
        if not keep.any():
            continue
        cols = np.flatnonzero(inverse == i)
        rows = np.flatnonzero(keep)
        mapping = np.cumsum(keep) - 1
        subsets = [mapping[ix[keep[ix]]] for ix in groups if keep[ix].any()]
        x = X[np.ix_(rows, cols)]
        centered = np.zeros_like(x)
        scales = np.zeros((len(subsets), len(cols)))
        for k, ix in enumerate(subsets):
            if len(ix) > 1:
                values = x[ix] - x[ix[:1]]
                values -= values.mean(axis=0, keepdims=True)
                centered[ix] = values
                scales[k] = (values * values).sum(axis=0) / (len(ix) - 1)
        yield rows, cols, subsets, centered, scales


def permutation_screen(X, time, event, strata, *, n_resamples, seed, xp=np, block=BATCH, sample_memory=None):
    """Preserve reference draws, complete cases, maxT, adjusted p-values and omnibus.

    CPU generates the random permutations and mask-specific Breslow residuals. The
    selected array backend computes batched float64 numerators and variances.
    """
    if n_resamples < 1 or block < 1:
        raise ValueError("resamples and block must be positive")
    X, time, event, groups = reference._inputs(X, time, event, strata)
    if X.ndim != 2:
        raise ValueError("one patient-by-feature matrix is required")
    observed = reference._complete_cases(X, time, event, groups)
    prepared = [(rows, cols, subs, xp.asarray(centered), xp.asarray(scales))
                for rows, cols, subs, centered, scales in column_groups(X, groups)]
    rng = np.random.default_rng(seed)
    result = np.zeros((n_resamples + 1, X.shape[1]))
    result[0] = observed["z"]
    product_wall = 0.0
    launches = Counter()
    for start in range(0, n_resamples, block):
        size = min(block, n_resamples - start)
        order = np.empty((size, len(time)), dtype=np.intp)
        for b in range(size):
            for ix in groups:
                order[b, ix] = rng.permutation(ix)
        for rows, cols, subs, centered, scales in prepared:
            # Restrict the outcome permutation after shuffling on the full cohort.
            chosen = order[:, rows]
            r = residual_block(time[chosen], event[chosen], subs)
            squares = np.column_stack([np.einsum("bi,bi->b", r[:, ix], r[:, ix]) for ix in subs])
            t0 = perf_counter()
            U = xp.asarray(r) @ centered
            V = xp.asarray(squares) @ scales
            z = xp.where(V > 0, U / xp.sqrt(xp.where(V > 0, V, 1.0)), 0.0)
            host = z if xp is np else xp.asnumpy(z)
            product_wall += perf_counter() - t0
            if sample_memory is not None:
                sample_memory()
            result[start + 1:start + size + 1, cols] = host
            launches[(size, len(cols))] += 1
    maxima = reference.maxT(result)
    max_observed = float(maxima[0])
    tol = 16 * np.finfo(np.float64).eps * max(1, max_observed)
    return {**observed, "maxT": max_observed, "p": float(np.mean(maxima >= max_observed - tol)),
            "adjusted_p": np.mean(maxima[:, None] >= np.abs(observed["z"]) - tol, axis=0),
            "omnibus": reference._joint_test(result), "null_z": result[1:], "null_maxT": maxima[1:],
            "product_wall_s": product_wall,
            "score_batches": [{"resamples": b, "features": c, "count": n} for (b, c), n in sorted(launches.items())]}


def same_outputs(left, right):
    """Compare the whole null grid in blocks and require exact empirical p-values."""
    deltas = {}
    for k in ("U", "V", "z", "null_maxT", "null_z"):
        a, b = np.atleast_2d(left[k]), np.atleast_2d(right[k])
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for {k}")
        maximum = 0.0
        for i in range(0, len(a), BATCH):
            x, y = a[i:i + BATCH], b[i:i + BATCH]
            if not np.allclose(x, y, **TOLERANCE):
                raise ValueError(f"CPU and GPU differ for {k} at block {i}")
            maximum = max(maximum, float(np.max(np.abs(x - y))))
        deltas[k] = maximum
    if (left["p"] != right["p"] or not np.array_equal(left["adjusted_p"], right["adjusted_p"])
            or left["omnibus"]["p"] != right["omnibus"]["p"]
            or left["omnibus"]["component_p"] != right["omnibus"]["component_p"]):
        raise ValueError("CPU and GPU empirical p-values differ")
    return deltas
