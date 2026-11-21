"""CPU reference: Breslow ties, permutation variance and joint calibration."""

from itertools import permutations, product
from math import ceil, factorial

import numpy as np
from scipy.special import ndtr


def _inputs(X, time, event, strata):
    X = np.asarray(X, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    strata = np.asarray(strata)
    if X.ndim not in (2, 3) or not all(X.shape):
        raise ValueError("X must have nonempty shape [patients, features] or [studies, patients, features]")
    n = X.shape[-2]
    if time.shape != (n,) or event.shape != (n,) or strata.shape != (n,):
        raise ValueError("time, event and strata must each have shape [patients]")
    if not np.isfinite(X).all() or not np.isfinite(time).all() or (time < 0).any():
        raise ValueError("X and time must be finite; time must be nonnegative")
    if not np.isin(event, [0, 1]).all():
        raise ValueError("event must contain only 0 or 1")
    if strata.dtype.kind not in "biufUS" or (strata.dtype.kind in "f" and not np.isfinite(strata).all()):
        raise ValueError("strata must contain finite numeric or string labels")
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    return X, time, event.astype(np.float64), groups


def _residuals(time, event, groups):
    r = np.zeros_like(time)
    for ix in groups:
        order = np.argsort(time[ix], kind="stable")
        indices = ix[order]
        times = time[indices]
        _, first, counts = np.unique(times, return_index=True, return_counts=True)
        deaths = np.add.reduceat(event[indices], first)
        # Same-time deaths and censorings are all at risk at the event time.
        hazard = deaths / (len(ix) - first)
        r[indices] = event[indices] - np.repeat(np.cumsum(hazard), counts)
        r[ix] -= r[ix].mean()  # Remove only numerical error in the zero sum.
    return r


def _score(X, r, groups):
    centered = np.zeros_like(X)
    V = np.zeros(X.shape[:-2] + (X.shape[-1],), dtype=np.float64)
    for ix in groups:
        if len(ix) < 2 or not np.any(r[ix]):
            continue
        # Anchoring keeps constant columns exactly zero and avoids large offsets.
        x = X[..., ix, :] - X[..., ix[:1], :]
        x -= x.mean(axis=-2, keepdims=True)
        centered[..., ix, :] = x
        V += np.sum(x * x, axis=-2) * np.dot(r[ix], r[ix]) / (len(ix) - 1)
    U = np.einsum("...pc,p->...c", centered, r)
    z = np.divide(U, np.sqrt(V), out=np.zeros_like(U), where=V > 0)
    return {"U": U, "V": V, "z": z}, centered


def score(X, time, event, strata):
    """Return float64 U, V, z, each [C] or [S,C] for X[P,C] or X[S,P,C].

    Outcomes are shared [P] vectors. U is the stratified Cox partial-likelihood
    score at beta=0 with Breslow ties. V=sum_s Sxx_s*Srr_s/(n_s-1) is its exact
    within-stratum permutation variance, not Cox information. A feature with no
    permutation variance has U=V=z=0. Event-free and singleton strata contribute 0.
    """
    X, time, event, groups = _inputs(X, time, event, strata)
    result, _ = _score(X, _residuals(time, event, groups), groups)
    return result


def _z_array(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim not in (1, 2) or not all(z.shape) or not np.isfinite(z).all():
        raise ValueError("z must be a finite nonempty [features] or [studies, features] array")
    return z


def maxT(z):
    """Maximum absolute randomisation-standardised score on the last axis."""
    return np.max(np.abs(_z_array(z)), axis=-1)


def threshold(design, null_studies):
    """Return an unconditional design cutoff from independent null z[S,C].

    ``design`` is the study-design dict, optionally containing ``alpha`` (0.05).
    The caller must generate fresh noisy null studies at this same design, using
    independent calibration seeds. ``null_studies`` may also be a score-result
    dict containing batched ``z``. It must not be permutations of one study.

    Reject only when maxT(z) > cutoff. The ceil((B+1)*(1-alpha)) order statistic
    gives a conservative finite-calibration rank rule for exchangeable future
    studies under the specified null model. Too few draws give +inf. Calibration
    is valid for that model/design, not conditional on an individual study.
    """
    if not isinstance(design, dict):
        raise TypeError("design must be a dict")
    alpha = float(design.get("alpha", 0.05))
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    z = _z_array(null_studies["z"] if isinstance(null_studies, dict) else null_studies)
    if z.ndim != 2 or len(z) < 2:
        raise ValueError("null_studies must contain at least two independent null z rows")
    maxima = maxT(z)
    k = ceil((len(maxima) + 1) * (1 - alpha))
    return float(np.partition(maxima, k - 1)[k - 1]) if k <= len(maxima) else float("inf")


def _components(z):
    p = np.sort(2 * ndtr(-np.abs(z)), axis=-1)
    m = z.shape[-1]
    p = np.clip(p, np.finfo(np.float64).tiny, 1)
    fraction = np.arange(1, m + 1, dtype=np.float64) / m
    valid = p <= 0.5
    hc = np.divide(
        np.sqrt(m) * (fraction - p), np.sqrt(p * (1 - p)),
        out=np.zeros_like(p), where=valid,
    )
    return np.column_stack((maxT(z), np.sum(z * z, axis=-1), np.maximum(0, hc.max(axis=-1))))


def _joint_test(rows):
    components = _components(rows)
    # Include mathematical ties despite dot-product roundoff after row shuffles.
    p = np.empty_like(components)
    for j in range(components.shape[1]):
        values = components[:, j]
        tolerance = 16 * np.finfo(np.float64).eps * np.maximum(1, np.abs(values))
        p[:, j] = (len(rows) - np.searchsorted(np.sort(values), values - tolerance)) / len(rows)
    min_p = p.min(axis=1)
    return {
        "statistics": dict(zip(("maxT", "sum_z2", "HC"), components[0])),
        "component_p": dict(zip(("maxT", "sum_z2", "HC"), p[0])),
        "p": float(np.mean(min_p <= min_p[0])),
    }


def omnibus(z, null_z):
    """Jointly test maxT, sum(z²), HC using shared null rows [B,C].

    All rows, including observed z[C], are ranked symmetrically and the minimum
    component p is calibrated jointly. Null rows must be exchangeable with the
    observed row under the null (independent design-null studies, or joint
    within-stratum shuffles). Normal tails only define the HC statistic; its
    inference comes from empirical calibration. This is an exploratory
    conditional association when the rows are conditional permutations.
    """
    z, null_z = _z_array(z), _z_array(null_z)
    if z.ndim != 1 or null_z.ndim != 2 or null_z.shape[1] != len(z):
        raise ValueError("expected observed z[C] and null_z[B,C] with matching features")
    return _joint_test(np.vstack((z, null_z)))


def permutation_test(X, time, event, strata, *, n_resamples=9999, seed=0, exact=False, max_exact=100000):
    """Conditional single-step maxT and joint omnibus for one study.

    Shuffle outcome-pair residuals within strata, with one shared shuffle for all
    columns. Exact mode enumerates every index permutation (including identity)
    with equal weight and refuses more than max_exact combinations. Monte Carlo
    mode uses the observed row plus n_resamples draws and the +1 tail rule.
    """
    X, time, event, groups = _inputs(X, time, event, strata)
    if X.ndim != 2:
        raise ValueError("permutation_test accepts one study X[P,C]")
    r = _residuals(time, event, groups)
    observed, centered = _score(X, r, groups)
    sd = np.sqrt(observed["V"])
    if exact:
        total = 1
        for ix in groups:
            total *= factorial(len(ix))
            if total > max_exact:
                raise ValueError("exact permutation count exceeds max_exact; use Monte Carlo")
        draws = product(*(permutations(ix) for ix in groups))
        rows = np.empty((total, X.shape[1]), dtype=np.float64)
    else:
        if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
            raise ValueError("n_resamples must be a positive integer")
        rng = np.random.default_rng(seed)
        draws = (tuple(rng.permutation(ix) for ix in groups) for _ in range(n_resamples))
        rows = np.empty((n_resamples + 1, X.shape[1]), dtype=np.float64)
        rows[0] = observed["z"]
    shuffled = np.empty_like(r)
    for row, perm in enumerate(draws, start=0 if exact else 1):
        for ix, order in zip(groups, perm):
            shuffled[ix] = r[list(order)]
        U = np.einsum("pc,p->c", centered, shuffled)
        rows[row] = np.divide(U, sd, out=np.zeros_like(U), where=sd > 0)
    # The first exact tuple is the identity and must use the observed arithmetic.
    rows[0] = observed["z"]
    maxima = maxT(rows)
    observed_max = maxT(observed["z"])
    tolerance = 16 * np.finfo(np.float64).eps * max(1, observed_max)
    return {
        **observed,
        "maxT": float(observed_max),
        "p": float(np.mean(maxima >= observed_max - tolerance)),
        "adjusted_p": np.mean(maxima[:, None] >= np.abs(observed["z"]) - tolerance, axis=0),
        "omnibus": _joint_test(rows),
        "null_z": rows if exact else rows[1:],
        "exact": bool(exact),
    }
