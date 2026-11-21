"""Clinical-null Cox score screen: every candidate is a column of X (R-15, C-003).

Reference statistic (C-003 R-15). For candidate j the analysis set is the patients
with a finite X_j, complete clinical C, an endpoint and a stratum. On that set a
stratified Cox model on C alone (Breslow ties) is fitted with statsmodels PHReg,
refitted once per distinct complete-case mask. U_j is the Cox score for beta_j at
(beta_C-hat, 0), V_j = I_jj - I_jC I_CC^-1 I_Cj the nuisance-adjusted information
and z_j = U_j / sqrt(V_j). Breslow ties: tied deaths share one risk set, and a
patient censored at a death time is at risk at it. Zero-covariate mode (C = none)
is the same statistic with beta_C empty, so V is the plain Cox information, not the
legacy permutation variance of ``stats.reference``.

Everything per candidate is a matrix product with the risk operator M[b,i,l]
(death i's weight on patient l in resample b): U = X'm, with m the martingale
residuals, and I_AB = A' diag(a) B - (MA)'(MB), with a = M'1.

Adjusted null: a clinical-null parametric bootstrap. Each resample draws event
times from the fitted stratified Breslow model on C and censoring times from a
Breslow Cox censoring model on C within strata (censored at the stratum's last
follow-up), refits the null model per mask and recomputes U, V and z for every
candidate; maxT is the largest |z|. Assumptions: proportional hazards in C, and
censoring independent of X given C and stratum.
"""

import hashlib
import warnings

import numpy as np
from statsmodels.duration.hazard_regression import PHReg

from . import reference

NULL_METHOD = {
    "name": "clinical-null parametric bootstrap",
    "events": "Breslow stratified Cox model on C, fitted on patients with an endpoint, stratum and complete C",
    "censoring": "Breslow stratified Cox model on C for the censoring times, truncated at each stratum's last follow-up",
    "refit": "null model refitted per resample and per complete-case mask; U, V, z recomputed for every candidate",
    "assumptions": [
        "proportional hazards in C within strata",
        "censoring independent of X given C and stratum",
        "no candidate effect given C (the null being tested)",
    ],
}


def get_backend(name="auto"):
    """Array module: cupy when a CUDA device is present (auto), else numpy."""
    if name not in ("auto", "numpy", "cupy"):
        raise ValueError("backend must be 'auto', 'numpy' or 'cupy'")
    if name != "numpy":
        try:
            import cupy

            if cupy.cuda.runtime.getDeviceCount() > 0:
                return cupy
        except Exception:  # no cupy, no driver or no device
            pass
        if name == "cupy":
            raise RuntimeError("cupy with a CUDA device is unavailable; use backend='numpy'")
    return np


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _risk(xp, t, e, w, same):
    """Breslow risk operator for outcomes t, e and weights w, each [B,n]."""
    R = (t[:, None, :] >= t[:, :, None]) & same
    S0 = (R * w[:, None, :]).sum(-1)
    M = R * (e / S0)[:, :, None] * w[:, None, :]
    a = M.sum(1)
    return M, e - a, a


def _info(xp, A, Bm, M, a):
    """I_AB[b] = A' diag(a_b) B - (M_b A)'(M_b B) for A[n,p], B[n,q] -> [B,p,q]."""
    return xp.matmul(A.T[None], a[:, :, None] * Bm[None]) - xp.matmul(
        xp.matmul(M, A).transpose(0, 2, 1), xp.matmul(M, Bm))


def _stats(xp, X, C, M, m, a):
    """U, V, z [B,p] for centred X[n,p] given clinical C[n,q] at the null fit."""
    U = m @ X  # one dense product X'[m_1 .. m_B]
    MX = xp.matmul(M, X)
    V = a @ (X * X) - (MX * MX).sum(1)
    if C.shape[1]:
        Ixc = _info(xp, X, C, M, a)
        Icc = _info(xp, C, C, M, a)
        V = V - (Ixc * xp.linalg.solve(Icc, Ixc.transpose(0, 2, 1)).transpose(0, 2, 1)).sum(-1)
    scale = 1e-12 * xp.maximum(1.0, a @ (X * X))  # roundoff floor for constant columns
    good = V > scale
    z = xp.where(good, U / xp.sqrt(xp.where(good, V, 1.0)), 0.0)
    return U, xp.where(good, V, 0.0), z


def _newton(xp, C, t, e, same, iters=50, tol=1e-9):
    """Batched Breslow Cox fit of the null model on C for outcomes [B,n]."""
    beta = xp.zeros((t.shape[0], C.shape[1]))
    for _ in range(iters):
        M, m, a = _risk(xp, t, e, xp.exp(beta @ C.T), same)
        step = xp.linalg.solve(_info(xp, C, C, M, a), (m @ C)[:, :, None])[:, :, 0]
        step = xp.clip(step, -5.0, 5.0)  # monotone-likelihood resamples drift instead of overflowing
        beta = beta + step
        if float(xp.abs(step).max()) < tol:
            break
    return beta


def _phreg(C, t, e, codes):
    """Observed null fit with statsmodels PHReg (Breslow ties); empty C gives []."""
    if not C.shape[1] or not e.any():
        return np.zeros(C.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = PHReg(t, C, status=e, strata=codes, ties="breslow").fit(disp=False)
    return np.asarray(fit.params, dtype=np.float64)


def _breslow_steps(t, e, w, codes):
    """Per stratum: distinct death times and the Breslow cumulative baseline hazard."""
    steps = []
    for s in range(codes.max() + 1):
        ix = codes == s
        deaths = np.unique(t[ix & (e > 0)])
        jumps = [e[ix & (t == d)].sum() / w[ix & (t >= d)].sum() for d in deaths]
        steps.append((deaths, np.cumsum(jumps)))
    return steps


def _draw(steps, w, codes, exp_draws):
    """First step time with w * H(t) >= E for each patient; inf when never reached."""
    out = np.full(exp_draws.shape, np.inf)
    for s, (times, H) in enumerate(steps):
        ix = codes == s
        if len(times):
            k = np.searchsorted(H, exp_draws[:, ix] / w[ix], side="left")
            out[:, ix] = np.where(k < len(times), times[np.minimum(k, len(times) - 1)], np.inf)
    return out


def _inputs(X, time, event, clinical, strata):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not all(X.shape):
        raise ValueError("X must be a nonempty [patients, candidates] matrix")
    n = X.shape[0]
    if np.isinf(X).any():
        raise ValueError("X must not contain infinity; mark undefined values NaN")
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    C = np.zeros((n, 0)) if clinical is None else np.asarray(clinical, dtype=np.float64)
    C = C[:, None] if C.ndim == 1 else C
    strata = np.zeros(n) if strata is None else np.asarray(strata)
    if time.shape != (n,) or event.shape != (n,) or strata.shape != (n,) or C.shape[0] != n:
        raise ValueError("time, event, strata and clinical rows must match X's patients")
    if (time < 0).any():
        raise ValueError("time must be nonnegative")
    endpoint = np.isfinite(time) & np.isin(event, [0, 1])
    if np.isfinite(event).any() and not np.isin(event[np.isfinite(event)], [0, 1]).all():
        raise ValueError("event must be 0, 1 or NaN (missing)")
    stratum = np.isfinite(strata) if strata.dtype.kind == "f" else np.ones(n, dtype=bool)
    base = endpoint & stratum & np.isfinite(C).all(axis=1)
    codes = np.zeros(n, dtype=np.int64)
    if base.any():
        codes[base] = np.unique(strata[base], return_inverse=True)[1]
    return X, time, event, C, codes, base


def _fingerprint(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def screen(X, time, event, clinical=None, strata=None, n_perm=999, backend="auto", *,
           seed=0, names=None, clinical_names=None, block=64):
    """Score every candidate column of X against survival, adjusted for clinical C.

    X[P, J]: candidates, NaN where undefined. time[P], event[P] (NaN = missing
    endpoint), clinical[P, Q] or None (zero-covariate mode), strata[P] labels or
    None (one stratum). n_perm is B, the number of clinical-null parametric
    bootstrap resamples (0 skips the null). Returns float64 numpy U, V, z, the
    per-candidate analysis size n, maxT, its bootstrap p = (1 + #{maxT_b >= maxT})
    / (B + 1), single-step adjusted_p per candidate, null_maxT[B] and metadata.
    """
    xp = get_backend(backend)
    X, time, event, C, codes, base = _inputs(X, time, event, clinical, strata)
    P, J = X.shape
    if not isinstance(n_perm, (int, np.integer)) or n_perm < 0:
        raise ValueError("n_perm must be a nonnegative integer")
    keep_all = np.isfinite(X) & base[:, None]
    masks, inverse = np.unique(keep_all.T, axis=0, return_inverse=True)
    groups = []  # (columns, rows, device X, device C, device same, observed beta)
    for g, keep in enumerate(masks):
        cols = np.flatnonzero(inverse.ravel() == g)
        rows = np.flatnonzero(keep)
        if len(rows) < 2 or not event[rows].any():
            continue  # no information: U = V = z = 0
        Xg = X[np.ix_(rows, cols)]
        Xg = Xg - Xg.mean(0)  # U and I are shift-invariant; centring keeps roundoff small
        Cg = C[rows]
        Cg = Cg[:, np.ptp(Cg, axis=0) > 0] if len(Cg) else Cg  # a constant covariate has no effect
        Cg = Cg - Cg.mean(0)
        beta = _phreg(Cg, time[rows], event[rows], codes[rows])
        groups.append((cols, rows, xp.asarray(Xg), xp.asarray(Cg),
                       xp.asarray(codes[rows][:, None] == codes[rows][None]), beta))
    U, V, z = np.zeros(J), np.zeros(J), np.zeros(J)
    n = keep_all.sum(0)
    for cols, rows, Xg, Cg, same, beta in groups:
        w = xp.exp(Cg @ xp.asarray(beta))[None]
        M, m, a = _risk(xp, xp.asarray(time[rows])[None], xp.asarray(event[rows])[None], w, same)
        for out, value in zip((U, V, z), _stats(xp, Xg, Cg, M, m, a)):
            out[cols] = _host(value)[0]
    observed = float(np.abs(z).max())
    null_maxT = _bootstrap(xp, groups, time, event, C, codes, base, J, n_perm, seed, block)
    tol = 16 * np.finfo(np.float64).eps * max(1.0, observed)
    p = float((1 + np.sum(null_maxT >= observed - tol)) / (n_perm + 1)) if n_perm else float("nan")
    adjusted = ((1 + np.sum(null_maxT[:, None] >= np.abs(z)[None] - tol, axis=0)) / (n_perm + 1)
                if n_perm else np.full(J, np.nan))
    q = C.shape[1]
    return {
        "U": U, "V": V, "z": z, "n": n, "maxT": observed, "p": p,
        "adjusted_p": adjusted, "null_maxT": null_maxT,
        "metadata": {
            "statistic": "stratified Cox score at (beta_C-hat, 0), Breslow ties, nuisance-adjusted information",
            "missing": "complete cases per candidate; null model refitted once per distinct mask",
            "masks": len(masks),
            "clinical": {
                "mode": "zero-covariate" if q == 0 else "adjusted",
                "names": list(clinical_names) if clinical_names is not None else [f"C{k}" for k in range(q)],
                "shape": [P, q], "sha256": _fingerprint(C),
            },
            "candidates": {"count": J, "names": list(names) if names is not None else None,
                           "sha256": _fingerprint(np.nan_to_num(X, nan=np.pi))},
            "patients": {"rows": P, "analysis_base": int(base.sum()), "events": int(event[base].sum()),
                         "strata": int(codes[base].max() + 1) if base.any() else 0},
            "null": {**NULL_METHOD, "B": int(n_perm), "seed": seed} if n_perm else None,
            "backend": xp.__name__,
        },
    }


def _bootstrap(xp, groups, time, event, C, codes, base, J, B, seed, block):
    """maxT of z over all candidates for B clinical-null resamples."""
    if not B or not groups:
        return np.zeros(B)
    ix = np.flatnonzero(base)
    t, e, Cb, cb = time[ix], event[ix], C[ix], codes[ix]
    Cb = Cb[:, np.ptp(Cb, axis=0) > 0] - Cb[:, np.ptp(Cb, axis=0) > 0].mean(0)
    beta = _phreg(Cb, t, e, cb)
    gamma = _phreg(Cb, t, 1 - e, cb)
    w, wc = np.exp(Cb @ beta), np.exp(Cb @ gamma)
    death = _breslow_steps(t, e, w, cb)
    censor = _breslow_steps(t, 1 - e, wc, cb)
    end = np.array([t[cb == s].max() for s in range(cb.max() + 1)])[cb]
    position = np.full(len(time), -1)
    position[ix] = np.arange(len(ix))
    rng = np.random.default_rng(seed)
    maxima = np.empty(B)
    for start in range(0, B, block):
        size = min(block, B - start)
        T = _draw(death, w, cb, rng.standard_exponential((size, len(ix))))
        Cens = np.minimum(_draw(censor, wc, cb, rng.standard_exponential((size, len(ix)))), end)
        tb, eb = np.minimum(T, Cens), (T <= Cens).astype(np.float64)
        zmax = np.zeros(size)
        for cols, rows, Xg, Cg, same, _ in groups:
            tg, eg = xp.asarray(tb[:, position[rows]]), xp.asarray(eb[:, position[rows]])
            fitted = _newton(xp, Cg, tg, eg, same) if Cg.shape[1] else xp.zeros((size, 0))
            M, m, a = _risk(xp, tg, eg, xp.exp(fitted @ Cg.T), same)
            zmax = np.maximum(zmax, _host(xp.abs(_stats(xp, Xg, Cg, M, m, a)[2]).max(1)))
        maxima[start:start + size] = zmax
    return maxima


def permutation_maxT(X, time, event, strata, n_perm=9999, *, seed=0, backend="auto", block=1024):
    """Legacy zero-covariate maxT: X'[P_1 r, ..., P_B r] streamed in column blocks.

    Valid only where outcomes are exchangeable within strata (no clinical
    adjustment). Uses the randomisation-standardised statistic and the same
    permutation draws as ``stats.reference.permutation_test``; only the maxima
    are kept. Requires fully observed X.
    """
    xp = get_backend(backend)
    X, time, event, groups = reference._inputs(X, time, event, strata)
    if X.ndim != 2 or np.isnan(X).any():
        raise ValueError("permutation_maxT needs one fully observed X[P,C]")
    r = reference._residuals(time, event, groups)
    observed, centered = reference._score(X, r, groups)
    sd = np.sqrt(observed["V"])
    inv = xp.asarray(np.divide(1.0, sd, out=np.zeros_like(sd), where=sd > 0))
    Xt = xp.asarray(centered).T
    rng = np.random.default_rng(seed)
    maxima = np.empty(n_perm)
    for start in range(0, n_perm, block):
        size = min(block, n_perm - start)
        shuffled = np.empty((len(r), size))
        for b in range(size):
            for ix in groups:
                shuffled[ix, b] = r[rng.permutation(ix)]
        Z = (Xt @ xp.asarray(shuffled)) * inv[:, None]  # [C, block]
        maxima[start:start + size] = _host(xp.abs(Z).max(0))
    observed_max = float(np.abs(observed["z"]).max())
    tol = 16 * np.finfo(np.float64).eps * max(1, observed_max)
    return {**observed, "maxT": observed_max, "null_maxT": maxima,
            "p": float((1 + np.sum(maxima >= observed_max - tol)) / (n_perm + 1))}
