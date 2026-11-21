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
    if name not in ("auto", "numpy", "cupy", "gpu"):
        raise ValueError("backend must be 'auto', 'numpy', 'cupy' or 'gpu' (numba.cuda)")
    if name == "gpu":
        from .gpu_score import require_cuda

        require_cuda()  # raises without a device or the simulator; never falls back
        return np  # host arrays; the kernel moves X-dependent work to the device
    if name in ("auto", "cupy"):
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


def _cross(xp, Bm, M, a):
    """H[b] with I_BA[b] = H[b] @ A = B' diag(a_b) A - (M_b B)'(M_b A), for B[n,q] -> [B,q,n]."""
    MB = xp.ascontiguousarray(_rmul(M, Bm).transpose(0, 2, 1))
    return xp.ascontiguousarray((a[:, :, None] * Bm[None]).transpose(0, 2, 1)) - xp.matmul(MB, M)


def _rmul(M, A):
    """M[B,n,n] @ A[n,p] as one 2-D product (numpy's broadcast matmul skips BLAS)."""
    return (M.reshape(-1, M.shape[-1]) @ A).reshape(M.shape[:-1] + (A.shape[-1],))


def _stats(xp, X, C, M, m, a):
    """U, V, z [B,p] for centred X[n,p] given clinical C[n,q] at the null fit."""
    U = m @ X  # one dense product X'[m_1 .. m_B]
    MX = _rmul(M, X)
    Ixx = a @ (X * X) - (MX * MX).sum(1)
    Icx = _rmul(_cross(xp, C, M, a), X) if C.shape[1] else None  # [B,q,p]
    return _finish(xp, U, Ixx, Icx, M, a, C)


def _solve(xp, A, b):
    """Batched A x = b. A singular I_CC (a resample with no events has I_CC = 0) takes the
    pseudo-inverse, which is the Schur complement's generalised inverse and gives 0 when A = 0."""
    try:
        x = xp.linalg.solve(A, b)
        if bool(xp.isfinite(x).all()):
            return x
    except np.linalg.LinAlgError:
        pass
    return xp.asarray(np.linalg.pinv(_host(A), hermitian=True) @ _host(b))


def _finish(xp, U, Ixx, Icx, M, a, C):
    V = Ixx
    if C.shape[1]:
        V = V - (Icx * _solve(xp, _rmul(_cross(xp, C, M, a), C), Icx)).sum(1)
    good = V > 1e-10 * Ixx  # relative roundoff floor, so z does not depend on X's units
    z = xp.where(good, U / xp.sqrt(xp.where(good, V, 1.0)), 0.0)
    return U, xp.where(good, V, 0.0), z


def _scores(xp, gpu, X, C, same, t, e, w):
    """U, V, z [B,p] for one mask group's outcomes t, e [B,n] at null weights w [B,n]."""
    M, m, a = _risk(xp, t, e, w, same)
    if not gpu:
        return _stats(xp, X, C, M, m, a)
    from .gpu_score import moments  # X-dependent B x J x n^2 work on the device

    U, Ixx, Ixc = moments(X, C, t, e, w, same)
    return _finish(xp, U, Ixx, Ixc.transpose(0, 2, 1), M, a, C)


def _newton(xp, C, t, e, same, iters=50, tol=1e-9):
    """Batched Breslow Cox fit of the null model on C for outcomes [B,n]: (beta, converged[B]).

    A row with no events has zero score and information, so it stays at beta = 0 and
    contributes zero. Rows whose likelihood is monotone (no finite maximum) do not
    converge; callers count them and disclose the count."""
    beta = xp.zeros((t.shape[0], C.shape[1]))
    if not C.shape[1]:
        return beta, np.ones(t.shape[0], dtype=bool)
    for _ in range(iters):
        M, m, a = _risk(xp, t, e, xp.exp(beta @ C.T), same)
        step = _solve(xp, _rmul(_cross(xp, C, M, a), C), (m @ C)[:, :, None])[:, :, 0]
        step = xp.clip(step, -5.0, 5.0)  # monotone-likelihood resamples drift instead of overflowing
        beta = beta + step
        if float(xp.abs(step).max()) < tol:
            break
    return beta, _host(xp.abs(step).max(1) < tol)


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
    """Time where w * H(t) reaches E, H linear between steps (no artificial ties); inf past the last."""
    out = np.full(exp_draws.shape, np.inf)
    for s, (times, H) in enumerate(steps):
        ix = codes == s
        if len(times):
            target = exp_draws[:, ix] / w[ix]
            k = np.searchsorted(H, target, side="left")
            j = np.minimum(k, len(times) - 1)
            t0, H0 = np.append(0.0, times)[j], np.append(0.0, H)[j]
            T = t0 + (target - H0) / (H[j] - H0) * (times[j] - t0)
            out[:, ix] = np.where(k < len(times), T, np.inf)
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


def _centre_within_strata(Xg, codes):
    """U and I are invariant to a per-stratum shift of X; centring within strata keeps the moments
    free of cancellation, and a column constant within a stratum becomes exactly 0 there."""
    out = np.empty_like(Xg)
    for code in np.unique(codes):
        ix = codes == code
        sub = Xg[ix]
        centred = sub - sub.mean(0)
        centred[:, np.ptp(sub, axis=0) == 0] = 0.0
        out[ix] = centred
    return out


def _groups(xp, X, C, codes, keep_all, events):
    """Candidates grouped by analysis mask: (columns, rows, device X, device C, device same) per mask."""
    masks, inverse = np.unique(keep_all.T, axis=0, return_inverse=True)
    groups = []
    for g, keep in enumerate(masks):
        cols = np.flatnonzero(inverse.ravel() == g)
        rows = np.flatnonzero(keep)
        if len(rows) < 2 or not events[:, rows].any():
            continue  # no information: U = V = z = 0
        Xg = _centre_within_strata(X[np.ix_(rows, cols)], codes[rows])
        Cg = C[rows]
        Cg = Cg[:, np.ptp(Cg, axis=0) > 0] if len(Cg) else Cg  # a constant covariate has no effect
        Cg = Cg - Cg.mean(0)
        groups.append((cols, rows, xp.asarray(Xg), xp.asarray(Cg),
                       xp.asarray(codes[rows][:, None] == codes[rows][None])))
    return masks, groups


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
    xp, gpu = get_backend(backend), backend == "gpu"
    X, time, event, C, codes, base = _inputs(X, time, event, clinical, strata)
    P, J = X.shape
    if not isinstance(n_perm, (int, np.integer)) or n_perm < 0:
        raise ValueError("n_perm must be a nonnegative integer")
    keep_all = np.isfinite(X) & base[:, None]
    masks, groups = _groups(xp, X, C, codes, keep_all, event[None])
    for g in groups:  # observed fit: PHReg, after checking it has a finite maximum
        if g[3].shape[1] and not _newton(xp, g[3], xp.asarray(time[g[1]])[None], xp.asarray(event[g[1]])[None], g[4])[1][0]:
            raise ValueError("the observed Cox model on C has no finite maximum (monotone likelihood) in a mask group")
    groups = [(*g, _phreg(_host(g[3]), time[g[1]], event[g[1]], codes[g[1]])) for g in groups]
    U, V, z = np.zeros(J), np.zeros(J), np.zeros(J)
    n = keep_all.sum(0)
    for cols, rows, Xg, Cg, same, beta in groups:
        w = xp.exp(Cg @ xp.asarray(beta))[None]
        tg, eg = xp.asarray(time[rows])[None], xp.asarray(event[rows])[None]
        for out, value in zip((U, V, z), _scores(xp, gpu, Xg, Cg, same, tg, eg, w)):
            out[cols] = _host(value)[0]
    observed = float(np.abs(z).max())
    null_maxT, nonconverged = _bootstrap(xp, gpu, groups, time, event, C, codes, base, n_perm, seed, block)
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
            "null": {**NULL_METHOD, "B": int(n_perm), "seed": seed, "nonconverged": nonconverged} if n_perm else None,
            "backend": "gpu (numba.cuda)" if gpu else xp.__name__,
        },
    }


def screen_batch(X, time, event, clinical=None, strata=None, backend="auto", *, block=64):
    """The observed screen for S simulated studies on one cohort in one call.

    time[S, P] and event[S, P] are the studies' outcomes (finite, event 0 or 1);
    X, clinical and strata are shared, as in ``screen``. Each study gets its own
    null fit (batched Newton, equal to PHReg) per mask group. Returns float64
    U, V, z [S, J], maxT [S], n [J] and metadata; no resampling (n_perm = 0).
    """
    xp, gpu = get_backend(backend), backend == "gpu"
    time, event = np.atleast_2d(np.asarray(time, dtype=np.float64)), np.atleast_2d(np.asarray(event, dtype=np.float64))
    if time.shape != event.shape or not np.isfinite(time).all() or not np.isin(event, [0, 1]).all():
        raise ValueError("time and event must be finite [S, P] arrays with event 0 or 1")
    X, _, _, C, codes, base = _inputs(X, time[0], event[0], clinical, strata)
    if (time < 0).any():
        raise ValueError("time must be nonnegative")
    keep_all = np.isfinite(X) & base[:, None]
    masks, groups = _groups(xp, X, C, codes, keep_all, event)
    J = X.shape[1]
    U, V, z = (np.zeros((len(time), J)) for _ in range(3))
    failed = 0
    for start in range(0, len(time), block):
        part = slice(start, start + block)
        if groups:
            *values, bad = _refit_scores(xp, gpu, groups, time[part], event[part], np.arange(len(base)), J)
            for out, value in zip((U, V, z), values):
                out[part] = value
            failed += bad
    return {"U": U, "V": V, "z": z, "maxT": np.abs(z).max(1) if J else np.zeros(len(time)), "n": keep_all.sum(0),
            "metadata": {"studies": len(time), "masks": len(masks), "nonconverged": failed, "clinical": {"shape": list(C.shape)},
                         "backend": "gpu (numba.cuda)" if gpu else xp.__name__}}


def _null_draws(time, event, C, codes, base, B, seed, block):
    """Yield clinical-null resamples (time, event), each [block, n_base], over base rows."""
    ix = np.flatnonzero(base)
    t, e, Cb, cb = time[ix], event[ix], C[ix], codes[ix]
    Cb = Cb[:, np.ptp(Cb, axis=0) > 0]
    Cb = Cb - Cb.mean(0)
    same = cb[:, None] == cb[None]
    # Generating fits use the batched Newton, which equals PHReg (test_vs_statsmodels).
    w, wc = (np.exp(Cb @ _fit_or_raise(Cb, t, d, same)) if Cb.shape[1] and d.any()
             else np.ones(len(t)) for d in (e, 1 - e))
    death, censor = _breslow_steps(t, e, w, cb), _breslow_steps(t, 1 - e, wc, cb)
    end = np.array([t[cb == s].max() for s in range(cb.max() + 1)])[cb]
    rng = np.random.default_rng(seed)
    for start in range(0, B, block):
        size = min(block, B - start)
        T = _draw(death, w, cb, rng.standard_exponential((size, len(ix))))
        Cens = np.minimum(_draw(censor, wc, cb, rng.standard_exponential((size, len(ix)))), end)
        yield np.minimum(T, Cens), (T <= Cens).astype(np.float64)


def _fit_or_raise(C, t, e, same):
    beta, ok = _newton(np, C, t[None], e[None], same)
    if not ok[0]:
        raise ValueError("the generating Cox model on C has no finite maximum (monotone likelihood)")
    return beta[0]


def _bootstrap(xp, gpu, groups, time, event, C, codes, base, B, seed, block):
    """maxT of z over all candidates for B clinical-null resamples."""
    if not B or not groups:
        return np.zeros(B), 0
    position = np.full(len(time), -1)
    position[base] = np.arange(base.sum())
    maxima, failed = [], 0
    for tb, eb in _null_draws(time, event, C, codes, base, B, seed, block):
        U, V, z, bad = _refit_scores(xp, gpu, groups, tb, eb, position)
        maxima.append(np.abs(z).max(1))
        failed += bad
    return np.concatenate(maxima), failed


def _refit_scores(xp, gpu, groups, tb, eb, position, J=None):
    """U, V, z [S,J] for S outcome sets [S, *], refitting the null per set and per mask group."""
    J = max(int(g[0].max()) + 1 for g in groups) if J is None else J
    U, V, z = (np.zeros((len(tb), J)) for _ in range(3))
    failed = 0  # resample x mask fits without a finite maximum; their z stays as computed at the last step
    for cols, rows, Xg, Cg, same, *_ in groups:
        tg, eg = xp.asarray(tb[:, position[rows]]), xp.asarray(eb[:, position[rows]])
        fitted, ok = _newton(xp, Cg, tg, eg, same)
        failed += int((~ok).sum())
        for out, value in zip((U, V, z), _scores(xp, gpu, Xg, Cg, same, tg, eg, xp.exp(fitted @ Cg.T))):
            out[:, cols] = _host(value)
    return U, V, z, failed


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
