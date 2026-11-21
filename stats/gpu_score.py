"""numba.cuda kernel for the X-dependent parts of the R-15 screen: U, I_jj and I_jC.

One thread per (resample, candidate) sums Breslow risk-set moments over deaths, so
the B x J x n^2 work is on the device. The clinical-only pieces (I_CC and the null
refit) are q-sized and stay on the host. Runs under NUMBA_ENABLE_CUDASIM=1 in CI;
raises when neither a CUDA device nor the simulator is present.
"""

import numpy as np
from numba import cuda, float64

QMAX = 8  # clinical covariates per thread-local buffer


@cuda.jit
def _moments(X, C, q, t, e, w, same, U, Ixx, Ixc):
    j, b = cuda.grid(2)
    n, p = X.shape
    if j >= p or b >= t.shape[0]:
        return
    s1c = cuda.local.array(QMAX, float64)
    s2xc = cuda.local.array(QMAX, float64)
    acc = cuda.local.array(QMAX, float64)
    for k in range(q):
        acc[k] = 0.0
    u = 0.0
    ixx = 0.0
    for i in range(n):
        if e[b, i] == 0:
            continue
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for k in range(q):
            s1c[k] = 0.0
            s2xc[k] = 0.0
        for l in range(n):
            if same[i, l] and t[b, l] >= t[b, i]:
                wl = w[b, l]
                xl = X[l, j]
                s0 += wl
                s1 += wl * xl
                s2 += wl * xl * xl
                for k in range(q):
                    s1c[k] += wl * C[l, k]
                    s2xc[k] += wl * xl * C[l, k]
        xbar = s1 / s0
        u += X[i, j] - xbar
        ixx += s2 / s0 - xbar * xbar
        for k in range(q):
            acc[k] += s2xc[k] / s0 - xbar * s1c[k] / s0
    U[b, j] = u
    Ixx[b, j] = ixx
    for k in range(q):
        Ixc[b, j, k] = acc[k]


def require_cuda():
    if not cuda.is_available():
        raise RuntimeError("backend='gpu' needs a CUDA device (or NUMBA_ENABLE_CUDASIM=1 for the simulator)")


def moments(X, C, t, e, w, same):
    """U[B,J], I_jj[B,J], I_jC[B,J,q] for host float64 arrays; one kernel launch."""
    require_cuda()
    if C.shape[1] > QMAX:
        raise ValueError(f"backend='gpu' supports at most {QMAX} clinical covariates")
    B, (n, J), q = t.shape[0], X.shape, C.shape[1]
    U = cuda.device_array((B, J))
    Ixx = cuda.device_array((B, J))
    Ixc = cuda.device_array((B, J, max(q, 1)))
    threads = (32, 4)
    blocks = ((J + 31) // 32, (B + 3) // 4)
    Xd, Cd, td, ed, wd = (cuda.to_device(np.ascontiguousarray(a, dtype=np.float64))
                          for a in (X, C if q else np.zeros((n, 1)), t, e, w))  # no zero-width device arrays
    _moments[blocks, threads](Xd, Cd, q, td, ed, wd, cuda.to_device(np.ascontiguousarray(same, dtype=np.uint8)),
                              U, Ixx, Ixc)
    return U.copy_to_host(), Ixx.copy_to_host(), Ixc.copy_to_host()[:, :, :q]


SMAX = 16  # strata per study in the per-study kernel's thread-local buffers


@cuda.jit
def _study_moments(X, codes, t, e, U, Ixx):
    """U and I_jj for S studies that each have their own X[s] (no clinical covariates).

    Thread (j, s) uses study s's complete cases for candidate j (finite X[s, :, j]), centres
    them within strata (a column constant within a stratum becomes exactly 0 there, as on the
    CPU path) and sums Breslow risk-set moments over that study's deaths.
    """
    j, s = cuda.grid(2)
    S, n, p = X.shape
    if j >= p or s >= S:
        return
    tot = cuda.local.array(SMAX, float64)
    cnt = cuda.local.array(SMAX, float64)
    lo = cuda.local.array(SMAX, float64)
    hi = cuda.local.array(SMAX, float64)
    for g in range(SMAX):
        tot[g] = 0.0
        cnt[g] = 0.0
        lo[g] = np.inf
        hi[g] = -np.inf
    for l in range(n):
        x = X[s, l, j]
        if x == x:
            g = codes[s, l]
            tot[g] += x
            cnt[g] += 1.0
            lo[g] = min(lo[g], x)
            hi[g] = max(hi[g], x)
    for g in range(SMAX):
        tot[g] = tot[g] / cnt[g] if cnt[g] > 0 else 0.0
    u = 0.0
    ixx = 0.0
    for i in range(n):
        xi = X[s, i, j]
        if e[s, i] == 0 or xi != xi:
            continue
        gi = codes[s, i]
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for l in range(n):
            xl = X[s, l, j]
            if codes[s, l] == gi and t[s, l] >= t[s, i] and xl == xl:
                xc = xl - tot[gi] if hi[gi] > lo[gi] else 0.0
                s0 += 1.0
                s1 += xc
                s2 += xc * xc
        xbar = s1 / s0
        u += (xi - tot[gi] if hi[gi] > lo[gi] else 0.0) - xbar
        ixx += s2 / s0 - xbar * xbar
    U[s, j] = u
    Ixx[s, j] = ixx


def study_moments(X, codes, t, e):
    """U[S,J], I_jj[S,J] for host X[S,P,J] (NaN = missing), stratum codes[S,P] in 0..SMAX-1, t, e [S,P]."""
    require_cuda()
    S, n, J = X.shape
    if codes.size and codes.max() >= SMAX:
        raise ValueError(f"backend='gpu' supports at most {SMAX} strata per study")
    U = cuda.device_array((S, J))
    Ixx = cuda.device_array((S, J))
    threads = (32, 4)
    blocks = ((J + 31) // 32, (S + 3) // 4)
    Xd, td, ed = (cuda.to_device(np.ascontiguousarray(a, dtype=np.float64)) for a in (X, t, e))
    _study_moments[blocks, threads](Xd, cuda.to_device(np.ascontiguousarray(codes, dtype=np.int64)), td, ed, U, Ixx)
    return U.copy_to_host(), Ixx.copy_to_host()
