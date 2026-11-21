"""Fused CPU re-imaging: mis-call, crop and counting in one Numba pass over visible focal cells only.

Bit-identical to cpu.reimage_region (counts every visible cell, then core.truth.region_sums): the same
IEEE operations, and each S entry receives its terms in increasing cell index, as np.add.at applies them.
"""

import numpy as np
from numba import njit

from core.truth import R
from . import cpu
from ._common import checked


@njit(cache=True)
def _region(ptr, idx, ends, types, u, Mcum, xy, has_crop, x0, y0, s, t_of, T, K):
    n = len(types)
    obs = np.zeros(n, np.int64)
    vis = np.zeros(n, np.bool_)
    seen = np.zeros(K, np.int64)
    nvis = 0
    for i in range(n):
        if has_crop and not (xy[i, 0] >= x0 and xy[i, 0] < x0 + s and xy[i, 1] >= y0 and xy[i, 1] < y0 + s):
            continue
        c = 0
        for k in range(K):  # core.reimage_ref.miscall: first k with cumsum(M[type])[k] > u
            if Mcum[types[i], k] <= u[i]:
                c += 1
        obs[i] = min(c, K - 1)
        vis[i] = True
        seen[obs[i]] += 1
        nvis += 1
    S = np.zeros((T, K, R, 2))
    m = np.zeros(T, np.int64)
    if nvis == 0:
        return S, m
    p = seen / nvis
    cnt = np.zeros((R, K), np.int64)
    for i in range(n):
        if not vis[i] or t_of[obs[i]] < 0:
            continue
        t = t_of[obs[i]]
        cnt[:] = 0
        for e in range(ptr[i], ptr[i + 1]):
            j = idx[e]
            if vis[j]:
                for r in range(R):
                    if e < ends[i, r]:
                        cnt[r, obs[j]] += 1
        m[t] += 1
        for r in range(R):
            tot = 0
            for k in range(K):
                tot += cnt[r, k]
            for k in range(K):
                frac = cnt[r, k] / np.float64(tot) if tot > 0 else 0.0
                S[t, k, r, 0] += frac
                S[t, k, r, 1] += frac / p[k] if p[k] > 0 else 0.0
    return S, m


def prepare(nbr, K):
    """The CPU backend's validated, contiguous adjacency (checked once per region)."""
    return cpu.prepare(nbr, K)


def reimage_region(nbr, xy, types, focal, K, crop=None, eps=0.0, M=None, u=None, rng=None, prepared=None):
    """Drop-in for cpu.reimage_region -> (S (T, K, R, 2), m (T,)); JIT cost occurs on the first call."""
    if prepared is None:
        prepared = cpu.prepare(nbr, K)
    elif (not isinstance(prepared, cpu._PreparedCounts) or prepared.backend != "cpu" or prepared.K != K
          or prepared.n != len(types) or prepared.source is not nbr):
        raise ValueError("prepared adjacency belongs to a different region, K, or backend")
    xy, types, M, u = checked(xy, types, focal, K, crop, eps, M, u, rng)
    t_of = np.full(K, -1, np.int64)
    t_of[list(focal)] = np.arange(len(focal))
    x0, y0, s = crop if crop is not None else (0.0, 0.0, 0.0)
    return _region(prepared.ptr, prepared.idx, prepared.ends, np.ascontiguousarray(types, np.int64), u,
                   np.cumsum(M, axis=1), np.ascontiguousarray(xy, np.float64), crop is not None,
                   float(x0), float(y0), float(s), t_of, len(focal), K)
