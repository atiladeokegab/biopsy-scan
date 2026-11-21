"""Fused CPU re-imaging: mis-call and counting in one Numba pass (crop_mask's own mask) over visible focal cells only.

Bit-identical to cpu.reimage_region (counts every visible cell, then core.truth.region_sums): the same
IEEE operations, and each S entry receives its terms in increasing cell index, as np.add.at applies them.
"""

import numpy as np
from numba import njit

from core.reimage_ref import crop_mask
from core.truth import R
from . import cpu
from ._common import checked


@njit(cache=True)
def _region(ptr, idx, ends, types, u, Mcum, visible, t_of, T, K):
    n = len(types)
    obs = np.zeros(n, np.int64)
    vis = np.zeros(n, np.bool_)
    seen = np.zeros(K, np.int64)
    nvis = 0
    for i in range(n):
        if not visible[i]:
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
        lo = ptr[i]
        for r in range(R):  # row i's neighbours within radius r but not r - 1, then cumulate: exact integers
            if r > 0:
                cnt[r] = cnt[r - 1]
            for e in range(lo, ends[i, r]):
                j = idx[e]
                if vis[j]:
                    cnt[r, obs[j]] += 1
            lo = ends[i, r]
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
          or prepared.source is not nbr):
        raise ValueError("prepared adjacency belongs to a different region, K, or backend")
    if prepared.n != len(types):  # either path: the kernel indexes adjacency by cell without bounds checks
        raise ValueError("adjacency and labels describe regions of different sizes")
    xy, types, M, u = checked(xy, types, focal, K, crop, eps, M, u, rng)
    t_of = np.full(K, -1, np.int64)
    t_of[list(focal)] = np.arange(len(focal))
    visible = crop_mask(xy, crop)  # the reference's own dtype and scalar rules (a float32 xy stays float32)
    return _region(prepared.ptr, prepared.idx, prepared.ends, np.ascontiguousarray(types, np.int64), u,
                   np.cumsum(M, axis=1), visible, t_of, len(focal), K)
