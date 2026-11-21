"""The truth layer: distance-sorted neighbour lists, true counts and region sums (float64, CPU, once)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from core.grammar import RADII_UM

R = len(RADII_UM)


def neighbours(xy: np.ndarray, rmax: float = RADII_UM[-1]) -> dict:
    """CSR neighbour lists of one region, local indices, each row sorted by (distance, index), self excluded.

    idx int32 (E,), dist float64 (E,), ptr int64 (n+1,), rad_end int64 (n, R): row i's neighbours
    within RADII_UM[r] are idx[ptr[i]:rad_end[i, r]].
    """
    n = len(xy)
    tree = cKDTree(xy)
    lists = tree.query_ball_point(xy, rmax)
    lens = np.fromiter((len(a) for a in lists), np.int64, n)
    i = np.repeat(np.arange(n), lens)
    j = np.concatenate([np.asarray(a, np.int64) for a in lists]) if n else np.zeros(0, np.int64)
    keep = i != j
    i, j = i[keep], j[keep]
    d = np.sqrt(((xy[i] - xy[j]) ** 2).sum(1))
    order = np.lexsort((j, d, i))
    i, j, d = i[order], j[order], d[order]
    ptr = np.concatenate([[0], np.cumsum(np.bincount(i, minlength=n))]).astype(np.int64)
    rad_end = np.stack([ptr[:-1] + np.bincount(i[d <= r], minlength=n) for r in RADII_UM], 1)
    return {"idx": j.astype(np.int32), "dist": d, "ptr": ptr, "rad_end": rad_end}


def truth_counts(xy: np.ndarray, types: np.ndarray, K: int) -> np.ndarray:
    """(n, R, K) int64: neighbours of each type within each radius, by direct ball queries per radius.

    Deliberately independent of neighbours(): the re-imager is checked against this.
    """
    n = len(xy)
    tree = cKDTree(xy)
    out = np.zeros((n, R, K), np.int64)
    for r, rad in enumerate(RADII_UM):
        pairs = tree.query_pairs(rad, output_type="ndarray")
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            d = np.sqrt(((xy[a] - xy[b]) ** 2).sum(1))
            a, b = a[d <= rad], b[d <= rad]  # same distance rule as neighbours()
            rows = np.concatenate([a, b])
            nbt = types[np.concatenate([b, a])]
            out[:, r, :] = np.bincount(rows * K + nbt, minlength=n * K).reshape(n, K)
    return out


def region_sums(counts: np.ndarray, types: np.ndarray, visible: np.ndarray, focal: list[int], K: int):
    """Region output from per-cell counts: S (T, K, R, 2) float64 summed over focal cells, m (T,) int64.

    Shared by the truth layer and every re-imager, so equal counts give bit-equal sums.
    """
    T = len(focal)
    S = np.zeros((T, K, R, 2))
    m = np.zeros(T, np.int64)
    nvis = int(visible.sum())
    if nvis == 0:
        return S, m
    p = np.bincount(types[visible], minlength=K) / nvis
    t_of = np.full(K, -1)
    t_of[focal] = np.arange(T)
    foc = np.flatnonzero(visible & (t_of[types] >= 0))
    c = counts[foc].astype(np.float64)  # (F, R, K)
    tot = c.sum(-1, keepdims=True)
    frac = np.divide(c, tot, out=np.zeros_like(c), where=tot > 0)
    enr = np.divide(frac, p, out=np.zeros_like(frac), where=p > 0)
    t = t_of[types[foc]]
    np.add.at(S[..., 0], t, frac.transpose(0, 2, 1))
    np.add.at(S[..., 1], t, enr.transpose(0, 2, 1))
    m += np.bincount(t, minlength=T)
    return S, m


def truth_region(xy: np.ndarray, types: np.ndarray, focal: list[int], K: int):
    """True region output: full region, true types."""
    return region_sums(truth_counts(xy, types, K), types, np.ones(len(xy), bool), focal, K)
