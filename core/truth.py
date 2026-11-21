"""The truth layer: distance-sorted neighbour lists, true counts and region sums (float64, CPU, once)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from core.grammar import RADII_UM, features, features_v2, grammar

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


def truth_expression_sums(xy: np.ndarray, types: np.ndarray, channels: np.ndarray, K: int):
    """Independent cKDTree counts (n, R, K) and neighbour channel sums (n, R, K, M) float64."""
    n = len(xy)
    ch64 = np.asarray(channels, dtype=np.float64)
    M = ch64.shape[1]
    tree = cKDTree(xy)
    counts = np.zeros((n, R, K), np.int64)
    ch_sums = np.zeros((n, R, K, M), np.float64)
    for r, rad in enumerate(RADII_UM):
        pairs = tree.query_pairs(rad, output_type="ndarray")
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            d = np.sqrt(((xy[a] - xy[b]) ** 2).sum(1))
            a, b = a[d <= rad], b[d <= rad]
            rows = np.concatenate([a, b])
            nbrs = np.concatenate([b, a])
            nbt = types[nbrs]
            counts[:, r, :] = np.bincount(rows * K + nbt, minlength=n * K).reshape(n, K)
            np.add.at(ch_sums[:, r], (rows, nbt), ch64[nbrs])
    return counts, ch_sums


def region_sums_v2(counts: np.ndarray, ch_sums: np.ndarray, types: np.ndarray, visible: np.ndarray,
                   focal: list[int], K: int):
    """Shared v1+v2 region aggregator: (S_counts (T,K,R,2), S_expr (T,K,R,M), m (T,))."""
    S_counts, m = region_sums(counts, types, visible, focal, K)
    T = len(focal)
    M = ch_sums.shape[-1]
    S_expr = np.zeros((T, K, R, M), dtype=np.float64)
    if int(visible.sum()) == 0:
        return S_counts, S_expr, m
    t_of = np.full(K, -1)
    t_of[focal] = np.arange(T)
    foc = np.flatnonzero(visible & (t_of[types] >= 0))
    if len(foc):
        cnt = counts[foc, ..., None]
        mean_ch = np.divide(ch_sums[foc], cnt, out=np.zeros_like(ch_sums[foc], dtype=np.float64), where=cnt > 0)
        np.add.at(S_expr, t_of[types[foc]], mean_ch.transpose(0, 2, 1, 3))
    return S_counts, S_expr, m


def truth_region_v2(xy: np.ndarray, types: np.ndarray, channels: np.ndarray, focal: list[int], K: int):
    """True v1+v2 region output: full region, true types, unperturbed channels."""
    counts, ch_sums = truth_expression_sums(xy, types, channels, K)
    return region_sums_v2(counts, ch_sums, types, np.ones(len(xy), bool), focal, K)


def layout(cells: dict) -> dict:
    """Index helpers for a contract cells dict (cells grouped by region, in region order)."""
    rid = np.asarray(cells["region_id"])
    por = np.asarray(cells["patient_of_region"])
    if np.any(np.diff(rid) < 0):
        raise ValueError("cells must be grouped by region, in region order")
    ptr = np.searchsorted(rid, np.arange(len(por) + 1))
    # shortcut: without type_names every type is focal and K = max type + 1; real cohorts carry type_names.
    names = cells.get("type_names") or [f"type {i}" for i in range(int(np.max(cells["cell_type"])) + 1)]
    focal = grammar(names)["focal_types"] if "type_names" in cells else list(range(len(names)))
    P = len(cells["time"]) if "time" in cells else int(por.max()) + 1
    x, y, ty = cells["x"], cells["y"], np.asarray(cells["cell_type"])
    ch = cells.get("channels")
    out = {"P": P, "K": len(names), "focal": focal,
           "regions_of": [np.flatnonzero(por == p) for p in range(P)],
           "xy": lambda r: np.stack([x[ptr[r]:ptr[r + 1]], y[ptr[r]:ptr[r + 1]]], 1).astype(np.float64),
           "types": lambda r: ty[ptr[r]:ptr[r + 1]]}
    if ch is not None:
        out["channels"] = lambda r: ch[ptr[r]:ptr[r + 1]]
        out["M"] = int(ch.shape[1])
    return out


def truth(cells: dict) -> np.ndarray:
    """The truth layer X float64 [P, C]: every region, full field, true types, patient index order."""
    L = layout(cells)
    T, K = len(L["focal"]), L["K"]
    S = np.zeros((L["P"], T, K, R, 2))
    m = np.zeros((L["P"], T), np.int64)
    for p, rs in enumerate(L["regions_of"]):
        for r in rs:
            s_, m_ = truth_region(L["xy"](r), L["types"](r), L["focal"], K)
            S[p] += s_
            m[p] += m_
    return features(S, m)


def truth_v2(cells: dict) -> np.ndarray:
    """The v1+v2 truth layer X float64 [P, C_v1 + C_expr] from true types and z-scored channels."""
    if "channels" not in cells:
        raise ValueError("cells must provide channels for truth_v2")
    L = layout(cells)
    T, K, M = len(L["focal"]), L["K"], L["M"]
    S_c = np.zeros((L["P"], T, K, R, 2))
    S_e = np.zeros((L["P"], T, K, R, M))
    m = np.zeros((L["P"], T), np.int64)
    for p, rs in enumerate(L["regions_of"]):
        for r in rs:
            sc, se, m_ = truth_region_v2(L["xy"](r), L["types"](r), L["channels"](r), L["focal"], K)
            S_c[p] += sc
            S_e[p] += se
            m[p] += m_
    return features_v2(S_c, S_e, m)
