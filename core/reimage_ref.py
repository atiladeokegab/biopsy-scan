"""Float64 NumPy reference re-imager for one region: the oracle reimage/ kernels must match cell for cell.

One simulated look at a region: crop a square field of view, mis-call cell types at rate eps,
then recount every visible focal cell's neighbours from its distance-sorted list, skipping
neighbours the crop cannot see. With no crop and eps = 0 it reproduces the truth layer exactly.
"""
from __future__ import annotations

import numpy as np

from core.grammar import RADII_UM, features
from core.truth import R, layout, neighbours, region_sums


def confusion_matrix(K: int, eps: float) -> np.ndarray:
    """Default mis-call model: keep the type with prob 1 - eps, else a uniform other type."""
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be in [0, 1]")
    return np.full((K, K), eps / (K - 1)) + np.eye(K) * (1.0 - eps - eps / (K - 1))


def miscall(types: np.ndarray, M: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Observed types: cell i becomes the first k with cumsum(M[types[i]])[k] > u[i] (u uniform [0,1))."""
    cum = np.cumsum(M[types], axis=1)
    return np.minimum((cum <= u[:, None]).sum(1), len(M) - 1).astype(types.dtype)


def crop_mask(xy: np.ndarray, crop: tuple[float, float, float] | None) -> np.ndarray:
    """Visible cells: x0 <= x < x0 + s and y0 <= y < y0 + s; None = the full region."""
    if crop is None:
        return np.ones(len(xy), bool)
    x0, y0, s = crop
    return (xy[:, 0] >= x0) & (xy[:, 0] < x0 + s) & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + s)


def random_crop(xy: np.ndarray, s: float, rng: np.random.Generator) -> tuple[float, float, float] | None:
    """A field of view of side s placed uniformly inside the region's bounding box (None if s covers it)."""
    a = rng.random(2)  # always drawn, so the random stream does not depend on region size
    lo, hi = xy.min(0), xy.max(0)
    if s >= (hi - lo).max():
        return None
    x0, y0 = lo + a * np.maximum(hi - lo - s, 0)
    return (float(x0), float(y0), float(s))


def reimage_counts(nbr: dict, obs: np.ndarray, visible: np.ndarray, K: int) -> np.ndarray:
    """(n, R, K) int64 counts of observed neighbour types within each radius, visible neighbours only.

    The walk the kernel does: each row's sorted list, bin by first radius >= distance, cumulate.
    """
    n = len(obs)
    row = np.repeat(np.arange(n), np.diff(nbr["ptr"]))
    j = nbr["idx"]
    ok = visible[row] & visible[j]
    rbin = np.searchsorted(np.asarray(RADII_UM), nbr["dist"][ok], side="left")
    flat = (row[ok] * R + rbin) * K + obs[j[ok]]
    return np.cumsum(np.bincount(flat, minlength=n * R * K).reshape(n, R, K), axis=1)


def reimage_region(nbr: dict, xy: np.ndarray, types: np.ndarray, focal: list[int], K: int,
                   crop=None, eps: float = 0.0, M: np.ndarray | None = None, u: np.ndarray | None = None,
                   rng: np.random.Generator | None = None):
    """One look at one region -> (S (T, K, R, 2), m (T,)), the same shape as truth_region().

    u: the per-cell uniform draws for the mis-call (shared with the kernels so they match cell for
    cell); drawn from rng when None. M overrides the default confusion_matrix(K, eps).
    """
    M = confusion_matrix(K, eps) if M is None else M
    if u is None:
        u = (rng or np.random.default_rng()).random(len(types))
    obs = miscall(types, M, u)
    visible = crop_mask(xy, crop)
    return region_sums(reimage_counts(nbr, obs, visible, K), obs, visible, focal, K)


def draw(cells: dict, design: dict, seed) -> dict:
    """The study's random draws, in the contract's order: patients, regions, then per look crop and u.

    Returns {"patients": int [P'], "looks": per drawn patient a list of (region, crop, u)}.
    """
    L = layout(cells)
    rng = np.random.default_rng(seed)
    pats = design.get("patients")
    if pats is None:
        idx = np.arange(L["P"])
    elif np.ndim(pats) == 0:
        idx = rng.integers(0, L["P"], int(pats))
    else:
        idx = np.asarray(pats, np.int64)
    k, fov = design.get("regions_per_patient"), design.get("fov_um")
    regs = []
    for p in idx:
        rs = L["regions_of"][p]
        regs.append(rng.choice(rs, k, replace=False) if k is not None and k < len(rs) else rs)
    looks = []
    for rs in regs:
        row = []
        for r in rs:
            xy = L["xy"](r)
            crop = None if fov is None else random_crop(xy, float(fov), rng)
            row.append((int(r), crop, rng.random(len(xy))))
        looks.append(row)
    return {"patients": idx, "looks": looks}


def reimage(cells: dict, design: dict, seed, region_kernel=None) -> np.ndarray:
    """One simulated study -> X float64 [P', C], rows in drawn-patient order, NaN where no focal cell.

    This wrapper owns patient and region sampling and the RNG order; it calls
    region_kernel(nbr, xy, types, focal, K, crop=crop, M=M, u=u) -> (S, m) once per sampled region,
    in draw order (default reimage_region), so numba/GPU kernels drop in with identical sampling.
    Neighbour lists are built once per region and cached in cells["_nbr"].
    """
    region_kernel = region_kernel or reimage_region
    L = layout(cells)
    d = draw(cells, design, seed)
    M = confusion_matrix(L["K"], float(design.get("eps") or 0.0))
    T, K = len(L["focal"]), L["K"]
    S = np.zeros((len(d["patients"]), T, K, R, 2))
    m = np.zeros((len(d["patients"]), T), np.int64)
    cache = cells.setdefault("_nbr", {})
    for i, row in enumerate(d["looks"]):
        for r, crop, u in row:
            xy, ty = L["xy"](r), L["types"](r)
            if r not in cache:
                cache[r] = neighbours(xy)
            s_, m_ = region_kernel(cache[r], xy, ty, L["focal"], K, crop=crop, M=M, u=u)
            S[i] += s_
            m[i] += m_
    return features(S, m)
