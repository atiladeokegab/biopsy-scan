"""Shared input checks and acquisition draws for the native count backends."""

import hashlib

import numpy as np


def check_types(observed, K):
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError("K must be a positive integer")
    observed = np.asarray(observed)
    if observed.ndim != 1 or observed.dtype.kind not in "iu":
        raise ValueError("observed types must be a one-dimensional integer array")
    if np.any(observed < 0) or np.any(observed >= K):
        raise ValueError("observed types must be in [0, K)")
    return observed


def count_inputs(nbr, observed, visible, K):
    observed, visible = check_types(observed, K), np.asarray(visible)
    n = len(observed)
    if visible.shape != (n,) or visible.dtype != np.bool_:
        raise ValueError("visible must be a boolean array with one entry per cell")
    ptr, idx, ends = (np.asarray(nbr[key]) for key in ("ptr", "idx", "rad_end"))
    if (ptr.shape != (n + 1,) or idx.ndim != 1 or ends.shape != (n, 3)
            or any(a.dtype.kind not in "iu" for a in (ptr, idx, ends))):
        raise ValueError("CSR arrays must have integral shapes ptr[n+1], idx[E], rad_end[n,3]")
    if ptr[0] != 0 or ptr[-1] != len(idx) or np.any(ptr[1:] < ptr[:-1]):
        raise ValueError("CSR row offsets must start at zero and end at the edge count")
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError("CSR neighbour indices must be in [0, n)")
    if (np.any(ends < ptr[:-1, None]) or np.any(ends > ptr[1:, None])
            or np.any(ends[:, 1:] < ends[:, :-1])):
        raise ValueError("radius offsets must be ordered inside their CSR row")
    return tuple(np.ascontiguousarray(a) for a in (ptr, idx, ends, observed, visible))


def check_channels(channels, n):
    ch = np.asarray(channels)
    if ch.ndim != 2 or ch.shape[0] != n or ch.shape[1] < 1:
        raise ValueError("channels must be a 2D array with shape [cells, M>=1]")
    if ch.dtype.kind == "b" or ch.dtype.kind not in "iuf":
        raise ValueError("channels must have numeric dtype")
    ch64 = np.ascontiguousarray(ch, dtype=np.float64)
    if not np.isfinite(ch64).all():
        raise ValueError("channels must contain only finite values")
    return ch64


def channels_digest(channels_f64: np.ndarray) -> bytes:
    """Exact content digest for prepared v2 channel snapshots."""
    return hashlib.sha256(np.ascontiguousarray(channels_f64, dtype=np.float64).tobytes()).digest()


def region(counter, nbr, xy, types, focal, K, crop, eps, M, u, rng):
    from core.reimage_ref import confusion_matrix, crop_mask, miscall
    from core.truth import region_sums

    xy, types = np.asarray(xy), check_types(types, K)
    if xy.shape != (len(types), 2) or not np.isfinite(xy).all():
        raise ValueError("xy must be finite with shape [cells, 2]")
    if M is None:
        M = confusion_matrix(K, eps)
    M = np.asarray(M, dtype=np.float64)
    if (M.shape != (K, K) or not np.isfinite(M).all() or (M < 0).any()
            or not np.allclose(M.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("M must be a finite row-stochastic K by K matrix")
    if u is None:
        u = (rng or np.random.default_rng()).random(len(types))
    u = np.asarray(u, dtype=np.float64)
    if u.shape != (len(types),) or not np.isfinite(u).all() or (u < 0).any() or (u >= 1).any():
        raise ValueError("u must contain one finite draw in [0, 1) per cell")
    if (any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer))
            or t < 0 or t >= K for t in focal) or len(set(focal)) != len(focal)):
        raise ValueError("focal must contain distinct type indices in [0, K)")
    if crop is not None and (len(crop) != 3 or not np.isfinite(crop).all() or crop[2] <= 0):
        raise ValueError("crop must be a finite (x0, y0, positive side) tuple")
    observed = miscall(types, M, u)
    visible = crop_mask(xy, crop)
    return region_sums(counter(nbr, observed, visible, K), observed, visible, focal, K)


def region_v2(counter_v2, nbr, xy, types, channels, focal, K, crop, eps, M, u, rng):
    from core.reimage_ref import confusion_matrix, crop_mask, miscall
    from core.truth import region_sums_v2

    xy, types = np.asarray(xy), check_types(types, K)
    channels = check_channels(channels, len(types))
    if xy.shape != (len(types), 2) or not np.isfinite(xy).all():
        raise ValueError("xy must be finite with shape [cells, 2]")
    if M is None:
        M = confusion_matrix(K, eps)
    M = np.asarray(M, dtype=np.float64)
    if (M.shape != (K, K) or not np.isfinite(M).all() or (M < 0).any()
            or not np.allclose(M.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("M must be a finite row-stochastic K by K matrix")
    if u is None:
        u = (rng or np.random.default_rng()).random(len(types))
    u = np.asarray(u, dtype=np.float64)
    if u.shape != (len(types),) or not np.isfinite(u).all() or (u < 0).any() or (u >= 1).any():
        raise ValueError("u must contain one finite draw in [0, 1) per cell")
    if (any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer))
            or t < 0 or t >= K for t in focal) or len(set(focal)) != len(focal)):
        raise ValueError("focal must contain distinct type indices in [0, K)")
    if crop is not None and (len(crop) != 3 or not np.isfinite(crop).all() or crop[2] <= 0):
        raise ValueError("crop must be a finite (x0, y0, positive side) tuple")
    observed = miscall(types, M, u)
    visible = crop_mask(xy, crop)
    counts, ch_sums = counter_v2(nbr, observed, visible, channels, K)
    return region_sums_v2(counts, ch_sums, observed, visible, focal, K)
