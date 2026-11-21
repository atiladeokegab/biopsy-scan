"""Parallel Numba reference-equivalent counting: one CSR walk per visible cell."""

import numpy as np
from numba import njit, prange

from ._common import check_channels, check_types, count_inputs, region, region_v2


@njit(parallel=True, cache=True)
def _counts(ptr, idx, ends, observed, visible, K):
    out = np.zeros((len(observed), 3, K), dtype=np.int64)
    for i in prange(len(observed)):
        if visible[i]:
            for entry in range(ptr[i], ptr[i + 1]):
                j = idx[entry]
                if visible[j]:
                    for r in range(3):
                        if entry < ends[i, r]:
                            out[i, r, observed[j]] += 1
    return out


def reimage_counts(nbr, observed, visible, K):
    """Return int64[cells, 3 radii, K] counts, exactly as the NumPy oracle."""
    return _counts(*count_inputs(nbr, observed, visible, K), K)


class _PreparedCounts:
    def __init__(self, nbr, K):
        n = len(nbr["ptr"]) - 1
        self.ptr, self.idx, self.ends, _, _ = count_inputs(
            nbr, np.zeros(n, dtype=np.int8), np.ones(n, dtype=bool), K,
        )
        self.n, self.K, self.source = n, int(K), nbr

    def reimage_counts(self, observed, visible):
        observed, visible = check_types(observed, self.K), np.asarray(visible)
        if len(observed) != self.n or visible.shape != (self.n,) or visible.dtype != np.bool_:
            raise ValueError("labels and boolean visibility must match the prepared region")
        return _counts(self.ptr, self.idx, self.ends,
                       np.ascontiguousarray(observed), np.ascontiguousarray(visible), self.K)


def prepare(nbr, K):
    """Validate immutable adjacency once, matching the GPU preparation boundary."""
    return _PreparedCounts(nbr, K)


def reimage_region(nbr, xy, types, focal, K, crop=None, eps=0.0, M=None, u=None, rng=None,
                   prepared=None):
    """Reference-compatible acquisition; JIT cost occurs on the first call."""
    if prepared is not None and (prepared.K != K or prepared.n != len(types) or prepared.source is not nbr):
        raise ValueError("prepared adjacency belongs to a different region or K")
    counter = reimage_counts if prepared is None else (
        lambda _nbr, observed, visible, _K: prepared.reimage_counts(observed, visible)
    )
    return region(counter, nbr, xy, types, focal, K, crop, eps, M, u, rng)


@njit(parallel=True, cache=True)
def _counts_and_channels(ptr, idx, ends, observed, visible, channels, K, M):
    out_counts = np.zeros((len(observed), 3, K), dtype=np.int64)
    out_ch = np.zeros((len(observed), 3, K, M), dtype=np.float64)
    for i in prange(len(observed)):
        if visible[i]:
            for entry in range(ptr[i], ptr[i + 1]):
                j = idx[entry]
                if visible[j]:
                    obs_j = observed[j]
                    for r in range(3):
                        if entry < ends[i, r]:
                            out_counts[i, r, obs_j] += 1
                            for c in range(M):
                                out_ch[i, r, obs_j, c] += float(channels[j, c])
    return out_counts, out_ch


def reimage_counts_and_channels(nbr, observed, visible, channels, K):
    ptr, idx, ends, obs, vis = count_inputs(nbr, observed, visible, K)
    ch = check_channels(channels, len(obs))
    return _counts_and_channels(ptr, idx, ends, obs, vis, ch, K, ch.shape[1])


class _PreparedCountsAndChannels:
    def __init__(self, nbr, channels, K):
        n = len(nbr["ptr"]) - 1
        self.ptr, self.idx, self.ends, _, _ = count_inputs(
            nbr, np.zeros(n, dtype=np.int8), np.ones(n, dtype=bool), K,
        )
        self.channels = check_channels(channels, n)
        self.n, self.K, self.M, self.source = n, int(K), int(self.channels.shape[1]), nbr

    def reimage_counts_and_channels(self, observed, visible):
        observed, visible = check_types(observed, self.K), np.asarray(visible)
        if len(observed) != self.n or visible.shape != (self.n,) or visible.dtype != np.bool_:
            raise ValueError("labels and boolean visibility must match the prepared region")
        return _counts_and_channels(
            self.ptr, self.idx, self.ends,
            np.ascontiguousarray(observed), np.ascontiguousarray(visible),
            self.channels, self.K, self.M,
        )


def prepare_v2(nbr, channels, K):
    """Validate adjacency and channels once for repeated v2 CPU region acquisitions."""
    return _PreparedCountsAndChannels(nbr, channels, K)


def reimage_region_v2(nbr, xy, types, channels, focal, K, crop=None, eps=0.0, M=None, u=None,
                      rng=None, prepared=None):
    if prepared is not None and (
        prepared.K != K or prepared.n != len(types) or prepared.M != np.asarray(channels).shape[1]
        or prepared.source is not nbr
    ):
        raise ValueError("prepared v2 snapshot belongs to a different region, K, or channel count")
    counter = reimage_counts_and_channels if prepared is None else (
        lambda _nbr, observed, visible, _channels, _K: prepared.reimage_counts_and_channels(observed, visible)
    )
    return region_v2(counter, nbr, xy, types, channels, focal, K, crop, eps, M, u, rng)
