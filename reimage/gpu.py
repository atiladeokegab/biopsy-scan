"""CUDA CSR counting: a cooperative warp per focal cell, with resident adjacency."""

import numpy as np
from numba import cuda, float64, int64

from ._common import channels_digest, check_channels, check_types, count_inputs, region, region_v2


@cuda.jit
def _counts(ptr, idx, ends, observed, visible, K, out):
    i = cuda.blockIdx.x
    lane = cuda.threadIdx.x
    counts = cuda.shared.array(0, dtype=int64)
    for bin_index in range(lane, 3 * K, cuda.blockDim.x):
        counts[bin_index] = 0
    cuda.syncthreads()
    if visible[i]:
        for entry in range(ptr[i] + lane, ptr[i + 1], cuda.blockDim.x):
            j = idx[entry]
            if visible[j]:
                for r in range(3):
                    if entry < ends[i, r]:
                        cuda.atomic.add(counts, r * K + observed[j], 1)
    cuda.syncthreads()
    for bin_index in range(lane, 3 * K, cuda.blockDim.x):
        out[i, bin_index // K, bin_index % K] = counts[bin_index]


class _ResidentCounts:
    def __init__(self, nbr, K):
        if not cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select the CPU backend explicitly")
        n = len(nbr["ptr"]) - 1
        ptr, idx, ends, _, _ = count_inputs(
            nbr, np.zeros(n, dtype=np.int8), np.ones(n, dtype=bool), K,
        )
        self.backend = "gpu"
        self.n, self.K = n, int(K)
        self.source = nbr
        self.shared_bytes = 3 * self.K * np.dtype(np.int64).itemsize
        if (not cuda.config.ENABLE_CUDASIM
                and self.shared_bytes > cuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK):
            raise ValueError("K exceeds the device's shared-memory capacity")
        self.ptr, self.idx, self.ends = (cuda.to_device(a) for a in (ptr, idx, ends))

    def reimage_counts(self, observed, visible):
        """Reuse uploaded adjacency; include labels, mask and result transfers each time."""
        observed, visible = check_types(observed, self.K), np.asarray(visible)
        if len(observed) != self.n or visible.shape != (self.n,) or visible.dtype != np.bool_:
            raise ValueError("labels and boolean visibility must match the prepared region")
        if not self.n:
            return np.zeros((0, 3, self.K), dtype=np.int64)
        out = cuda.device_array((self.n, 3, self.K), dtype=np.int64)
        _counts[self.n, 32, 0, self.shared_bytes](
            self.ptr, self.idx, self.ends, cuda.to_device(np.ascontiguousarray(observed)),
            cuda.to_device(np.ascontiguousarray(visible)), self.K, out,
        )
        return out.copy_to_host()


def prepare(nbr, K):
    """Upload an immutable adjacency snapshot once for repeated region acquisitions."""
    return _ResidentCounts(nbr, K)


def reimage_counts(nbr, observed, visible, K):
    """Drop-in count entry point; includes adjacency upload and result download."""
    # Validate before allocating or launching native code, including the empty-region case.
    count_inputs(nbr, observed, visible, K)
    return prepare(nbr, K).reimage_counts(observed, visible)


def reimage_region(nbr, xy, types, focal, K, crop=None, eps=0.0, M=None, u=None, rng=None,
                   prepared=None):
    """Reference-compatible region output, optionally reusing prepare(nbr, K)."""
    if prepared is not None and (
        not isinstance(prepared, _ResidentCounts)
        or getattr(prepared, "backend", None) != "gpu"
        or prepared.K != K
        or prepared.n != len(types)
        or prepared.source is not nbr
    ):
        raise ValueError("prepared adjacency belongs to a different backend, region, or K")
    counter = reimage_counts if prepared is None else (
        lambda _nbr, observed, visible, _K: prepared.reimage_counts(observed, visible)
    )
    return region(counter, nbr, xy, types, focal, K, crop, eps, M, u, rng)


@cuda.jit
def _counts_and_channels(ptr, idx, ends, observed, visible, channels, K, M, out_counts, out_ch):
    """Single CSR warp-per-cell pass accumulating counts and M channels in exact CSR order."""
    i = cuda.blockIdx.x
    lane = cuda.threadIdx.x
    s_j = cuda.shared.array(32, dtype=int64)
    s_obs = cuda.shared.array(32, dtype=int64)
    s_rmin = cuda.shared.array(32, dtype=int64)
    for rk in range(lane, 3 * K, cuda.blockDim.x):
        out_counts[i, rk // K, rk % K] = 0
    for slot in range(lane, 3 * K * M, cuda.blockDim.x):
        rk = slot // M
        out_ch[i, rk // K, rk % K, slot % M] = 0.0
    cuda.syncthreads()
    if visible[i]:
        end_i = ptr[i + 1]
        e0 = ends[i, 0]
        e1 = ends[i, 1]
        e2 = ends[i, 2]
        for base_e in range(ptr[i], end_i, 32):
            rem = end_i - base_e
            batch_len = 32 if rem > 32 else rem
            if lane < batch_len:
                entry = base_e + lane
                j = int64(idx[entry])
                s_j[lane] = j
                s_obs[lane] = int64(observed[j])
                if visible[j]:
                    if entry < e0:
                        s_rmin[lane] = 0
                    elif entry < e1:
                        s_rmin[lane] = 1
                    elif entry < e2:
                        s_rmin[lane] = 2
                    else:
                        s_rmin[lane] = 3
                else:
                    s_rmin[lane] = 3
            cuda.syncthreads()
            if lane < 3:
                r = lane
                for u in range(batch_len):
                    if s_rmin[u] <= r:
                        out_counts[i, r, s_obs[u]] += 1
            for slot in range(lane, 3 * M, cuda.blockDim.x):
                r = slot // M
                c = slot % M
                for u in range(batch_len):
                    if s_rmin[u] <= r:
                        out_ch[i, r, s_obs[u], c] += float64(channels[s_j[u], c])
            cuda.syncthreads()


class _ResidentCountsAndChannels:
    def __init__(self, nbr, channels, K):
        if not cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select the CPU backend explicitly")
        n = len(nbr["ptr"]) - 1
        ptr, idx, ends, _, _ = count_inputs(
            nbr, np.zeros(n, dtype=np.int8), np.ones(n, dtype=bool), K,
        )
        ch = check_channels(channels, n)
        self.backend = "gpu"
        self.n, self.K, self.M = n, int(K), int(ch.shape[1])
        self.source = nbr
        self.channel_digest = channels_digest(ch)
        self.ptr, self.idx, self.ends, self.channels = (
            cuda.to_device(a) for a in (ptr, idx, ends, ch)
        )

    def matches(self, nbr, channels, K):
        """Return True iff cached GPU snapshot still matches nbr, K, and channel values."""
        ch = check_channels(channels, self.n)
        return (
            self.source is nbr
            and self.K == K
            and self.M == ch.shape[1]
            and self.channel_digest == channels_digest(ch)
        )

    def reimage_counts_and_channels(self, observed, visible):
        observed, visible = check_types(observed, self.K), np.asarray(visible)
        if len(observed) != self.n or visible.shape != (self.n,) or visible.dtype != np.bool_:
            raise ValueError("labels and boolean visibility must match the prepared region")
        if not self.n:
            return (
                np.zeros((0, 3, self.K), dtype=np.int64),
                np.zeros((0, 3, self.K, self.M), dtype=np.float64),
            )
        out_counts = cuda.device_array((self.n, 3, self.K), dtype=np.int64)
        out_ch = cuda.device_array((self.n, 3, self.K, self.M), dtype=np.float64)
        _counts_and_channels[self.n, 32](
            self.ptr,
            self.idx,
            self.ends,
            cuda.to_device(np.ascontiguousarray(observed)),
            cuda.to_device(np.ascontiguousarray(visible)),
            self.channels,
            self.K,
            self.M,
            out_counts,
            out_ch,
        )
        return out_counts.copy_to_host(), out_ch.copy_to_host()


def prepare_v2(nbr, channels, K):
    """Upload adjacency and fixed cell channels once for repeated v2 region acquisitions."""
    return _ResidentCountsAndChannels(nbr, channels, K)


def reimage_counts_and_channels(nbr, observed, visible, channels, K):
    """GPU v2 count and channel-sum entry point."""
    count_inputs(nbr, observed, visible, K)
    return prepare_v2(nbr, channels, K).reimage_counts_and_channels(observed, visible)


def reimage_region_v2(nbr, xy, types, channels, focal, K, crop=None, eps=0.0, M=None, u=None,
                      rng=None, prepared=None):
    """Reference-compatible v1+v2 GPU region output, optionally reusing prepare_v2."""
    if prepared is not None and (
        not isinstance(prepared, _ResidentCountsAndChannels)
        or getattr(prepared, "backend", None) != "gpu"
        or prepared.n != len(types)
        or not prepared.matches(nbr, channels, K)
    ):
        raise ValueError("prepared v2 snapshot belongs to a different backend, region, K, or channels")
    counter = reimage_counts_and_channels if prepared is None else (
        lambda _nbr, observed, visible, _channels, _K: prepared.reimage_counts_and_channels(observed, visible)
    )
    return region_v2(counter, nbr, xy, types, channels, focal, K, crop, eps, M, u, rng)
