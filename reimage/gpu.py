"""CUDA CSR counting: a cooperative warp per focal cell, with resident adjacency."""

import numpy as np
from numba import cuda, int64

from ._common import check_types, count_inputs, region


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
    if prepared is not None and (prepared.K != K or prepared.n != len(types) or prepared.source is not nbr):
        raise ValueError("prepared adjacency belongs to a different region or K")
    counter = reimage_counts if prepared is None else (
        lambda _nbr, observed, visible, _K: prepared.reimage_counts(observed, visible)
    )
    return region(counter, nbr, xy, types, focal, K, crop, eps, M, u, rng)
