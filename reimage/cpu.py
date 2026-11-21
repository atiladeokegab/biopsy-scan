"""Parallel Numba reference-equivalent counting: one CSR walk per visible cell."""

import numpy as np
from numba import njit, prange

from ._common import check_types, count_inputs, region


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
