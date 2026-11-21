"""Cell-by-cell re-imaging backends and shared-core study batching."""

import numpy as np


def run(design, seeds):
    """Stack seeded patient-feature matrices using the core's exact sampling order."""
    from core import reimage_ref

    seeds = np.asarray(seeds)
    if seeds.ndim != 1 or not len(seeds) or seeds.dtype.kind not in "iu" or (seeds < 0).any():
        raise ValueError("seeds must be a nonempty one-dimensional sequence of nonnegative integers")
    if "cells" not in design:
        raise ValueError("design must provide cells")
    backend = design.get("backend", "numpy")
    if backend == "numpy":
        kernel = reimage_ref.reimage_region
    elif backend in ("cpu", "gpu"):
        if backend == "cpu":
            from . import cpu as module
        else:
            from . import gpu as module
        prepared = {}

        def kernel(nbr, xy, types, focal, K, **kwargs):
            key = (id(nbr), K)
            if key not in prepared:
                prepared[key] = module.prepare(nbr, K)
            return module.reimage_region(nbr, xy, types, focal, K,
                                         prepared=prepared[key], **kwargs)

    else:
        raise ValueError("backend must be numpy, cpu or gpu")
    return np.stack([
        reimage_ref.reimage(design["cells"], design, int(seed), region_kernel=kernel)
        for seed in seeds
    ])
