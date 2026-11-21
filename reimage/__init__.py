"""Cell-by-cell re-imaging backends and shared-core study batching."""

import numpy as np


def run(design, seeds, *, devices=None, _prepared=None):
    """Stack seeded patient-feature matrices (v1 default, or v1+v2 when grammar='v2')."""
    from core import reimage_ref

    seeds = np.asarray(seeds)
    if seeds.ndim != 1 or not len(seeds) or seeds.dtype.kind not in "iu" or (seeds < 0).any():
        raise ValueError("seeds must be a nonempty one-dimensional sequence of nonnegative integers")
    if "cells" not in design:
        raise ValueError("design must provide cells")
    grammar_ver = design.get("grammar", "v1")
    if grammar_ver not in ("v1", "v2"):
        raise ValueError("design grammar must be 'v1' or 'v2'")
    if grammar_ver == "v2" and "channels" not in design["cells"]:
        raise ValueError("design must provide cells with channels for grammar='v2'")
    backend = design.get("backend", "numpy")
    if devices is not None:
        if isinstance(devices, (bool, np.bool_)) or not isinstance(devices, (int, np.integer)) or devices < 1:
            raise ValueError("devices must be a positive integer")
        if int(devices) != 1:
            from . import fleet

            return fleet.run_sharded(design, seeds, devices=int(devices))

    prepared = {} if _prepared is None else _prepared
    if grammar_ver == "v2":
        if backend == "numpy":
            kernel = reimage_ref.reimage_region_v2
        elif backend in ("cpu", "gpu"):
            module = __import__(f"reimage.{backend}", fromlist=[backend])

            def kernel(nbr, xy, types, channels, focal, K, **kwargs):
                key = ("v2", id(nbr), K, channels.shape[1])
                if key not in prepared:
                    prepared[key] = module.prepare_v2(nbr, channels, K)
                return module.reimage_region_v2(
                    nbr, xy, types, channels, focal, K, prepared=prepared[key], **kwargs
                )
        else:
            raise ValueError("backend must be numpy, cpu or gpu")
        return np.stack([
            reimage_ref.reimage_v2(design["cells"], design, int(seed), region_kernel=kernel)
            for seed in seeds
        ])

    if backend == "numpy":
        kernel = reimage_ref.reimage_region
    elif backend in ("cpu", "gpu"):
        if backend == "cpu":
            from . import cpu as module
        else:
            from . import gpu as module

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


def run_v2(design, seeds, *, devices=None, fake_shards=False, _prepared=None):
    """Convenience entry point for v1+v2 re-imaging (with optional multi-GPU fleet sharding)."""
    v2_design = {**design, "grammar": "v2"}
    if fake_shards:
        from . import fleet

        return fleet.run_sharded(v2_design, seeds, devices=devices, fake_shards=True)
    return run(v2_design, seeds, devices=devices, _prepared=_prepared)
