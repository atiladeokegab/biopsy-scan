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


def _run_v2_worker(payload):
    shard_id, device_id, design, shard_seeds = payload
    if design.get("backend") == "gpu":
        from numba import cuda

        cuda.select_device(device_id)
    return shard_id, run_v2(design, shard_seeds, devices=1)


def run_v2(design, seeds, *, devices=None, fake_shards=False):
    """Stack seeded v1+v2 patient-feature matrices using shared draw order, optionally sharded across GPUs."""
    import multiprocessing as mp
    import os
    import sys

    from core import reimage_ref

    seeds = np.asarray(seeds)
    if seeds.ndim != 1 or not len(seeds) or seeds.dtype.kind not in "iu" or (seeds < 0).any():
        raise ValueError("seeds must be a nonempty one-dimensional sequence of nonnegative integers")
    if "cells" not in design or "channels" not in design["cells"]:
        raise ValueError("design must provide cells with channels for run_v2")
    backend = design.get("backend", "numpy")

    if devices is not None:
        if isinstance(devices, (bool, np.bool_)) or not isinstance(devices, (int, np.integer)) or devices < 1:
            raise ValueError("devices must be a positive integer")
        devices = int(devices)

    sim_on = os.environ.get("NUMBA_ENABLE_CUDASIM", "0") not in ("", "0") or bool(
        getattr(getattr(sys.modules.get("numba"), "config", None), "ENABLE_CUDASIM", 0)
    )
    if fake_shards or (devices is not None and devices > 1):
        if backend == "gpu":
            from numba import cuda

            if fake_shards:
                if not sim_on:
                    raise RuntimeError("fake_shards requires NUMBA_ENABLE_CUDASIM=1")
                n_gpus = len(cuda.gpus)
                target = min(devices or 2, len(seeds))
                dev_for = lambda sid: sid % n_gpus  # noqa: E731
            else:
                if sim_on or not cuda.is_available():
                    raise RuntimeError("CUDA is unavailable; select the CPU backend explicitly")
                n_gpus = len(cuda.gpus)
                target = devices or n_gpus
                if target > n_gpus:
                    raise RuntimeError(f"requested {target} GPU(s) but only {n_gpus} visible")
                target = min(target, len(seeds))
                dev_for = lambda sid: sid  # noqa: E731
        else:
            if fake_shards:
                raise ValueError("fake_shards requires backend='gpu'")
            target = min(devices or 1, len(seeds))
            dev_for = lambda sid: 0  # noqa: E731
        if target > 1 or fake_shards:
            base, rem = divmod(len(seeds), target)
            payloads = []
            start = 0
            for sid in range(target):
                sz = base + (1 if sid < rem else 0)
                payloads.append((sid, dev_for(sid), dict(design), seeds[start:start + sz]))
                start += sz
            ctx = mp.get_context("spawn")
            with ctx.Pool(len(payloads)) as pool:
                parts = sorted(pool.map(_run_v2_worker, payloads), key=lambda item: item[0])
            return np.concatenate([p for _, p in parts], axis=0)

    if backend == "numpy":
        kernel = reimage_ref.reimage_region_v2
    elif backend in ("cpu", "gpu"):
        module = __import__(f"reimage.{backend}", fromlist=[backend])
        prepared = {}

        def kernel(nbr, xy, types, channels, focal, K, **kwargs):
            key = (id(nbr), K, channels.shape[1])
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
