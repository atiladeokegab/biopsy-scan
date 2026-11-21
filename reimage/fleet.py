"""Multi-GPU study-seed sharding with deterministic bit-identical merge."""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import Any

import numpy as np


def cuda_simulator() -> bool:
    """True when Numba's CUDA simulator is enabled (CPU emulation, never real GPU timing)."""
    if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") not in ("", "0"):
        return True
    numba = sys.modules.get("numba")
    return bool(getattr(getattr(numba, "config", None), "ENABLE_CUDASIM", 0))


def visible_gpus() -> int:
    """Count real CUDA devices visible to this process (0 when simulator or CUDA unavailable)."""
    if cuda_simulator():
        return 0
    try:
        from numba import cuda
    except Exception:  # noqa: BLE001
        return 0
    if not cuda.is_available():
        return 0
    return len(cuda.gpus)


def shard_bounds(n_items: int, n_devices: int) -> list[tuple[int, int, int]]:
    """Split n_items into up to n_devices non-empty contiguous [start, end) slices."""
    if isinstance(n_devices, (bool, np.bool_)) or not isinstance(n_devices, (int, np.integer)) or n_devices < 1:
        raise ValueError("devices must be a positive integer")
    if n_items < 1:
        raise ValueError("n_items must be a positive integer")
    workers = min(int(n_devices), int(n_items))
    base, rem = divmod(int(n_items), workers)
    shards: list[tuple[int, int, int]] = []
    start = 0
    for shard_id in range(workers):
        size = base + (1 if shard_id < rem else 0)
        shards.append((shard_id, start, start + size))
        start += size
    return shards


def _validate_seeds(seeds: Any) -> np.ndarray:
    arr = np.asarray(seeds)
    if arr.ndim != 1 or not len(arr) or arr.dtype.kind not in "iu" or (arr < 0).any():
        raise ValueError("seeds must be a nonempty one-dimensional sequence of nonnegative integers")
    return arr


def _selected_device_name() -> str:
    from numba import cuda

    dev = cuda.get_current_device()
    raw = getattr(dev, "name", "")
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _shard_worker(payload: tuple[int, int, dict[str, Any], np.ndarray]) -> tuple[int, np.ndarray, str]:
    """Execute one contiguous study-seed shard on one bound CUDA device (in a spawned worker)."""
    shard_id, device_id, design, shard_seeds = payload
    dev_name = ""
    if design.get("backend") == "gpu":
        from numba import cuda

        cuda.select_device(device_id)
        dev_name = _selected_device_name()
    from reimage import run

    out = run(design, shard_seeds, devices=1)
    return shard_id, out, dev_name


def run_sharded_with_meta(
    design: dict[str, Any],
    seeds: Any,
    *,
    devices: int | None = None,
    fake_shards: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run seeded re-imaging across contiguous device shards and return merged output + device metadata."""
    seed_arr = _validate_seeds(seeds)
    if "cells" not in design:
        raise ValueError("design must provide cells")
    backend = design.get("backend", "numpy")

    if devices is not None:
        if isinstance(devices, (bool, np.bool_)) or not isinstance(devices, (int, np.integer)) or devices < 1:
            raise ValueError("devices must be a positive integer")
        devices = int(devices)

    if backend != "gpu":
        if fake_shards:
            raise ValueError("fake_shards is only supported with backend='gpu'")
        if devices is None or devices == 1:
            from reimage import run

            return run({**design}, seed_arr, devices=1), {"devices_executed": 1, "device_names": []}
        bounds = shard_bounds(len(seed_arr), devices)
        ctx = mp.get_context("spawn")
        payloads = [(sid, 0, dict(design), seed_arr[s:e]) for sid, s, e in bounds]
        with ctx.Pool(len(payloads)) as pool:
            parts = sorted(pool.map(_shard_worker, payloads), key=lambda item: item[0])
        return np.concatenate([part for _, part, _ in parts], axis=0), {"devices_executed": len(parts), "device_names": []}

    from numba import cuda

    if fake_shards:
        if not cuda_simulator():
            raise RuntimeError("fake_shards requires NUMBA_ENABLE_CUDASIM=1")
        n_sim_gpus = len(cuda.gpus)
        if n_sim_gpus < 1:
            raise RuntimeError("CUDA simulator exposed zero devices")
        target = 2 if devices is None else devices
        bounds = shard_bounds(len(seed_arr), target)
        ctx = mp.get_context("spawn")
        payloads = [(sid, sid % n_sim_gpus, dict(design), seed_arr[s:e]) for sid, s, e in bounds]
        with ctx.Pool(len(payloads)) as pool:
            parts = sorted(pool.map(_shard_worker, payloads), key=lambda item: item[0])
        return (
            np.concatenate([part for _, part, _ in parts], axis=0),
            {"devices_executed": len(parts), "device_names": [n for _, _, n in parts]},
        )

    n_avail = visible_gpus()
    if n_avail < 1:
        raise RuntimeError("CUDA is unavailable; select the CPU backend explicitly")
    target = n_avail if devices is None else devices
    if target > n_avail:
        raise RuntimeError(f"requested {target} GPU(s) but only {n_avail} visible")

    bounds = shard_bounds(len(seed_arr), target)
    if len(bounds) == 1:
        cuda.select_device(0)
        name = _selected_device_name()
        from reimage import run

        return run(dict(design), seed_arr, devices=1), {"devices_executed": 1, "device_names": [name]}

    ctx = mp.get_context("spawn")
    payloads = [(sid, sid, dict(design), seed_arr[s:e]) for sid, s, e in bounds]
    with ctx.Pool(len(payloads)) as pool:
        parts = sorted(pool.map(_shard_worker, payloads), key=lambda item: item[0])
    return (
        np.concatenate([part for _, part, _ in parts], axis=0),
        {"devices_executed": len(parts), "device_names": [n for _, _, n in parts]},
    )


def run_sharded(
    design: dict[str, Any],
    seeds: Any,
    *,
    devices: int | None = None,
    fake_shards: bool = False,
) -> np.ndarray:
    """Run seeded re-imaging across contiguous device shards and merge deterministically."""
    out, _ = run_sharded_with_meta(design, seeds, devices=devices, fake_shards=fake_shards)
    return out
