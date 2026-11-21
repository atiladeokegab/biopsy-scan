"""Multi-GPU study-seed sharding with deterministic bit-identical merge."""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
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


def _selected_device_name(device_id: int) -> str:
    from numba import cuda

    if cuda_simulator():
        return "CUDA Simulator"
    dev = cuda.get_current_device() if hasattr(cuda, "get_current_device") else cuda.gpus[device_id]
    raw = getattr(dev, "name", "")
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _shard_worker(payload: tuple[int, int, dict[str, Any], np.ndarray]) -> tuple[int, np.ndarray, str]:
    """Execute one contiguous study-seed shard on one bound CUDA device (in a spawned worker)."""
    shard_id, device_id, design, shard_seeds = payload
    dev_name = ""
    if design.get("backend") == "gpu":
        from numba import cuda

        cuda.select_device(device_id)
        dev_name = _selected_device_name(device_id)
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
        name = _selected_device_name(0)
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


def _warm_timed_worker(
    shard_id: int,
    device_id: int,
    design: dict[str, Any],
    warmup_seeds: np.ndarray,
    warm_seeds: np.ndarray,
    barrier: Any,
    queue: Any,
) -> None:
    """Warm up CUDA/JIT + prepared snapshots, synchronize on barrier, then time warm shard."""
    try:
        dev_name = ""
        if design.get("backend") == "gpu":
            from numba import cuda

            cuda.select_device(device_id)
            dev_name = _selected_device_name(device_id)
        from reimage import run

        prepared: dict = {}
        run(design, warmup_seeds, devices=1, _prepared=prepared)
        barrier.wait(timeout=120)
        out = run(design, warm_seeds, devices=1, _prepared=prepared)
        queue.put((shard_id, out, dev_name, None))
    except Exception as exc:  # noqa: BLE001
        try:
            barrier.abort()
        except Exception:  # noqa: BLE001
            pass
        queue.put((shard_id, None, "", f"{type(exc).__name__}: {exc}"))


def measure_warm_sharded(
    design: dict[str, Any],
    warmup_seeds: Any,
    warm_seeds: Any,
    *,
    devices: int | None = None,
    fake_shards: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Spawn N persistent workers (including N=1), warm each in-worker, barrier-sync, and time."""
    warm_arr = _validate_seeds(warm_seeds)
    warmup_arr = _validate_seeds(warmup_seeds)
    if "cells" not in design:
        raise ValueError("design must provide cells")
    backend = design.get("backend", "numpy")
    if backend != "gpu":
        raise ValueError("measure_warm_sharded requires backend='gpu'")

    if devices is not None:
        if isinstance(devices, (bool, np.bool_)) or not isinstance(devices, (int, np.integer)) or devices < 1:
            raise ValueError("devices must be a positive integer")
        devices = int(devices)

    from numba import cuda

    if fake_shards:
        if not cuda_simulator():
            raise RuntimeError("fake_shards requires NUMBA_ENABLE_CUDASIM=1")
        n_sim_gpus = len(cuda.gpus)
        if n_sim_gpus < 1:
            raise RuntimeError("CUDA simulator exposed zero devices")
        target = 2 if devices is None else devices
        dev_for = lambda sid: sid % n_sim_gpus  # noqa: E731
    else:
        n_avail = visible_gpus()
        if n_avail < 1:
            raise RuntimeError("CUDA is unavailable; select the CPU backend explicitly")
        target = n_avail if devices is None else devices
        if target > n_avail:
            raise RuntimeError(f"requested {target} GPU(s) but only {n_avail} visible")
        dev_for = lambda sid: sid  # noqa: E731

    bounds = shard_bounds(len(warm_arr), target)
    n_workers = len(bounds)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n_workers + 1)
    queue = ctx.Queue()
    procs = []
    for sid, s, e in bounds:
        w_seed = warmup_arr[sid % len(warmup_arr): sid % len(warmup_arr) + 1]
        p = ctx.Process(
            target=_warm_timed_worker,
            args=(sid, dev_for(sid), dict(design), w_seed, warm_arr[s:e], barrier, queue),
        )
        p.start()
        procs.append(p)

    cold_t0 = time.perf_counter()
    try:
        barrier.wait(timeout=120)
        cold_warmup_s = time.perf_counter() - cold_t0
        t0 = time.perf_counter()
        results = [queue.get(timeout=300) for _ in range(n_workers)]
        wall_s = time.perf_counter() - t0
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

    for sid, _out, _name, err in results:
        if err is not None:
            raise RuntimeError(f"shard {sid} failed: {err}")
    results.sort(key=lambda item: item[0])
    merged = np.concatenate([part for _, part, _, _ in results], axis=0)
    names = [name for _, _, name, _ in results]
    return merged, {
        "devices_executed": len(results),
        "device_names": names,
        "wall_s": wall_s,
        "cold_warmup_s": cold_warmup_s,
    }
