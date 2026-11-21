"""Multi-GPU fleet scaling benchmark: reports studies/s at 1, 2, 4 and 8 GPUs honestly.

    uv run python -I bench/fleet.py [--trials 20] [--cells 6000] [--out bench/results]

Every device count times full contract studies (`reimage.fleet.run_sharded_with_meta`, including
patient sampling, region sampling, crops, mis-calls, GPU counting, and feature aggregation) after a
per-shard warmup study to discard cold samples. Each count is recorded strictly as MEASURED (when
all N shards executed on N distinct visible CUDA devices with resolved device names) or UNMEASURED.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.ladder import K, cuda_simulator, hardware, synthetic_pool
from core.truth import neighbours
from reimage import run as reimage_run
from reimage.fleet import run_sharded_with_meta, visible_gpus

DEFAULT_DEVICE_COUNTS = (1, 2, 4, 8)


def synthetic_cohort(n_patients: int = 8, regions_per_patient: int = 2, cells: int = 6000, side: float = 750.0,
                     seed: int = 0) -> tuple[dict, float]:
    """Synthetic contract cells dict with precomputed region neighbour lists (prep_s separate)."""
    n_regions = n_patients * regions_per_patient
    pool = synthetic_pool(n_regions, cells, side, seed)
    t0 = time.perf_counter()
    nbr_cache = {r: neighbours(reg["xy"]) for r, reg in enumerate(pool)}
    prep_s = time.perf_counter() - t0
    cells_dict = {
        "x": np.concatenate([reg["xy"][:, 0] for reg in pool]).astype(np.float32),
        "y": np.concatenate([reg["xy"][:, 1] for reg in pool]).astype(np.float32),
        "cell_type": np.concatenate([reg["types"] for reg in pool]).astype(np.int8),
        "region_id": np.repeat(np.arange(n_regions, dtype=np.int32), cells),
        "patient_of_region": np.repeat(np.arange(n_patients, dtype=np.int32), regions_per_patient),
        "type_names": [f"Tumor {i}" if i < 2 else f"Type {i}" for i in range(K)],
        "_nbr": nbr_cache,
    }
    return cells_dict, prep_s


def _unmeasured(n_dev: int, reason: str) -> dict:
    return {"devices": n_dev, "status": "UNMEASURED", "studies_per_s": None, "gpus": [], "reason": reason}


def run_fleet(
    *,
    studies: int = 20,
    cells: int = 6000,
    side: float = 750.0,
    patients: int = 8,
    regions_per_patient: int = 2,
    k: int = 2,
    fov_um: float = 500.0,
    eps: float = 0.1,
    seed: int = 0,
    device_counts: tuple[int, ...] | list[int] = DEFAULT_DEVICE_COUNTS,
) -> dict:
    """Measure full studies/s across requested GPU counts, marking unavailable or partial counts UNMEASURED."""
    counts = [int(d) for d in device_counts]
    if any(d < 1 for d in counts):
        raise ValueError("device_counts entries must be positive integers")

    hw = hardware()
    sim_active = cuda_simulator()
    avail = visible_gpus()
    cohort, prep_s = synthetic_cohort(patients, regions_per_patient, cells, side, seed)
    study_design = {
        "cells": cohort,
        "backend": "gpu",
        "patients": patients,
        "regions_per_patient": k,
        "fov_um": fov_um,
        "eps": eps,
        "seed": seed,
    }
    warm_seeds = np.arange(1000, 1000 + studies, dtype=np.int64)
    ref_zero = reimage_run({**study_design, "backend": "numpy"}, warm_seeds[:1])

    fleet_rows: list[dict] = []
    for n_dev in counts:
        if sim_active:
            fleet_rows.append(_unmeasured(n_dev, "CUDA simulator (NUMBA_ENABLE_CUDASIM) active; timings are not GPU numbers"))
            continue
        if n_dev > avail:
            fleet_rows.append(_unmeasured(n_dev, f"requested {n_dev} GPU(s); only {avail} visible on this host"))
            continue
        if studies < n_dev:
            fleet_rows.append(_unmeasured(n_dev, f"only {studies} study seed(s) supplied for {n_dev} requested GPU(s)"))
            continue
        try:
            # One warmup study per shard excludes per-device JIT and upload cold times before wall timing.
            warmup_seeds = np.arange(seed, seed + n_dev, dtype=np.int64)
            t_cold0 = time.perf_counter()
            _, warm_meta = run_sharded_with_meta(study_design, warmup_seeds, devices=n_dev)
            cold_wall_s = time.perf_counter() - t_cold0

            t0 = time.perf_counter()
            out, meta = run_sharded_with_meta(study_design, warm_seeds, devices=n_dev)
            wall_s = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            fleet_rows.append(_unmeasured(n_dev, f"{type(e).__name__}: {e}"[:200]))
            continue

        executed = int(meta.get("devices_executed", 0))
        device_names = [str(g).strip() for g in meta.get("device_names", []) if str(g).strip()]
        if executed != n_dev or warm_meta.get("devices_executed") != n_dev:
            fleet_rows.append(_unmeasured(n_dev, f"executed {executed} shard(s), expected {n_dev}"))
            continue
        if len(device_names) != n_dev:
            fleet_rows.append(_unmeasured(n_dev, "selected CUDA devices did not report all GPU names"))
            continue
        if not np.allclose(out[:1], ref_zero, rtol=0, atol=1e-9, equal_nan=True):
            fleet_rows.append(_unmeasured(n_dev, "excluded: sharded GPU study output differed from numpy reference"))
            continue

        fleet_rows.append({
            "devices": n_dev,
            "status": "MEASURED",
            "gpus": device_names,
            "studies": studies,
            "total_s": wall_s,
            "cold_warmup_s": cold_wall_s,
            "studies_per_s": studies / wall_s if wall_s > 0 else 0.0,
        })

    return {
        "hardware": hw,
        "data": "synthetic",
        "design": {"patients": patients, "regions_per_patient": k, "fov_um": fov_um, "eps": eps},
        "seed": seed,
        "prep_s": prep_s,
        "devices_visible": avail,
        "unit": {
            "study": "full contract study: patient sampling, region sampling, crop, mis-call, GPU counting, and features",
            "cold_handling": "each GPU shard runs one warmup study before timed batch",
            "excludes": "prep_s (neighbour lists) reported separately",
        },
        "fleet": fleet_rows,
    }


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=20, help="Full warm studies to time per device count")
    ap.add_argument("--cells", type=int, default=6000)
    ap.add_argument("--devices", nargs="+", type=int, default=list(DEFAULT_DEVICE_COUNTS))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    a = ap.parse_args(argv)
    res = run_fleet(studies=a.trials, cells=a.cells, device_counts=a.devices)
    a.out.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "-" for c in res["hardware"]["cpu"].lower()).strip("-")
    path = a.out / f"fleet-{slug}.json"
    path.write_text(json.dumps(res, indent=2) + "\n")
    print(path)
    for row in res["fleet"]:
        if row["status"] == "MEASURED":
            print(f"  {row['devices']} GPU(s) ({', '.join(row['gpus'])}): "
                  f"MEASURED {row['studies_per_s']:.3f} studies/s (total {row['total_s']:.3f} s)")
        else:
            print(f"  {row['devices']} GPU(s): UNMEASURED ({row['reason']})")
    return path


if __name__ == "__main__":
    main()
