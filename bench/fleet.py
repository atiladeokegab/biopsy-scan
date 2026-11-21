"""Multi-GPU fleet scaling benchmark: reports studies/s at 1, 2, 4 and 8 GPUs honestly.

    uv run python -I bench/fleet.py [--trials 20] [--cells 6000] [--out bench/results]

Every device count is recorded strictly as MEASURED (on that many visible real CUDA GPUs) or
UNMEASURED (never extrapolated or timed on the CUDA simulator).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.ladder import (
    cuda_simulator,
    hardware,
    make_trial,
    same_output,
    synthetic_pool,
    time_gpu_sharded,
)
from core.reimage_ref import reimage_region
from core.truth import neighbours
from reimage.fleet import visible_gpus

DEFAULT_DEVICE_COUNTS = (1, 2, 4, 8)


def run_fleet(
    *,
    studies: int = 20,
    cells: int = 6000,
    side: float = 750.0,
    n_regions: int = 8,
    design: dict | None = None,
    seed: int = 0,
    device_counts: tuple[int, ...] | list[int] = DEFAULT_DEVICE_COUNTS,
) -> dict:
    """Measure studies/s for each requested GPU count, marking unavailable counts UNMEASURED."""
    design = design or {"k": 4, "fov_um": 500.0, "eps": 0.1}
    counts = [int(d) for d in device_counts]
    if any(d < 1 for d in counts):
        raise ValueError("device_counts entries must be positive integers")

    hw = hardware()
    sim_active = cuda_simulator()
    avail = visible_gpus()

    t0 = time.perf_counter()
    pool = synthetic_pool(n_regions, cells, side, seed)
    for reg in pool:
        reg["nbr"] = neighbours(reg["xy"])
    prep_s = time.perf_counter() - t0

    ref_fn = make_trial(reimage_region)
    ref_first, _ = ref_fn(pool, design, seed=0)

    fleet_rows: list[dict] = []
    for n_dev in counts:
        if sim_active:
            fleet_rows.append({
                "devices": n_dev,
                "status": "UNMEASURED",
                "studies_per_s": None,
                "reason": "CUDA simulator (NUMBA_ENABLE_CUDASIM) active; timings are not GPU numbers",
            })
            continue
        if n_dev > avail:
            fleet_rows.append({
                "devices": n_dev,
                "status": "UNMEASURED",
                "studies_per_s": None,
                "reason": f"requested {n_dev} GPU(s); only {avail} visible on this host",
            })
            continue
        try:
            timing, first, _ = time_gpu_sharded(pool, design, studies, devices=n_dev)
        except Exception as e:  # noqa: BLE001
            fleet_rows.append({
                "devices": n_dev,
                "status": "UNMEASURED",
                "studies_per_s": None,
                "reason": f"{type(e).__name__}: {e}"[:200],
            })
            continue
        if not same_output(ref_first, first):
            fleet_rows.append({
                "devices": n_dev,
                "status": "UNMEASURED",
                "studies_per_s": None,
                "reason": "excluded: sharded GPU output differed from numpy_reference on seed 0",
            })
            continue
        fleet_rows.append({
            "devices": n_dev,
            "status": "MEASURED",
            "gpu": hw.get("gpu"),
            "studies": studies,
            "studies_per_s": timing["trials_per_s"],
            "total_s": timing["total_s"],
            "cold_s": timing["cold_s"],
            "reads_per_s": timing["reads_per_s"],
        })

    return {
        "hardware": hw,
        "data": "synthetic",
        "design": design,
        "seed": seed,
        "prep_s": prep_s,
        "devices_visible": avail,
        "unit": {
            "study": f"{design['k']} region looks: crop + mis-call + neighbour recount + region sums",
            "excludes": "patient sampling, feature aggregation, outcomes, statistics; prep_s reported separately",
        },
        "fleet": fleet_rows,
    }


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=20)
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
            print(f"  {row['devices']} GPU(s): MEASURED {row['studies_per_s']:.3f} studies/s (total {row['total_s']:.3f} s)")
        else:
            print(f"  {row['devices']} GPU(s): UNMEASURED ({row['reason']})")
    return path


if __name__ == "__main__":
    main()
