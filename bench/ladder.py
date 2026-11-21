"""The speedup ladder harness: the same re-imaging work on each rung, timed end to end on named hardware.

    uv run python -I bench/ladder.py [--trials 20] [--cells 6000] [--out bench/results]

Every number in the JSON is measured on the machine named in its "hardware" block. A rung whose
code has not landed yet is recorded as "pending" with no number: never quote an unmeasured multiplier.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `python -I bench/ladder.py` finds core/

from core.reimage_ref import random_crop, reimage_region
from core.truth import neighbours

K, FOCAL = 8, [0, 1]


def hardware() -> dict:
    """The machine every number in this run belongs to."""
    hw = {"cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(),
          "python": platform.python_version(), "numpy": np.__version__, "gpu": None}
    try:
        info = dict(line.split(":", 1) for line in subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=10, check=False).stdout.splitlines() if ":" in line)
        hw["cpu"] = info.get("Model name", hw["cpu"]).strip()
        hw["cores_per_socket"] = int(info.get("Core(s) per socket", "0") or 0)
        hw["sockets"] = int(info.get("Socket(s)", "0") or 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    if shutil.which("nvidia-smi"):
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10, check=False).stdout.strip()
        hw["gpu"] = q.splitlines()[0] if q else None
    return hw


def synthetic_pool(n_regions: int, cells: int, side: float, seed: int) -> list[dict]:
    """Invented tissue at CODEX-like density (6,000 cells in 750 um ~ 330 neighbours within 100 um).

    shortcut: uniform positions and types, not real tissue; switch to core.cohorts regions once `make data` is on the bench box.
    """
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(n_regions):
        xy = rng.random((cells, 2)) * side
        pool.append({"xy": xy, "types": rng.integers(0, K, cells).astype(np.int16)})
    return pool


def make_trial(region_fn):
    """One trial: k region looks, each a random crop and mis-call, through one backend's reimage_region."""
    def trial(pool: list[dict], design: dict, seed: int):
        rng = np.random.default_rng(seed)
        out, reads = [], 0
        for r in rng.integers(0, len(pool), design["k"]):
            reg = pool[r]
            crop = random_crop(reg["xy"], design["fov_um"], rng)
            out.append(region_fn(reg["nbr"], reg["xy"], reg["types"], FOCAL, K,
                                 crop=crop, eps=design["eps"], rng=rng))
            reads += len(reg["nbr"]["idx"])
        return out, reads
    return trial


def rungs() -> dict:
    """name -> trial function, or a string saying why the rung is pending."""
    found = {"numpy_reference": make_trial(reimage_region)}
    for name, mod in (("numba_all_core", "reimage.cpu"), ("gpu", "reimage.gpu")):
        try:
            found[name] = make_trial(import_module(mod).reimage_region)  # #2: drop-in for the core signature
        except Exception as e:  # noqa: BLE001  # no numba, no CUDA device, or #2 not merged
            found[name] = f"pending: {mod} unavailable ({type(e).__name__}: {e})"[:200]
    return found


def matches_reference(fn, pool, design) -> bool:
    """Correctness gate: the same seed must give the reference's region sums and counts."""
    ref, _ = make_trial(reimage_region)(pool, design, seed=0)
    got, _ = fn(pool, design, seed=0)
    return all(np.allclose(a[0], b[0], rtol=0, atol=1e-9) and np.array_equal(a[1], b[1]) for a, b in zip(ref, got))


def time_rung(fn, pool, design, studies: int) -> dict:
    """Cold = the first trial (JIT, upload); amortised = the median of the rest. Seeds are fixed per trial."""
    times, reads = [], 0
    for s in range(studies):
        t0 = time.perf_counter()
        _, n = fn(pool, design, seed=s)
        times.append(time.perf_counter() - t0)
        reads += n
    warm = times[1:] or times
    amort = statistics.median(warm)
    return {"trials": studies, "cold_s": times[0], "amortised_s_per_trial": amort,
            "total_s": sum(times), "neighbour_reads": reads,
            "reads_per_s": reads / sum(times), "trials_per_s": 1.0 / amort}


def run(studies=20, cells=6000, side=750.0, n_regions=8, design=None, seed=0) -> dict:
    design = design or {"k": 4, "fov_um": 500.0, "eps": 0.1}
    t0 = time.perf_counter()
    pool = synthetic_pool(n_regions, cells, side, seed)
    for reg in pool:
        reg["nbr"] = neighbours(reg["xy"])
    prep = time.perf_counter() - t0
    result = {"hardware": hardware(), "data": "synthetic", "design": design,
              "pool": {"regions": n_regions, "cells_per_region": cells, "side_um": side,
                       "mean_neighbours_100um": float(np.mean([len(r["nbr"]["idx"]) / cells for r in pool]))},
              "prep_s": prep, "seed": seed, "rungs": {},
              # one trial is k region looks (crop, mis-call, recount, region sums), not a whole study
              "unit": {"trial": f"{design['k']} region looks: crop + mis-call + neighbour recount + region sums",
                       "excludes": "patient sampling and aggregation, features, outcomes, statistics; prep_s (neighbour lists) is reported separately"},
              "baseline": "numpy_reference"}
    for name, fn in rungs().items():
        if isinstance(fn, str):
            result["rungs"][name] = fn
            continue
        try:
            ok = matches_reference(fn, pool, design)
        except Exception as e:  # noqa: BLE001  # e.g. CUDA unavailable on this box
            result["rungs"][name] = f"pending: {type(e).__name__}: {e}"[:200]
            continue
        if not ok:
            result["rungs"][name] = "excluded: output differs from numpy_reference on seed 0"  # never time a wrong kernel
        else:
            result["rungs"][name] = time_rung(fn, pool, design, studies)
    return result


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--cells", type=int, default=6000)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    a = ap.parse_args(argv)
    res = run(studies=a.trials, cells=a.cells)
    a.out.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "-" for c in res["hardware"]["cpu"].lower()).strip("-")
    path = a.out / f"ladder-{slug}.json"  # named by hardware, not hostname: the repo is public
    path.write_text(json.dumps(res, indent=2) + "\n")
    print(path)
    for name, r in res["rungs"].items():
        print(f"  {name}: " + (r if isinstance(r, str) else f"{r['amortised_s_per_trial']:.3f} s/trial, "
                                f"cold {r['cold_s']:.3f} s, {r['reads_per_s']:.3g} reads/s"))
    print(f"  on {res['hardware']['cpu']} ({res['hardware']['logical_cpus']} logical CPUs)")
    return path


if __name__ == "__main__":
    main()
