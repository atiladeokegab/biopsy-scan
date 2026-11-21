"""Why Numba with 8 threads lost to single-thread NumPy: controlled, warm, fixed-seed measurements.

uv run python -m bench.numba_negative --output bench/results/numba-negative.json
Each configuration runs in a fresh subprocess with its own thread environment; neighbour lists are
built once and shared through a pickle so every configuration times identical inputs.
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

STUDY_SEEDS = (1, 2)   # seed 0 is the warm-up, excluded from every timing
KERNEL_REPEATS = 1
DESIGN = dict(patients=159, regions_per_patient=2, fov_um=400.0, eps=0.1, seed=0)


def _snapshot():
    t = os.times()
    return perf_counter(), t.user + t.system, os.getloadavg()


def _since(start):
    wall, cpu, load = _snapshot()
    return {"wall_s": wall - start[0], "process_cpu_s": cpu - start[1],
            "load_average_before": list(start[2]), "load_average_after": list(load)}


def worker(nbr_path):
    """One configuration: timings for the current thread environment, as a JSON line on stdout."""
    import numba
    import numpy as np

    from core.cohorts import load
    from core import reimage_ref
    from core.reimage_ref import reimage
    from reimage import cpu

    cells = load("charville")
    with open(nbr_path, "rb") as f:
        cells["_nbr"] = pickle.load(f)
    from core.truth import layout
    K = layout(cells)["K"]
    prepared = {id(n): cpu.prepare(n, K) for n in cells["_nbr"].values()}

    def numba_region(nbr, xy, types, focal, K, **kw):
        return cpu.reimage_region(nbr, xy, types, focal, K, prepared=prepared[id(nbr)], **kw)

    # Count-kernel time inside each path, by wrapping the module globals the paths call.
    calls, spent = [], {"numpy": 0.0, "numba": 0.0}
    ref_counts, jit_counts = reimage_ref.reimage_counts, cpu._counts

    def timed_ref(*a):
        t = perf_counter(); out = ref_counts(*a); spent["numpy"] += perf_counter() - t; return out

    def timed_jit(*a):
        calls.append(a)
        t = perf_counter(); out = jit_counts(*a); spent["numba"] += perf_counter() - t; return out

    t = perf_counter()
    x_np = reimage(cells, DESIGN, 0)
    first_numpy = perf_counter() - t
    t = perf_counter()
    x_nb = reimage(cells, DESIGN, 0, region_kernel=numba_region)   # compile or cache load happens here
    first_numba = perf_counter() - t
    assert np.array_equal(x_np, x_nb, equal_nan=True), "warm-up parity failed"

    reimage_ref.reimage_counts, cpu._counts = timed_ref, timed_jit
    studies = []
    for k, seed in enumerate(STUDY_SEEDS):
        order = ("numpy", "numba") if k % 2 == 0 else ("numba", "numpy")
        row, out = {"seed": seed}, {}
        for name in order:
            spent[name] = 0.0
            calls.clear() if name == "numba" else None
            s = _snapshot()
            out[name] = reimage(cells, DESIGN, seed, **({"region_kernel": numba_region} if name == "numba" else {}))
            row[name] = {**_since(s), "count_kernel_s": spent[name]}
        assert np.array_equal(out["numpy"], out["numba"], equal_nan=True), f"parity failed at seed {seed}"
        studies.append(row)
    reimage_ref.reimage_counts, cpu._counts = ref_counts, jit_counts

    # Kernel-only replay of the last study's region calls: no Python glue, no validation.
    sizes = [len(a[3]) for a in calls]
    visible = [int(a[4].sum()) for a in calls]
    s = _snapshot()
    for _ in range(KERNEL_REPEATS):
        for a in calls:
            jit_counts(*a)
    replay = _since(s)

    @numba.njit(parallel=True, cache=False)
    def empty(n):
        out = np.zeros(n, np.int64)
        for i in numba.prange(n):
            out[i] = i
        return out

    empty(8)
    s = _snapshot()
    for _ in range(len(calls) * KERNEL_REPEATS):
        empty(8)
    launch = _since(s)

    print(json.dumps({
        "numba_threads": numba.get_num_threads(),
        "threading_layer": numba.threading_layer(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {k: os.environ.get(k) for k in (
            "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER", "OMP_NUM_THREADS", "OMP_WAIT_POLICY",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "first_call_s": {"numpy": first_numpy, "numba": first_numba},
        "studies": studies,
        "kernel_replay": {**replay, "region_calls": len(calls), "repeats": KERNEL_REPEATS,
                          "per_call_ms": 1e3 * replay["wall_s"] / (len(calls) * KERNEL_REPEATS),
                          "cells_per_region_median": int(np.median(sizes)),
                          "visible_fraction_mean": float(np.sum(visible) / np.sum(sizes))},
        "empty_parallel_launch": {**launch, "calls": len(calls) * KERNEL_REPEATS,
                                  "per_call_us": 1e6 * launch["wall_s"] / (len(calls) * KERNEL_REPEATS)},
    }), flush=True)


def build_neighbours(path):
    from core.cohorts import load
    from core.reimage_ref import draw
    from core.truth import layout, neighbours

    cells = load("charville")
    L = layout(cells)
    needed = {r for seed in (0, *STUDY_SEEDS) for row in draw(cells, DESIGN, seed)["looks"] for r, _, _ in row}
    nbr = {r: neighbours(L["xy"](r)) for r in sorted(needed)}
    with open(path, "wb") as f:
        pickle.dump(nbr, f)


CONFIGS = [  # (label, environment overrides); BLAS pinned to 1 unless the row tests it
    ("threads-1", {"NUMBA_NUM_THREADS": "1"}),
    ("threads-8", {"NUMBA_NUM_THREADS": "8"}),
    ("threads-8-workqueue", {"NUMBA_NUM_THREADS": "8", "NUMBA_THREADING_LAYER": "workqueue"}),
    ("threads-8-blas-unpinned", {"NUMBA_NUM_THREADS": "8", "OMP_NUM_THREADS": None,
                                 "OPENBLAS_NUM_THREADS": None, "MKL_NUM_THREADS": None}),
    ("threads-2", {"NUMBA_NUM_THREADS": "2"}),
    ("threads-4", {"NUMBA_NUM_THREADS": "4"}),
    ("threads-8-omp-passive-wait", {"NUMBA_NUM_THREADS": "8", "OMP_WAIT_POLICY": "PASSIVE"}),
    ("threads-1-repeat", {"NUMBA_NUM_THREADS": "1"}),
    ("threads-8-repeat", {"NUMBA_NUM_THREADS": "8"}),
]


def main(output, only=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"preserving existing evidence: {output}")
    with tempfile.TemporaryDirectory() as tmp:
        nbr_path = Path(tmp) / "nbr.pkl"
        t = perf_counter()
        build_neighbours(nbr_path)
        prep = perf_counter() - t
        runs = []
        for label, overrides in [c for c in CONFIGS if not only or c[0] in only]:
            env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            for k, v in overrides.items():
                env.pop(k, None) if v is None else env.__setitem__(k, v)
            p = subprocess.run([sys.executable, "-m", "bench.numba_negative", "--worker", str(nbr_path)],
                               env=env, capture_output=True, text=True, check=True, timeout=1800)
            run = {"label": label, **json.loads(p.stdout.strip().splitlines()[-1])}
            runs.append(run)
            print(json.dumps(run), flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"neighbour_build_s": prep, "design": DESIGN, "study_seeds": STUDY_SEEDS,
                                          "warmup_seed": 0, "complete": False, "runs": runs}, indent=2) + "\n")
            st = run["studies"]
            print(label, "numpy", round(sum(r["numpy"]["wall_s"] for r in st) / len(st), 3),
                  "numba", round(sum(r["numba"]["wall_s"] for r in st) / len(st), 3), flush=True)
    output.write_text(json.dumps({"neighbour_build_s": prep, "design": DESIGN, "study_seeds": STUDY_SEEDS,
                                  "warmup_seed": 0, "complete": True, "runs": runs}, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--only", nargs="+", help="run only these configuration labels")
    ap.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args()
    worker(a.worker) if a.worker else main(a.output, a.only)
