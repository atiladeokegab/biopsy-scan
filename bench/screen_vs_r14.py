"""Interleaved A/B on Charville v2: R-14's per-candidate PHReg (A) against the R-15 fit-once score screen (B).

Same inputs (fingerprinted), same CPU, same core affinity, same window: A, B, A, B, ... with the 1-minute
load average beside each rep. The screen runs with n_perm=0, so both runs produce one statistic per candidate
and no resampling. The outputs differ (Wald from full fits against score at the null), so the ratio is a
CPU algorithmic speed-up within a workflow comparison, never a GPU figure and never an equal-work speed-up.

Run (after make data): uv run python -m bench.screen_vs_r14 --reps 3 --out bench/results/screen-vs-r14.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time as clock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.impl_a import cpu_model, fingerprints, inputs, time_full_run  # noqa: E402
from stats import screen  # noqa: E402


def time_screen(X, time, event, C, strata):
    t0 = clock.perf_counter()
    out = screen(X, time, event, C, strata, n_perm=0, backend="numpy")
    return clock.perf_counter() - t0, out


def summary(walls):
    return {"median_s": float(np.median(walls)), "min_s": float(min(walls)), "max_s": float(max(walls))}


def run(d, reps, cores):
    args = (d["X"], d["time"], d["event"], d["C"], d["strata"])
    reps_out, a_out, b_out = [], None, None
    for r in range(reps):
        for arm in ("A", "B"):
            load = os.getloadavg()[0]
            if arm == "A":
                wall, a_out = time_full_run(*args, workers=cores)
            else:
                wall, b_out = time_screen(*args)
            reps_out.append({"rep": r + 1, "arm": arm, "wall_s": wall, "load_1min_before": load,
                             "load_1min_after": os.getloadavg()[0]})
            print(f"rep {r + 1} {arm}: {wall:.2f} s, load {load:.1f}", flush=True)
    return reps_out, a_out, b_out


def report(d, reps_out, b_out, cores):
    walls = {arm: [x["wall_s"] for x in reps_out if x["arm"] == arm] for arm in "AB"}
    A, B = summary(walls["A"]), summary(walls["B"])
    return {
        "label": "MEASURED",
        "comparison": "workflow comparison: CPU algorithmic speed-up (fit once, score every candidate) "
                      "against one full Cox fit per candidate; not a GPU speed-up, not equal work",
        "cohort": "charville", "features": d["features"],
        "fingerprints": fingerprints(d["patient_ids"], d["time"], d["event"], d["strata"], d["C"], d["X"]),
        "shape": {"patients": int(d["X"].shape[0]), "candidates": int(d["X"].shape[1]), "clinical": int(d["C"].shape[1])},
        "machine": {"cpu": cpu_model(), "cores": cores, "python": platform.python_version()},
        "order": "interleaved A, B per rep, one process, one window",
        "arms": {
            "A": {"what": "R-14 bench.impl_a.time_full_run: statsmodels PHReg per candidate", "workers": cores, **A},
            "B": {"what": "R-15 stats.screen(n_perm=0, backend='numpy'): one null fit per NaN mask",
                  "threads": "numpy BLAS on the same affinity", "masks": b_out["metadata"]["masks"], **B},
        },
        "speedup_median": A["median_s"] / B["median_s"],
        "speedup_range": [A["min_s"] / B["max_s"], A["max_s"] / B["min_s"]],
        "reps": reps_out,
        "caveat": "shared VM; read each rep beside its load average",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cores", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results/screen-vs-r14.json")
    a = ap.parse_args(argv)
    d = inputs("charville")
    reps_out, _, b_out = run(d, a.reps, a.cores)
    out = report(d, reps_out, b_out, a.cores)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"A median {out['arms']['A']['median_s']:.2f} s, B median {out['arms']['B']['median_s']:.2f} s, "
          f"ratio {out['speedup_median']:.1f}")


if __name__ == "__main__":
    main()
