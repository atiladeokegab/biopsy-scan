"""The race on this machine's CPU: the lab way against the Preflight way on a small slice of Charville studies.

    make race-local                                   # after `make data`; 8 studies per stream, every core
    uv run python -m bench.race_local --studies 20 --workers 4

Both arms are bench.design_batch's, unchanged (the harness behind #113 and #120's Modal figures):
  lab    per study: fresh cKDTree re-imaging, then a 1,000-shuffle conditional maxT permutation test.
  fused  per study: the fused Numba re-imaging kernel (#120), then the Cox score screen; one calibrated threshold
         from the calibration stream decides every study, so Preflight also pays for those studies.
s/study is each arm's compute wall per evaluated study (re-imaging and scoring; load, prep, adjacency and JIT
warm-up excluded), the race's own metric. Feature parity: each shared study's features are bit-identical.
Timings are MEASURED on this machine; not the Modal figures.
"""
from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter

from bench import design_batch as db

LABEL = "MEASURED on this machine; not the Modal figures"


def adjacency(cells: dict, workers: int) -> dict:
    """core.truth.neighbours of every region, built across the pool: the same CSR lists that the fused arm's
    prepare() builds one region at a time (88 s of a 167 s run here), keyed by the region's coordinates."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from core import truth
    L = truth.layout(cells)
    xys = [L["xy"](r) for r in range(len(cells["patient_of_region"]))]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
        return {xy.tobytes(): nb for xy, nb in zip(xys, pool.map(truth.neighbours, xys, chunksize=4))}


def race(cells: dict, studies: int, workers: int) -> dict:
    """Run the lab and fused arms over `studies` per stream; the lab skips calibration, as on Modal."""
    from core import truth
    plan = {s: studies for s in db.STREAMS}
    arms = {}
    t = perf_counter()
    built, serial, reference = adjacency(cells, workers), truth.neighbours, truth.truth
    adjacency_s = perf_counter() - t
    truth.neighbours = lambda xy, *a, **k: built[xy.tobytes()] if not (a or k) else serial(xy, *a, **k)
    once: dict = {}

    def truth_once(c, *a, **k):  # both arms' prepare() take the cohort's truth features: compute them once, not per arm
        if c is not cells or a or k:
            return reference(c, *a, **k)
        if "x" not in once:
            once["x"] = reference(c)
        return once["x"]
    truth.truth = truth_once
    try:
        for arm in ("lab", "fused"):
            db._S.clear()  # per-process kernel and CSR state: on Modal each arm has its own container
            out = db.run_arm(arm, cells, plan, workers=workers)
            out["stages"] = db.stage_times(out)
            arms[arm] = out
    finally:
        truth.neighbours, truth.truth = serial, reference
    lab, fused = (arms[a]["stages"]["wall_per_study_s"] for a in ("lab", "fused"))
    par = db.parity(arms)
    return {"label": LABEL, "workers": workers, "studies_per_stream": studies,
            "evaluated": {a: o["stages"]["evaluated_studies"] for a, o in arms.items()},
            "s_per_study": {"lab": lab, "fused": fused}, "ratio": lab / fused,
            "parity": {"x_bitwise": par["x_bitwise"], "x_compared": par["x_compared"]},
            "setup_s": {"adjacency_in_parallel": adjacency_s,
                        **{a: {"prep": o["setup"]["prep_s"], "warm": o["warm_s"] or 0.0} for a, o in arms.items()}}}


def report(r: dict) -> str:
    ok = "bit-identical" if r["parity"]["x_bitwise"] else "MISMATCH"
    return "\n".join([
        f"race-local: {r['label']}",
        f"  {r['workers']} workers, {r['studies_per_stream']} studies per stream "
        f"(lab {r['evaluated']['lab']}: null + alternative; Preflight {r['evaluated']['fused']}: plus calibration)",
        f"  lab way        {r['s_per_study']['lab']:.4f} s/study  (1,000-shuffle permutation test per study)",
        f"  Preflight way  {r['s_per_study']['fused']:.4f} s/study  (fused kernel, one calibrated threshold)",
        f"  ratio          {r['ratio']:.1f}x  (lab / Preflight, compute wall per evaluated study)",
        f"  feature parity {ok} on {r['parity']['x_compared']} shared studies",
        f"  setup, outside s/study: adjacency {r['setup_s']['adjacency_in_parallel']:.0f} s (every region, across the "
        f"workers); lab prep {r['setup_s']['lab']['prep']:.0f} s; Preflight prep {r['setup_s']['fused']['prep']:.0f} s, "
        f"JIT warm-up {r['setup_s']['fused']['warm']:.0f} s"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--studies", type=int, default=8, help="studies per stream (default 8)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="worker processes (default: every core)")
    a = ap.parse_args(argv)
    if a.studies < 1 or a.workers < 1:
        ap.error("--studies and --workers must be at least 1")
    from core.cohorts import load
    t = perf_counter()
    r = race(load("charville"), a.studies, a.workers)
    print(report(r))
    print(f"  total wall     {perf_counter() - t:.0f} s (including data load, preparation and JIT warm-up)")
    return 0 if r["parity"]["x_bitwise"] else 1


if __name__ == "__main__":
    sys.exit(main())
