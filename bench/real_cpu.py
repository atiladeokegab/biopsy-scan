"""Real Charville feature-engine timings: uv run python -m bench.real_cpu.

Paired, alternating backend order; shared topology and seeds; cumulative warm prefixes.
Use --checkpoints 10 100 1000 for the deferred longer run; default stops at 100.
No outcomes, survival statistics, null calibration or GPU timing are measured here.
"""

import argparse
from functools import partial
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numba
import numpy as np

from bench.ladder import hardware
from core.cohorts import load
from core.grammar import grammar, grammar_hash, n_features
from core.reimage_ref import reimage
from core.truth import layout, neighbours
from reimage import cpu


OUTPUT = Path(__file__).parent / "results" / "cpu-ladder.json"
CHECKPOINTS = (10, 100)
DESIGN = dict(patients=159, regions_per_patient=2, fov_um=400.0, eps=0.1, seed=0)


def save(report):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(OUTPUT)


def timed(fn, seed):
    start = perf_counter()
    result = fn(seed)
    return result, perf_counter() - start


def main(checkpoints=CHECKPOINTS):
    if not checkpoints or any(n < 1 for n in checkpoints) or list(checkpoints) != sorted(set(checkpoints)):
        raise ValueError("checkpoints must be increasing positive study counts")
    start = perf_counter()
    cells = load("charville")
    load_s = perf_counter() - start
    L = layout(cells)
    g = grammar(cells["type_names"])
    assert L["P"] == DESIGN["patients"] == 159, "wrong Charville patient cohort"
    assert n_features(g) == 630 and len(cells["patient_of_region"]) == 286
    print(f"Loaded {L['P']} patients, {len(cells['cell_type'])} real cells", flush=True)

    start = perf_counter()
    cells["_nbr"] = {r: neighbours(L["xy"](r)) for r in range(len(cells["patient_of_region"]))}
    adjacency_s = perf_counter() - start
    start = perf_counter()
    prepared = {id(nbr): cpu.prepare(nbr, L["K"]) for nbr in cells["_nbr"].values()}
    cpu_prepare_s = perf_counter() - start

    def cpu_region(nbr, xy, types, focal, K, **kwargs):
        return cpu.reimage_region(nbr, xy, types, focal, K, prepared=prepared[id(nbr)], **kwargs)

    functions = {
        "numpy_reference": partial(reimage, cells, DESIGN),
        "numba_cpu": partial(reimage, cells, DESIGN, region_kernel=cpu_region),
    }
    cold, outputs = {}, {}
    for name, fn in functions.items():
        outputs[name], cold[name] = timed(fn, 0)
    assert np.array_equal(outputs["numpy_reference"], outputs["numba_cpu"], equal_nan=True), "cold parity failed"

    report = {
        "schema_version": 1, "status": "MEASURED-CPU", "complete": False, "baseline": "lab_today",
        "scope": "Patient sampling, region sampling, crop, mis-call, recount and patient feature aggregation",
        "excludes": "Source loading, adjacency/preparation and first-call costs (reported separately); outcomes, survival statistics, null calibration and GPU",
        "sampling": "Patients with replacement; up to two regions without replacement per drawn patient",
        "method": "Alternating backend order per seed; shared prepared topology; cumulative prefixes",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "hardware": hardware(), "numba_threads": numba.get_num_threads(),
        "software": {name: version(name) for name in ("numpy", "numba", "scipy")},
        "cohort": {"name": "charville", "patients": L["P"], "regions": len(cells["patient_of_region"]),
                   "cells": len(cells["cell_type"]), "events": int(cells["event"].sum()),
                   "features": n_features(g), "grammar_hash": grammar_hash(g), "source": "Zenodo 13179600"},
        "design": DESIGN, "seed_range": [0, checkpoints[-1] - 1], "checkpoints": list(checkpoints),
        "preparation_s": {"load": load_s, "adjacency": adjacency_s, "cpu_prepare": cpu_prepare_s},
        "first_call_s": cold, "load_average_start": list(os.getloadavg()),
        "gpu": {"status": "UNMEASURED"}, "completed_studies": 0, "rows": [],
    }
    assert report["numba_threads"] == 8, "this run requires the assigned eight CPU threads"
    totals = {name: 0.0 for name in functions}
    save(report)
    print("Studies | NumPy seconds | Numba seconds | parity", flush=True)
    for seed in range(checkpoints[-1]):
        order = list(functions) if seed % 2 == 0 else list(reversed(functions))
        for name in order:
            outputs[name], elapsed = timed(functions[name], seed)
            totals[name] += elapsed
        assert outputs["numpy_reference"].shape == (159, 630)
        assert np.array_equal(outputs["numpy_reference"], outputs["numba_cpu"], equal_nan=True), f"parity failed at seed {seed}"
        count = seed + 1
        if count in checkpoints:
            row = {"studies": count, "seed_range": [0, seed],
                   "numpy_s": totals["numpy_reference"], "numba_s": totals["numba_cpu"],
                   "parity": "exact, including NaN placement"}
            report["rows"].append(row)
            print(f"{count} | {row['numpy_s']:.3f} | {row['numba_s']:.3f} | exact", flush=True)
        if count % 10 == 0 or count in checkpoints:
            report["completed_studies"] = count
            report["load_average_latest"] = list(os.getloadavg())
            report["complete"] = count == checkpoints[-1]
            save(report)
            print(f"Completed {count} paired studies", flush=True)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=CHECKPOINTS)
    main(tuple(parser.parse_args().checkpoints))
