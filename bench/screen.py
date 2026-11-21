"""Equal-work timing of the R-15 score screen: the same statistic and resampling workload on CPU and GPU.

    uv run python -I bench/screen.py [--patients 212] [--candidates 2000] [--B 999] [--r14 R14.json] [--out PATH]

Both backends run stats.screen on identical SYNTHETIC inputs (patient rows, X, C and B fingerprinted).
A backend that cannot run here is UNMEASURED, never estimated. An R-14 full-fit baseline appears only
as a separate workflow comparison, and only when its three fingerprints match this run's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from stats.screen import get_backend, screen  # noqa: E402


def _sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def workload(patients, candidates, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.integers(0, 2, (patients, 1)).astype(float)
    X = rng.normal(size=(patients, candidates)) + 0.5 * C
    # Features are undefined per patient (e.g. a missing cell type), so columns share a few NaN masks.
    patterns = rng.random((8, patients)) < 0.05
    X[patterns[rng.integers(0, 8, candidates)].T] = np.nan
    t = rng.exponential(np.exp(-0.5 * C[:, 0]))
    c = rng.exponential(2.0, patients)
    return X, np.minimum(t, c), (t <= c).astype(float), C, rng.integers(0, 2, patients)


def _time(args, B, backend):
    screen(args[0][:, :8], *args[1:], n_perm=2, backend=backend)  # warm-up: imports, kernels
    start = time.perf_counter()
    result = screen(*args, n_perm=B, backend=backend)
    return time.perf_counter() - start, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--patients", type=int, default=212)
    parser.add_argument("--candidates", type=int, default=2000)
    parser.add_argument("--B", type=int, default=999)
    parser.add_argument("--r14", type=Path, help="R-14 full-fit results JSON (workflow comparison)")
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "screen.json")
    a = parser.parse_args(argv)
    args = workload(a.patients, a.candidates)
    X, t, e, C, s = args
    fingerprints = {"patient_rows": _sha(t, e, s), "X": _sha(np.nan_to_num(X, nan=np.pi)),
                    "C": _sha(C), "B": a.B}
    runs = {}
    seconds, cpu = _time(args, a.B, "numpy")
    runs["cpu"] = {"label": "MEASURED", "seconds": seconds, "backend": "numpy",
                   "hardware": platform.processor() or platform.machine(), "cores": os.cpu_count(),
                   "threads": "BLAS matrix products on all cores; elementwise risk-set steps on one",
                   "maxT": cpu["maxT"], "p": cpu["p"]}
    if get_backend("auto") is np:
        runs["gpu"] = {"label": "UNMEASURED", "reason": "no cupy with a CUDA device on this host"}
    else:
        import cupy

        seconds, gpu = _time(args, a.B, "cupy")
        runs["gpu"] = {"label": "MEASURED", "seconds": seconds, "backend": "cupy",
                       "hardware": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                       "maxT": gpu["maxT"], "p": gpu["p"],
                       "matches_cpu": bool(np.allclose(gpu["z"], cpu["z"], rtol=1e-9, atol=1e-9))}
    measured = all(r["label"] == "MEASURED" for r in runs.values())
    workflow = {"label": "UNMEASURED", "reason": "no R-14 full-fit result with matching fingerprints given"}
    if a.r14:
        r14 = json.loads(a.r14.read_text())
        theirs = {k: r14.get("fingerprints", {}).get(k) for k in ("patient_rows", "X", "C")}
        if theirs == {k: fingerprints[k] for k in theirs}:
            workflow = {"label": "MEASURED", "kind": "workflow comparison (per-candidate PHReg fits vs score screen)",
                        "r14_seconds": r14.get("seconds"), "screen_seconds": runs["cpu"]["seconds"]}
        else:
            workflow["reason"] = "R-14 fingerprints differ from this workload"
    result = {
        "schema": "screen-bench/1",
        "data": "SYNTHETIC",
        "statistic": cpu["metadata"]["statistic"],
        "null": cpu["metadata"]["null"]["name"],
        "workload": {"patients": a.patients, "candidates": a.candidates, "resamples": a.B,
                     "clinical": cpu["metadata"]["clinical"]["shape"], "fingerprints": fingerprints},
        "runs": runs,
        "equal_work_speedup": ({"label": "MEASURED", "value": runs["cpu"]["seconds"] / runs["gpu"]["seconds"]}
                               if measured else {"label": "UNMEASURED", "reason": runs["gpu"].get("reason")}),
        "workflow_comparison": workflow,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"cpu_s": runs["cpu"]["seconds"], "gpu": runs["gpu"]["label"], "out": str(a.out)}))
    return result


if __name__ == "__main__":
    main()
