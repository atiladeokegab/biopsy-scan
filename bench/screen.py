"""Equal-work timing of the R-15 score screen: the same statistic and resampling workload on CPU and GPU.

    uv run python -I bench/screen.py [--patients 212] [--candidates 2000] [--B 999] [--studies 1000]
                                     [--r14 R14.json] [--out PATH]

Both backends run stats.screen, then stats.screen_batch over --studies simulated outcome sets (the grid's
equal-work unit), on identical SYNTHETIC inputs (patient rows, X, C and B fingerprinted).
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

from stats.screen import get_backend, screen, screen_batch  # noqa: E402


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


def gpu_backend():
    """'gpu' (numba.cuda) on a real device, else 'cupy' with a device, else None. The simulator is not a GPU."""
    try:
        from numba import cuda

        if cuda.is_available() and not cuda.config.ENABLE_CUDASIM:
            return "gpu"
    except ImportError:
        pass
    return None if get_backend("auto") is np else "cupy"


def studies(t, C, S, seed=1):
    """S simulated outcome sets on the workload's patients (clinical effect only), finite times."""
    rng = np.random.default_rng(seed)
    T = rng.exponential(np.exp(-0.5 * C[:, 0]), (S, len(t)))
    cens = rng.exponential(2.0, (S, len(t)))
    return np.minimum(T, cens), (T <= cens).astype(float)


def _time_batch(X, T, E, C, s, backend):
    screen_batch(X[:, :8], T[:2], E[:2], C, s, backend=backend)  # warm-up
    start = time.perf_counter()
    result = screen_batch(X, T, E, C, s, backend=backend)
    return time.perf_counter() - start, result


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
    parser.add_argument("--studies", type=int, default=1000, help="simulated studies for the batched path")
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
    T, E = studies(t, C, a.studies)
    seconds, cpu_batch = _time_batch(X, T, E, C, s, "numpy")
    batch = {"cpu": {"label": "MEASURED", "seconds": seconds, "backend": "numpy",
                     "studies_per_hour": 3600 * a.studies / seconds}}
    name = gpu_backend()
    if name is None:
        runs["gpu"] = {"label": "UNMEASURED", "reason": "no CUDA device on this host (numba.cuda or cupy)"}
        batch["gpu"] = dict(runs["gpu"])
    else:
        seconds, gpu = _time(args, a.B, name)
        hardware = (__import__("numba").cuda.get_current_device().name.decode() if name == "gpu" else
                    __import__("cupy").cuda.runtime.getDeviceProperties(0)["name"].decode())
        runs["gpu"] = {"label": "MEASURED", "seconds": seconds, "backend": name, "hardware": hardware,
                       "maxT": gpu["maxT"], "p": gpu["p"],
                       "matches_cpu": bool(np.allclose(gpu["z"], cpu["z"], rtol=1e-9, atol=1e-9))}
        seconds, gpu_batch = _time_batch(X, T, E, C, s, name)
        batch["gpu"] = {"label": "MEASURED", "seconds": seconds, "backend": name, "hardware": hardware,
                        "studies_per_hour": 3600 * a.studies / seconds,
                        "matches_cpu": bool(np.allclose(gpu_batch["z"], cpu_batch["z"], rtol=1e-9, atol=1e-9))}
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
        "batched": {"studies": a.studies, "seed": 1, "what": "screen_batch: observed screen per study, null refit per study",
                    "outcomes_sha256": _sha(T, E), "runs": batch,
                    "equal_work_speedup": ({"label": "MEASURED", "value": batch["cpu"]["seconds"] / batch["gpu"]["seconds"]}
                                           if batch["gpu"]["label"] == "MEASURED"
                                           else {"label": "UNMEASURED", "reason": batch["gpu"]["reason"]})},
        "workflow_comparison": workflow,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"cpu_s": runs["cpu"]["seconds"], "gpu": runs["gpu"]["label"], "out": str(a.out)}))
    return result


if __name__ == "__main__":
    main()
