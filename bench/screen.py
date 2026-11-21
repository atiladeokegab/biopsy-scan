"""Equal-work timing of the R-15 score screen: the same statistic and resampling workload on CPU and GPU.

    uv run python -I bench/screen.py [--patients 212] [--candidates 2000] [--B 999] [--studies 1000] [--out PATH]

Both backends run stats.screen, then stats.screen_batch over --studies simulated outcome sets (the grid's
equal-work unit), on identical SYNTHETIC inputs (patient rows, X, C and B fingerprinted).
A backend that cannot run here is UNMEASURED, never estimated. A speed-up is MEASURED only when every
timed output agrees within PARITY (U, V, z, null_maxT, adjusted and omnibus p); otherwise it is WITHHELD.
The R-14 workflow comparison is bench/screen_vs_r14.py, run on the real producer's inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bench.impl_a import cpu_model  # noqa: E402
from stats.screen import get_backend, screen, screen_batch  # noqa: E402

PARITY = {"rtol": 1e-9, "atol": 1e-9}  # float64 on both sides; only summation order may differ
SCREEN_OUTPUTS = ("U", "V", "z", "null_maxT", "adjusted_p", "p")
BATCH_OUTPUTS = ("U", "V", "z", "maxT")


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


def mismatched(cpu, gpu, keys):
    """Outputs whose CPU and GPU values differ beyond PARITY; a missing output counts as differing."""
    return [k for k in keys if k not in cpu or k not in gpu or np.shape(cpu[k]) != np.shape(gpu[k])
            or not np.allclose(gpu[k], cpu[k], equal_nan=True, **PARITY)]


def speedup(cpu, gpu, bad):
    if gpu["label"] != "MEASURED":
        return {"label": "UNMEASURED", "reason": gpu.get("reason")}
    if bad:
        return {"label": "WITHHELD", "reason": f"CPU and GPU outputs differ beyond {PARITY}: {', '.join(bad)}"}
    return {"label": "MEASURED", "value": cpu["seconds"] / gpu["seconds"], "parity": PARITY}


def blas():
    """Each OpenBLAS loaded (numpy's, scipy's) and its live thread count, read from the library itself."""
    import ctypes

    out = {"env": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
           "libraries": []}
    try:
        paths = sorted({line.split()[-1] for line in open("/proc/self/maps") if "openblas" in line.lower()})
    except OSError:
        paths = []  # not Linux: left empty, never assumed
    for path in paths:
        lib = ctypes.CDLL(path)
        names = [p + f for p in ("scipy_openblas_", "openblas_") for f in ("get_config64_", "get_config")]
        config = next((getattr(lib, n) for n in names if hasattr(lib, n)), None)
        threads = next((getattr(lib, n.replace("config", "num_threads")) for n in names
                        if hasattr(lib, n.replace("config", "num_threads"))), None)
        if config and threads:
            config.restype = ctypes.c_char_p
            out["libraries"].append({"path": Path(path).name, "config": config().decode(), "threads": int(threads())})
    return out


def cpu_config():
    """The timed CPU configuration as it is, not as intended."""
    return {"model": cpu_model(), "logical_cores": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
            "blas": blas(), "elementwise": "risk-set steps in numpy, one thread"}


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
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "screen.json")
    a = parser.parse_args(argv)
    args = workload(a.patients, a.candidates)
    X, t, e, C, s = args
    fingerprints = {"patient_rows": _sha(t, e, s), "X": _sha(np.nan_to_num(X, nan=np.pi)),
                    "C": _sha(C), "B": a.B}
    runs = {}
    seconds, cpu = _time(args, a.B, "numpy")
    runs["cpu"] = {"label": "MEASURED", "seconds": seconds, "backend": "numpy", "hardware": cpu_config(),
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
        bad = mismatched(cpu, gpu, SCREEN_OUTPUTS)
        runs["gpu"] = {"label": "MEASURED", "seconds": seconds, "backend": name, "hardware": hardware,
                       "maxT": gpu["maxT"], "p": gpu["p"], "matches_cpu": not bad, "mismatched": bad}
        seconds, gpu_batch = _time_batch(X, T, E, C, s, name)
        batch_bad = mismatched(cpu_batch, gpu_batch, BATCH_OUTPUTS)
        batch["gpu"] = {"label": "MEASURED", "seconds": seconds, "backend": name, "hardware": hardware,
                        "studies_per_hour": 3600 * a.studies / seconds,
                        "matches_cpu": not batch_bad, "mismatched": batch_bad}
    result = {
        "schema": "screen-bench/1",
        "data": "SYNTHETIC",
        "statistic": cpu["metadata"]["statistic"],
        "null": cpu["metadata"]["null"]["name"],
        "workload": {"patients": a.patients, "candidates": a.candidates, "resamples": a.B,
                     "clinical": cpu["metadata"]["clinical"]["shape"], "fingerprints": fingerprints},
        "runs": runs,
        "equal_work_speedup": speedup(runs["cpu"], runs["gpu"], runs["gpu"].get("mismatched")),
        "batched": {"studies": a.studies, "seed": 1, "what": "screen_batch: observed screen per study, null refit per study",
                    "outcomes_sha256": _sha(T, E), "runs": batch,
                    "equal_work_speedup": speedup(batch["cpu"], batch["gpu"], batch["gpu"].get("mismatched"))},
        "workflow_comparison": "bench/screen_vs_r14.py on the real producer's inputs (CPU workflow A/B, not equal work)",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"cpu_s": runs["cpu"]["seconds"], "gpu": runs["gpu"]["label"], "out": str(a.out)}))
    return result


if __name__ == "__main__":
    main()
