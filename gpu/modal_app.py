"""Preflight on real Modal GPUs: the GPU tests, the kernel timing and the multi-container fleet.

    uv run modal run gpu/modal_app.py::run_tests  --gpu A100
    uv run modal run gpu/modal_app.py::run_ladder --gpu A100 --reps 5 --containers 4 --studies 16

Each run writes bench/results/modal-<gpu>-<utc>.json, status MEASURED. Every number comes from
inside a Modal container on the GPU named there. Container start-up (scheduling, image pull, volume
mount, data load, adjacency) is reported apart from the timed compute and never enters a speed-up.
Only the local entrypoints call Modal; importing this file needs no credentials or network.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

GPUS = ("A100", "H100", "L4")
DEFAULT_GPU = "A100"
REPO = Path(__file__).resolve().parents[1]
APP_DIR = "/root/preflight"
DATA_DIR = "/data"
TEST_FILES = ("reimage/test_backends.py", "reimage/test_expression.py", "reimage/test_fleet.py")

app = modal.App("preflight-gpu")
data = modal.Volume.from_name("preflight-data", create_if_missing=True)
base = (
    # CUDA devel image: numba's CUDA target needs NVVM and libdevice from the toolkit, not just the driver.
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "git")
    .uv_sync(str(REPO))
    # shortcut: cupy is only for R-15's stats.screen GPU backend, not a project dependency; drop it if R-15 lands one.
    .uv_pip_install("cupy-cuda12x")
    .env({"DATA_DIR": DATA_DIR, "PYTHONPATH": APP_DIR, "CUDA_HOME": "/usr/local/cuda"})
)
SOURCE = dict(ignore=[".venv", ".git", ".worktrees", "data", "demo", "results", "bench/results", "**/__pycache__",
                      "node_modules"])
image = base.workdir(APP_DIR).add_local_dir(REPO, APP_DIR, **SOURCE)
# ab_screen's code: R-15 (stats.screen) merged with R-14 (bench/impl_a.py) at pinned heads from the public repo.
PINS = {"r15-gpu-score-screen": "8add7f9bfafcb12725bb9c685d5e6075b4f1643d",
        "r14-cpu-baseline": "54ad25a69cdfa3165ef28d0b13b9a4082e6a52dc"}
PINNED_DIR = "/root/pinned"
ab_image = base.run_commands(
    f"git clone -q https://github.com/atiladeokegab/sota-parallelism-nov21 {PINNED_DIR}",
    f"cd {PINNED_DIR} && git checkout -q {PINS['r15-gpu-score-screen']}"
    f" && git -c user.email=ci@invalid -c user.name=ci merge -q --no-edit {PINS['r14-cpu-baseline']}",
).env({"PYTHONPATH": PINNED_DIR}).workdir(PINNED_DIR)
gpu_function = dict(image=image, volumes={DATA_DIR: data}, gpu=DEFAULT_GPU, timeout=3600, memory=32768, cpu=8.0)


def gpu_info() -> dict:
    """The device every GPU number belongs to: nvidia-smi's model, memory and driver, plus the CUDA versions."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()
    name, memory, driver = (s.strip() for s in q[0].split(","))
    header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30, check=False).stdout
    driver_cuda = re.search(r"CUDA Version:\s*([\d.]+)", header)
    from numba import cuda

    runtime = cuda.runtime.get_version()
    return {"model": name, "memory": memory, "driver": driver, "count": len(q),
            "cuda_driver_api": driver_cuda.group(1) if driver_cuda else None,
            "cuda_runtime": ".".join(map(str, runtime)), "all": q,
            "modal_task_id": os.environ.get("MODAL_TASK_ID")}


@app.function(image=image, volumes={DATA_DIR: data}, timeout=3600)
def fetch_data() -> list[str]:
    """CPU only: download the Zenodo cohort into the volume once; reruns keep what is there."""
    subprocess.run(["bash", "scripts/fetch_data.sh"], cwd=APP_DIR, check=True, env={**os.environ, "DATA_DIR": DATA_DIR})
    data.commit()
    return sorted(p.name for p in Path(DATA_DIR).iterdir())


@app.function(**gpu_function)
def gpu_tests() -> dict:
    """The backend, expression and fleet tests on a real GPU, so the CUDA-gated tests run instead of skipping."""
    import xml.etree.ElementTree as ET

    start = time.perf_counter()
    junit = "/tmp/junit.xml"
    proc = subprocess.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}",
                           *TEST_FILES], cwd=APP_DIR, capture_output=True, text=True)
    suite = ET.parse(junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    total, failed, errors, skipped = (int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    skips = [f"{c.get('name')}: {c.find('skipped').get('message', '')}" for c in suite.iter("testcase")
             if c.find("skipped") is not None]
    return {"exit_code": proc.returncode, "passed": total - failed - errors - skipped, "failed": failed,
            "errors": errors, "skipped": skipped, "skips": skips, "files": list(TEST_FILES),
            "summary": proc.stdout.strip().splitlines()[-1:], "in_function_s": time.perf_counter() - start,
            "gpu": gpu_info()}


def _charville():
    """Charville cells with every region's adjacency prebuilt (timed: it is preparation, not the kernel)."""
    from core.cohorts import load
    from core.truth import layout, neighbours

    t0 = time.perf_counter()
    cells = load("charville")
    t1 = time.perf_counter()
    L = layout(cells)
    cells["_nbr"] = {r: neighbours(L["xy"](r)) for r in range(len(cells["patient_of_region"]))}
    return cells, L, {"load": t1 - t0, "adjacency": time.perf_counter() - t1}


def _spread(xs: list[float]) -> dict:
    import statistics

    return {"median_s": statistics.median(xs), "min_s": min(xs), "max_s": max(xs), "samples_s": xs}


@app.function(**gpu_function)
def time_kernel(reps: int = 5) -> dict:
    """bench/real_cpu.py's Charville design point: NumPy reference vs the GPU kernel, one study per sample.

    Same seeds for both, alternating order, warm-up (JIT, upload) excluded, exact cell-for-cell parity.
    """
    import numpy as np

    from bench.ladder import hardware
    from bench.real_cpu import DESIGN
    from core.reimage_ref import reimage
    from reimage import gpu

    entered = time.perf_counter()
    cells, L, prep = _charville()
    t0 = time.perf_counter()
    prepared = {id(nbr): gpu.prepare(nbr, L["K"]) for nbr in cells["_nbr"].values()}
    prep["gpu_upload"] = time.perf_counter() - t0

    def gpu_region(nbr, xy, types, focal, K, **kwargs):
        return gpu.reimage_region(nbr, xy, types, focal, K, prepared=prepared[id(nbr)], **kwargs)

    fns = {"numpy_reference": lambda s: reimage(cells, DESIGN, s),
           "gpu": lambda s: reimage(cells, DESIGN, s, region_kernel=gpu_region)}
    first_call, out = {}, {}
    for name, fn in fns.items():
        t0 = time.perf_counter()
        out[name] = fn(0)
        first_call[name] = time.perf_counter() - t0
    assert np.array_equal(out["numpy_reference"], out["gpu"], equal_nan=True), "warm-up parity failed"
    times = {name: [] for name in fns}
    for seed in range(reps):
        for name in (fns if seed % 2 == 0 else reversed(list(fns))):
            t0 = time.perf_counter()
            out[name] = fns[name](seed)
            times[name].append(time.perf_counter() - t0)
        assert np.array_equal(out["numpy_reference"], out["gpu"], equal_nan=True), f"parity failed at seed {seed}"
    rows = {name: _spread(ts) for name, ts in times.items()}
    return {"design": {**DESIGN, "cohort": "charville", "regions": len(cells["patient_of_region"]),
                       "cells": len(cells["cell_type"])},
            "seeds": list(range(reps)), "unit": "one study: patient and region sampling, crop, mis-call, recount, features",
            "parity": f"exact, including NaN placement, on all {reps} seeds and the warm-up",
            "rows": rows, "speedup_median": rows["numpy_reference"]["median_s"] / rows["gpu"]["median_s"],
            "first_call_s": first_call, "preparation_s": prep,
            "in_function_s": time.perf_counter() - entered, "host": hardware(), "gpu": gpu_info()}


# One fresh container per shard, so N inputs really are N GPUs.
@app.function(**gpu_function, single_use_containers=True)
def fleet_shard(seeds: list[int]) -> dict:
    """One contiguous seed shard of GPU studies on this container's GPU (reimage.fleet's per-device work)."""
    import numpy as np

    from bench.real_cpu import DESIGN
    from reimage import gpu, run

    entered = time.perf_counter()
    cells, L, prep = _charville()
    design = {**DESIGN, "cells": cells, "backend": "gpu"}
    prepared = {("gpu", "v1", id(nbr), L["K"]): gpu.prepare(nbr, L["K"]) for nbr in cells["_nbr"].values()}
    t0 = time.perf_counter()
    run(design, np.array(seeds[:1]), _prepared=prepared)  # warm-up: JIT; the result is recomputed below
    prep["warm_up"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    features = run(design, np.array(seeds), _prepared=prepared)
    compute = time.perf_counter() - t0
    return {"seeds": seeds, "features": features, "compute_s": compute, "preparation_s": prep,
            "in_function_s": time.perf_counter() - entered, "gpu": gpu_info()}


@app.function(**gpu_function)
def screen_hook() -> dict:
    """R-15's stats.screen on the GPU when that branch has landed; a clean SKIPPED when it has not."""
    try:
        from stats.screen import screen
    except ImportError as e:
        return {"status": "SKIPPED", "reason": f"stats.screen not importable ({e})"}
    import numpy as np

    rng = np.random.default_rng(0)
    X, t, ev = rng.normal(size=(159, 630)), rng.exponential(size=159), (rng.random(159) < 0.3).astype(float)
    result = {"status": "MEASURED", "shape": list(X.shape), "n_perm": 999, "data": "synthetic null"}
    for backend in ("numpy", "gpu", "cupy"):  # brief names "gpu"; the R-15 branch currently names it "cupy"
        try:
            t0 = time.perf_counter()
            out = screen(X, t, ev, n_perm=999, backend=backend)
            result[backend] = {"s": time.perf_counter() - t0, "maxT": out["maxT"], "p": out["p"],
                               "backend": out["metadata"]["backend"]}
        except (ValueError, RuntimeError) as e:
            result[backend] = f"unavailable: {e}"
    result["gpu"] = gpu_info()
    return result


@app.function(image=ab_image, volumes={DATA_DIR: data}, gpu=DEFAULT_GPU, timeout=3600, memory=65536, cpu=16.0)
def ab_screen(reps: int = 3, n_perm: int = 999) -> dict:
    """One quiet container, Charville v2 (3,780 candidates): R-14 full fits (all cores) vs R-15 screen on CPU and GPU.

    Interleaved A, B, C, A, B, C ... so all three share one window; a part that will not import is SKIPPED.
    A is a workflow comparison (different outputs); B vs C is the equal-work pair.
    """
    from functools import partial

    import numpy as np

    from bench.ladder import hardware

    entered = time.perf_counter()
    parts, skipped, screens = {}, {}, {}
    try:
        from bench import impl_a

        d = impl_a.inputs("charville")
        parts["A_r14_full_fits"] = lambda: impl_a.time_full_run(d["X"], d["time"], d["event"], d["C"], d["strata"])[0]
    except ImportError as e:
        skipped["A_r14_full_fits"] = str(e)
        d = None
    try:
        from stats.screen import screen
        if d is None:
            raise ImportError("needs bench.impl_a.inputs for the shared Charville v2 inputs")
        C = d["C"] if d["C"].shape[1] else None

        def run_screen(backend, key=None):
            out = screen(d["X"], d["time"], d["event"], clinical=C, strata=d["strata"], n_perm=n_perm, backend=backend)
            if key:
                screens[key] = out

        for key, backend in (("B_r15_screen_cpu", "numpy"), ("C_r15_screen_gpu", "cupy")):
            run_screen(backend)  # warm-up: imports, BLAS and CUDA kernels; not timed
            parts[key] = partial(run_screen, backend, key)
    except (ImportError, RuntimeError) as e:  # RuntimeError: cupy or its device unavailable
        for key in ("B_r15_screen_cpu", "C_r15_screen_gpu"):
            if key not in parts:
                skipped[key] = str(e)
        parts.pop("C_r15_screen_gpu", None)
    prep = time.perf_counter() - entered
    samples = {k: [] for k in parts}
    load = [os.getloadavg()[0]]
    for _ in range(reps):
        for key, fn in parts.items():
            t0 = time.perf_counter()
            fn()
            samples[key].append(time.perf_counter() - t0)
            load.append(os.getloadavg()[0])
    result = {"inputs": {"cohort": "charville", "features": d and d["features"], "patients": d and len(d["time"]),
                         "candidates": d and int(d["X"].shape[1]), "events": d and int(np.sum(d["event"]))},
              "order": list(parts), "reps": reps, "n_perm": n_perm, "skipped": skipped, "pins": PINS,
              "rows": {k: _spread(v) for k, v in samples.items()}, "preparation_s": prep,
              "cores": len(os.sched_getaffinity(0)), "load_avg_1m": {"min": min(load), "max": max(load)},
              "host": hardware(), "gpu": gpu_info(), "in_function_s": time.perf_counter() - entered,
              "labels": {"A_vs_B": "workflow comparison: full Cox fits vs a score screen, different outputs",
                         "B_vs_C": "equal work: same statistic and resamples on CPU and GPU"}}
    if "B_r15_screen_cpu" in screens and "C_r15_screen_gpu" in screens:
        b, c = screens["B_r15_screen_cpu"], screens["C_r15_screen_gpu"]
        result["gpu_matches_cpu_z"] = bool(np.allclose(b["z"], c["z"], rtol=1e-9, atol=1e-9))
    return result


# ---- local side: validates arguments, calls Modal, writes the evidence file ----

def _check_gpu(gpu: str) -> str:
    if gpu not in GPUS:
        raise SystemExit(f"--gpu must be one of {', '.join(GPUS)}")
    return gpu


def _timed_call(fn, *args, **kwargs):
    """(result, wall seconds seen from here): wall minus in-function time is start-up plus transfer."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def _save(gpu: str, payload: dict) -> Path:
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {"schema_version": 1, "status": "MEASURED", "runner": "modal", "gpu_requested": gpu,
              "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
              "utc": stamp, **payload}
    path = REPO / "bench" / "results" / f"modal-{gpu.lower()}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=_jsonable) + "\n")
    print(path)
    return path


def _jsonable(x):
    tolist = getattr(x, "tolist", None)
    if tolist is None:
        raise TypeError(f"not JSON serialisable: {type(x).__name__}")
    return tolist()


def _tests(gpu: str) -> dict:
    result, wall = _timed_call(gpu_tests.with_options(gpu=gpu).remote)
    result["modal_wall_s"] = wall
    result["startup_and_transfer_s"] = wall - result["in_function_s"]
    return result


def _fleet(gpu: str, containers: int, studies: int) -> dict:
    """Studies 0..n-1 on 1 container, then sharded over N; merged features must be bit-identical."""
    import numpy as np

    fn = fleet_shard.with_options(gpu=gpu)
    seeds = list(range(studies))
    runs, merged = {}, {}
    for n in sorted({1, containers}):
        shards = [a.tolist() for a in np.array_split(seeds, n)]  # contiguous, larger first: reimage.fleet.shard_bounds
        parts, wall = _timed_call(lambda: list(fn.map(shards, order_outputs=True)))
        merged[n] = np.concatenate([p["features"] for p in parts])
        slowest = max(parts, key=lambda p: p["in_function_s"])
        runs[str(n)] = {
            "containers_requested": n,
            "containers_seen": len({p["gpu"]["modal_task_id"] for p in parts}),
            "gpus": sorted({p["gpu"]["model"] for p in parts}),
            "studies_per_container": [len(p["seeds"]) for p in parts],
            "modal_wall_s": wall,
            "compute_s_max": max(p["compute_s"] for p in parts),
            "in_function_s_max": slowest["in_function_s"],
            "startup_and_transfer_s": wall - slowest["in_function_s"],
            "preparation_s_slowest": slowest["preparation_s"],
            "studies_per_s_compute": studies / max(p["compute_s"] for p in parts),
            "studies_per_s_wall": studies / wall,
        }
    ref = merged[1]
    for n, arr in merged.items():
        runs[str(n)]["bit_identical_to_1"] = bool(arr.shape == ref.shape and np.array_equal(arr, ref, equal_nan=True))
    return {"studies": studies, "seeds": [0, studies - 1], "design": "bench/real_cpu.py DESIGN on Charville",
            "runs": runs, "note": "compute_s excludes start-up, data load, adjacency and warm-up; those are reported per run"}


@app.local_entrypoint()
def run_tests(gpu: str = DEFAULT_GPU):
    """make modal-gpu-tests: pytest on a real GPU; exits non-zero if any test failed."""
    result = _tests(_check_gpu(gpu))
    _save(gpu, {"kind": "gpu_tests", "gpu_tests": result})
    print(f"{result['passed']} passed, {result['failed']} failed, {result['errors']} errors, "
          f"{result['skipped']} skipped on {result['gpu']['model']} (driver {result['gpu']['driver']})")
    if result["exit_code"]:
        raise SystemExit(result["exit_code"])


@app.local_entrypoint()
def run_ladder(gpu: str = DEFAULT_GPU, reps: int = 5, containers: int = 4, studies: int = 16):
    """make modal-ladder: kernel timing, fleet scaling and the R-15 hook, into one MEASURED file."""
    _check_gpu(gpu)
    if reps < 3 or containers < 1 or studies < containers:
        raise SystemExit("need --reps >= 3, --containers >= 1 and --studies >= --containers")
    print("volume:", fetch_data.remote())
    kernel, wall = _timed_call(time_kernel.with_options(gpu=gpu).remote, reps)
    kernel["modal_wall_s"] = wall
    kernel["startup_and_transfer_s"] = wall - kernel["in_function_s"]
    payload = {"kind": "ladder", "kernel": kernel, "fleet": _fleet(gpu, containers, studies),
               "screen": screen_hook.with_options(gpu=gpu).remote(),
               "baseline_note": "numpy_reference runs in the same container as the GPU row; the EPYC rungs are in "
                                "bench/results/cpu-ladder.json (same DESIGN, other host)"}
    _save(gpu, payload)
    r = kernel["rows"]
    print(f"{kernel['gpu']['model']}: numpy {r['numpy_reference']['median_s']:.3f} s/study, "
          f"gpu {r['gpu']['median_s']:.3f} s/study, x{kernel['speedup_median']:.1f} (median of {reps}), parity exact")
    for n, row in payload["fleet"]["runs"].items():
        print(f"  {n} container(s): wall {row['modal_wall_s']:.1f} s, compute {row['compute_s_max']:.2f} s, "
              f"bit-identical {row['bit_identical_to_1']}")


@app.local_entrypoint()
def run_ab(gpu: str = DEFAULT_GPU, reps: int = 3):
    """make modal-ab: R-14 vs R-15 CPU vs R-15 GPU, interleaved on one 16-core GPU container."""
    _check_gpu(gpu)
    if reps < 3:
        raise SystemExit("need --reps >= 3")
    print("volume:", fetch_data.remote())
    result, wall = _timed_call(ab_screen.with_options(gpu=gpu).remote, reps)
    result["modal_wall_s"] = wall
    result["startup_and_transfer_s"] = wall - result["in_function_s"]
    _save(gpu, {"kind": "ab_screen", "ab_screen": result})
    for key, row in result["rows"].items():
        print(f"  {key}: median {row['median_s']:.2f} s [{row['min_s']:.2f}, {row['max_s']:.2f}]")
    print("  skipped:", result["skipped"] or "none")
