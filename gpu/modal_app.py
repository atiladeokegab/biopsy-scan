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
# Branch heads not merged yet, pinned by sha: R-15's stats.screen and R-14's bench.impl_a. _pinned() loads them.
PINS = {"stats.screen": ("8add7f9bfafcb12725bb9c685d5e6075b4f1643d", "stats/screen.py"),
        "bench.impl_a": ("54ad25a69cdfa3165ef28d0b13b9a4082e6a52dc", "bench/impl_a.py")}
PINNED_DIR = "/root/pins"
RAW = "https://raw.githubusercontent.com/atiladeokegab/sota-parallelism-nov21"
base = (
    # CUDA devel image: numba's CUDA target needs NVVM and libdevice from the toolkit, not just the driver.
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "git")
    .uv_sync(str(REPO))
    # shortcut: cupy is only for R-15's stats.screen GPU backend, not a project dependency; drop it if R-15 lands one.
    .uv_pip_install("cupy-cuda12x")
    .env({"DATA_DIR": DATA_DIR, "PYTHONPATH": APP_DIR, "CUDA_HOME": "/usr/local/cuda"})
    .run_commands(f"mkdir -p {PINNED_DIR}", *(f"curl -fsSL {RAW}/{sha}/{path} -o {PINNED_DIR}/{name}.py"
                                              for name, (sha, path) in PINS.items()))
)


def tracked_only(repo: Path):
    """An add_local_dir ignore that uploads git-tracked files only: no .env, keys, caches or other untracked files."""
    tracked: set[str] = set()

    def ignore(rel: Path) -> bool:
        if not tracked:  # lazy: the container re-imports this module and has no .git
            out = subprocess.run(["git", "ls-files", "-z"], cwd=repo, capture_output=True, text=True, check=True).stdout
            tracked.update(f for f in out.split("\0") if f)
        return rel.as_posix() not in tracked
    return ignore


SOURCE = dict(ignore=tracked_only(REPO))
image = base.workdir(APP_DIR).add_local_dir(REPO, APP_DIR, **SOURCE)
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


def _pinned(name: str):
    """Import `name` from this tree if it has landed, else from its pinned branch file; ImportError if neither."""
    import importlib
    import importlib.util
    import sys

    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    path = Path(PINNED_DIR) / f"{name}.py"
    if not path.exists():
        raise ImportError(f"{name} is neither in this tree nor pinned at {path}")
    importlib.import_module(name.rsplit(".", 1)[0])  # the parent package, for the module's relative imports
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


NEEDED = ("charville_labels.csv", "charville_raw_data.zip")


@app.function(image=image, volumes={DATA_DIR: data}, timeout=3600)
def fetch_data() -> list[str]:
    """CPU only: make sure the Charville cohort is in the volume; download only what is missing."""
    have = lambda: sorted(p.name for p in Path(DATA_DIR).iterdir())  # noqa: E731
    if not all((Path(DATA_DIR) / f).exists() for f in NEEDED):
        subprocess.run(["bash", "scripts/fetch_data.sh"], cwd=APP_DIR, check=True, env={**os.environ, "DATA_DIR": DATA_DIR})
        data.commit()
    missing = [f for f in NEEDED if not (Path(DATA_DIR) / f).exists()]
    if missing:  # Zenodo answers 403 to Modal's IPs: upload from a machine that has the files
        raise RuntimeError(f"volume lacks {missing}; run: uv run modal volume put preflight-data <dir>/<file> /<file>")
    return have()


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
    same = lambda: bool(np.array_equal(out["numpy_reference"], out["gpu"], equal_nan=True))  # noqa: E731
    parity = same()
    times = {name: [] for name in fns}
    for seed in range(reps):
        for name in (fns if seed % 2 == 0 else reversed(list(fns))):
            t0 = time.perf_counter()
            out[name] = fns[name](seed)
            times[name].append(time.perf_counter() - t0)
        parity = parity and same()
    rows = {name: _spread(ts) for name, ts in times.items()}
    return {"design": {**DESIGN, "cohort": "charville", "regions": len(cells["patient_of_region"]),
                       "cells": len(cells["cell_type"])},
            "seeds": list(range(reps)), "unit": "one study: patient and region sampling, crop, mis-call, recount, features",
            "parity": "exact, including NaN placement, on every seed and the warm-up", "parity_exact": parity,
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
        screen = _pinned("stats.screen").screen
    except ImportError as e:
        return {"status": "SKIPPED", "reason": f"stats.screen not importable ({e})"}
    import numpy as np

    rng = np.random.default_rng(0)
    X, t, ev = rng.normal(size=(159, 630)), rng.exponential(size=159), (rng.random(159) < 0.3).astype(float)
    # Backend results and the device record live under separate keys: a backend named "gpu" must not be
    # overwritten by the device's metadata.
    backends = {}
    for backend in ("numpy", "gpu", "cupy"):  # brief names "gpu"; the R-15 branch currently names it "cupy"
        try:
            t0 = time.perf_counter()
            out = screen(X, t, ev, n_perm=999, backend=backend)
            backends[backend] = {"s": time.perf_counter() - t0, "maxT": out["maxT"], "p": out["p"],
                                 "backend": out["metadata"]["backend"]}
        except (ValueError, RuntimeError) as e:
            backends[backend] = f"unavailable: {e}"
    return {"status": "MEASURED", "shape": list(X.shape), "n_perm": 999, "data": "synthetic null",
            "backends": backends, "gpu": gpu_info()}


@app.function(image=image, volumes={DATA_DIR: data}, gpu=DEFAULT_GPU, timeout=3600, memory=65536, cpu=16.0)
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
        impl_a = _pinned("bench.impl_a")
        d = impl_a.inputs("charville")
        parts["A_r14_full_fits"] = lambda: impl_a.time_full_run(d["X"], d["time"], d["event"], d["C"], d["strata"])[0]
    except ImportError as e:
        skipped["A_r14_full_fits"] = str(e)
        d = None
    try:
        screen = _pinned("stats.screen").screen
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


# ---- the design point: S studies end to end (gpu/design_point.py), the speed unit ----

SINGLE_THREAD = {k: "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}
CPU_CORES = 32.0
_POOL: dict = {}  # set before fork; workers inherit it


def _lab_one(seed):
    from bench.real_cpu import DESIGN
    from gpu import design_point as dp

    return dp.lab_study(_POOL["cells"], DESIGN, seed)


def _ours_one(seed):
    from bench.real_cpu import DESIGN
    from gpu import design_point as dp

    return dp.our_study(_POOL["cells"], DESIGN, seed, _POOL["kernel"], _POOL["score"])


@app.function(image=image, volumes={DATA_DIR: data}, cpu=CPU_CORES, memory=65536, timeout=3600, env=SINGLE_THREAD)
def cpu_design(method: str, seeds: list[int]) -> dict:
    """All cores, one single-threaded process per core: 'lab' (tree recount + 1,000 shuffles per study) or
    'ours' (CSR kernel on CPU + R-15 screen, numpy backend). Pool wall includes fork and per-worker JIT."""
    import multiprocessing as mp

    import numpy as np

    from bench.ladder import hardware

    entered = time.perf_counter()
    if method == "lab":
        from core.cohorts import load

        cells, prep = load("charville"), {}
        cells["_nbr"] = {r: None for r in range(len(cells["patient_of_region"]))}  # the tree path builds no CSR
        one = _lab_one
    elif method == "ours":
        from reimage import cpu

        cells, L, prep = _charville()
        prepared = {id(nbr): cpu.prepare(nbr, L["K"]) for nbr in cells["_nbr"].values()}
        screen = _pinned("stats.screen").screen
        _POOL["kernel"] = lambda nbr, xy, types, focal, K, **kw: cpu.reimage_region(
            nbr, xy, types, focal, K, prepared=prepared[id(nbr)], **kw)
        _POOL["score"] = lambda X, t, e, s: screen(X, t, e, strata=s, n_perm=0, backend="numpy")["z"]
        one = _ours_one
    else:
        raise ValueError("method must be 'lab' or 'ours'")
    _POOL["cells"] = cells
    workers = len(os.sched_getaffinity(0))
    t0 = time.perf_counter()
    with mp.get_context("fork").Pool(workers) as pool:
        out = pool.map(one, seeds, chunksize=1)
    wall = time.perf_counter() - t0
    res = {"method": method, "seeds": [seeds[0], seeds[-1]], "studies": len(seeds), "workers": workers,
           "compute_s": wall, "preparation_s": prep, "in_function_s": time.perf_counter() - entered,
           "host": hardware(), "load_avg_1m_end": os.getloadavg()[0]}
    if method == "lab":
        res["maxT"], res["p"] = (np.array(c) for c in zip(*out))
    else:
        res["z"], res["null_z"] = (np.array(c) for c in zip(*out))
    return res


@app.function(**gpu_function, single_use_containers=True)
def gpu_design(seeds: list[int], score_backend: str = "cupy") -> dict:
    """Our method on this container's GPU: CSR kernel + R-15 screen; first seed checked against NumPy."""
    import numpy as np

    from bench.real_cpu import DESIGN
    from core.reimage_ref import reimage
    from gpu import design_point as dp
    from reimage import gpu

    entered = time.perf_counter()
    cells, L, prep = _charville()
    prepared = {id(nbr): gpu.prepare(nbr, L["K"]) for nbr in cells["_nbr"].values()}
    screen = _pinned("stats.screen").screen

    def kernel(nbr, xy, types, focal, K, **kw):
        return gpu.reimage_region(nbr, xy, types, focal, K, prepared=prepared[id(nbr)], **kw)

    def score(X, t, e, s):
        return screen(X, t, e, strata=s, n_perm=0, backend=score_backend)["z"]

    t0 = time.perf_counter()  # warm-up (JIT, cupy kernels) doubles as the parity check; not timed below
    parity = bool(np.array_equal(reimage(cells, DESIGN, seeds[0]),
                                 reimage(cells, DESIGN, seeds[0], region_kernel=kernel), equal_nan=True))
    dp.our_study(cells, DESIGN, seeds[0], kernel, score)
    prep["warm_up_and_parity"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pairs = [dp.our_study(cells, DESIGN, s, kernel, score) for s in seeds]
    compute = time.perf_counter() - t0
    return {"seeds": seeds, "z": np.array([p[0] for p in pairs]), "null_z": np.array([p[1] for p in pairs]),
            "compute_s": compute, "preparation_s": prep, "parity_seed": seeds[0], "parity_exact": parity,
            "score_backend": score_backend, "in_function_s": time.perf_counter() - entered, "gpu": gpu_info()}


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


_LAUNCH: dict = {}


def _launch() -> dict:
    """The revision the run was launched from (recorded at start: the tree may move while it runs)."""
    if not _LAUNCH:
        git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
        _LAUNCH.update(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                       launched_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return _LAUNCH


PARITY_KEYS = {"parity_exact", "parity_exact_all", "bit_identical_to_1", "bit_identical_to_N1",
               "exact_all_containers", "gpu_matches_cpu_z", "bit_identical_across_gpus"}


def parity_failures(x, path: str = "") -> list[str]:
    """Every parity flag anywhere in the payload that is not exactly True."""
    if isinstance(x, dict):
        return [f for k, v in x.items() for f in ([f"{path}{k}"] if k in PARITY_KEYS and v is not True
                                                  else parity_failures(v, f"{path}{k}."))]
    return []


def _save(gpu: str, payload: dict) -> Path:
    """Write the evidence file. Fail-closed: if any parity flag is false the run is FAILED and the file holds no
    timing or ratio, only what failed; the process then exits non-zero."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    failed = parity_failures(payload)
    head = {"schema_version": 1, "runner": "modal", "gpu_requested": gpu, **_launch(), "utc": stamp}
    report = ({**head, "status": "FAILED", "kind": payload.get("kind"), "parity_failed": failed,
               "note": "parity false: no timing or ratio is published for this run"} if failed
              else {**head, "status": "MEASURED", **payload})
    path = REPO / "bench" / "results" / f"modal-{gpu.lower()}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=_jsonable) + "\n")
    print(path)
    if failed:
        raise SystemExit(f"FAILED: parity false at {', '.join(failed)}; no timing published")
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
    _launch()
    result = _tests(_check_gpu(gpu))
    _save(gpu, {"kind": "gpu_tests", "gpu_tests": result})
    print(f"{result['passed']} passed, {result['failed']} failed, {result['errors']} errors, "
          f"{result['skipped']} skipped on {result['gpu']['model']} (driver {result['gpu']['driver']})")
    if result["exit_code"]:
        raise SystemExit(result["exit_code"])


@app.local_entrypoint()
def run_ladder(gpu: str = DEFAULT_GPU, reps: int = 5, containers: int = 4, studies: int = 16):
    """make modal-ladder: kernel timing, fleet scaling and the R-15 hook, into one MEASURED file."""
    _launch()
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
    _launch()
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


# modal.com/pricing, read 2026-11-21 17:36Z, per second x 3600: A100 40GB $0.000583, H100 $0.001097, L4 $0.000222,
# physical core $0.0000131, GiB $0.00000222 (preemptible Function list price). gpu="A100" gave A100-SXM4-40GB.
RATES = {"A100": 2.0988, "H100": 3.9492, "L4": 0.7992, "cpu_core": 0.04716, "memory_gib": 0.007992}
SPEND = REPO / "bench" / "results" / "modal-spend.json"
CAP_USD = 150.0


def _gpu_rate(gpu: str) -> float:
    return RATES[gpu] + 8 * RATES["cpu_core"] + 32 * RATES["memory_gib"]  # gpu_function's 8 cores, 32 GiB


def _spend(run: str, est_usd: float, basis: str) -> None:
    """Log the estimate before a run; refuse when the running total would pass the cap."""
    log = json.loads(SPEND.read_text())
    total = round(log["entries"][-1].get("running_total_usd", 0) + est_usd, 2) if log["entries"] else est_usd
    if total > CAP_USD:
        raise SystemExit(f"{run}: running total ${total} would pass the ${CAP_USD} cap; stopping, tell the lead")
    log["entries"].append({"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "run": run,
                           "est_usd": round(est_usd, 2), "basis": basis, "running_total_usd": total})
    SPEND.write_text(json.dumps(log, indent=2) + "\n")
    print(f"spend: {run} est ${est_usd:.2f}, running total ${total}")


@app.local_entrypoint()
def run_design(gpu: str = DEFAULT_GPU, studies: int = 1000, fleet: str = "1,2,4,8,16", lab_studies: int = 1000,
               score_backend: str = "cupy"):
    """make modal-design: one design point of `studies` Charville studies, every slot on the same unit."""
    import sys

    import numpy as np

    sys.path.insert(0, str(REPO))  # the local side needs the repo's packages too
    from gpu import design_point as dp

    _launch()
    _check_gpu(gpu)
    ns = sorted({int(n) for n in fleet.split(",")})
    if not 1 <= lab_studies <= studies or ns[0] != 1 or ns[-1] > studies:
        raise SystemExit("need 1 <= lab_studies <= studies and a fleet list starting at 1")
    print("volume:", fetch_data.remote())
    seeds = list(range(studies))
    cpu_usd = 2 * (CPU_CORES * RATES["cpu_core"] + 64 * RATES["memory_gib"]) * 0.5  # two CPU boxes, <= 30 min each
    _spend(f"design-point CPU slots (a, b), {studies} studies", cpu_usd, "2 x (32 cores + 64 GiB) x 0.5 h")
    lab_call = cpu_design.spawn("lab", seeds[:lab_studies])
    ours_call = cpu_design.spawn("ours", seeds)
    gfn = gpu_design.with_options(gpu=gpu)
    runs, merged = {}, {}
    for n in ns:
        hours = (n * 180 + studies * 1.5) / 3600  # per container ~3 min start-up and prep; ~1.5 s/study on 1 GPU
        _spend(f"design-point fleet N={n} x {gpu}, {studies} studies", hours * _gpu_rate(gpu),
               f"{hours:.2f} GPU-h x ${_gpu_rate(gpu):.2f}/h")
        shards = [a.tolist() for a in np.array_split(seeds, n)]
        parts, wall = _timed_call(lambda: list(gfn.map(shards, kwargs={"score_backend": score_backend})))
        merged[n] = (np.concatenate([p["z"] for p in parts]), np.concatenate([p["null_z"] for p in parts]))
        slowest = max(parts, key=lambda p: p["in_function_s"])
        runs[str(n)] = {"containers_requested": n, "containers_seen": len({p["gpu"]["modal_task_id"] for p in parts}),
                        "gpus": sorted({p["gpu"]["model"] for p in parts}),
                        "compute_s": max(p["compute_s"] for p in parts), "modal_wall_s": wall,
                        "startup_and_transfer_s": wall - slowest["in_function_s"],
                        "preparation_s_slowest": slowest["preparation_s"],
                        "parity_exact_all": all(p["parity_exact"] for p in parts),
                        "bit_identical_to_N1": bool(all(np.array_equal(a, b) for a, b in zip(merged[n], merged[1])))}
        print(f"  N={n}: compute {runs[str(n)]['compute_s']:.1f} s, wall {wall:.1f} s")
    t1 = runs["1"]["compute_s"]
    for n in ns:
        r = runs[str(n)]
        r["speedup_vs_N1"] = t1 / r["compute_s"]
        r["efficiency"] = t1 / (n * r["compute_s"])
    lab, ours_cpu = lab_call.get(), ours_call.get()
    calls = {}
    for name, (z, z0) in (("b_ours_cpu", (ours_cpu["z"], ours_cpu["null_z"])), ("c_ours_1_gpu", merged[1])):
        c, cut = dp.our_calls(z, z0)
        calls[name] = {"threshold": cut, "positives": int(c.sum()), "agreement_with_lab": dp.agreement(
            lab["p"], c[:lab_studies])}
    lab_scale = studies / lab_studies
    unit = {
        "a_lab_all_cores": {"label": "MEASURED" if lab_scale == 1 else "ESTIMATE", "seconds": lab["compute_s"] * lab_scale,
                            "measured_studies": lab_studies, "measured_seconds": lab["compute_s"],
                            "basis": "measured" if lab_scale == 1 else f"linear: {lab_studies} measured studies x {lab_scale:g}",
                            "workers": lab["workers"], "host": lab["host"], "positives": int((lab["p"] <= dp.ALPHA).sum())},
        "b_ours_cpu_all_cores": {"label": "MEASURED", "seconds": ours_cpu["compute_s"], "workers": ours_cpu["workers"],
                                 "host": ours_cpu["host"]},
        "c_ours_1_gpu": {"label": "MEASURED", "seconds": t1, "gpu": runs["1"]["gpus"]},
        "d_ours_n_gpu": {"label": "MEASURED", "runs": runs},
    }
    payload = {"kind": "design_point", "studies": studies, "unit": f"one design point: {studies} Charville studies end "
               "to end (re-image, score, call), bench/real_cpu.py DESIGN", "methods": dp.__doc__, "slots": unit,
               "e_gpu_parity": {"exact_all_containers": all(r["parity_exact_all"] for r in runs.values()),
                                "suite": "bench/results/modal-a100-20261121T164808Z.json: 39 passed, 1 skipped"},
               "f_same_call": {k: f"{v['agreement_with_lab']['same_call']} of {v['agreement_with_lab']['of']}"
                               for k, v in calls.items()},
               "calls": calls, "score_backend": score_backend,
               "timing": "seconds = in-container compute for the whole unit; start-up, data load and adjacency apart"}
    _save(gpu, payload)
    for k, v in unit.items():
        if "seconds" in v:
            print(f"  {k}: {v['seconds']:.1f} s ({v['label']})")
    print("  same call:", payload["f_same_call"])


@app.local_entrypoint()
def run_gpu_vs(studies: int = 1000, blocks: int = 4, gpus: str = "A100,H100"):
    """make modal-gpu-vs: the same seeds on one container of each GPU, block by block, both GPUs at once per block
    and the launch order alternating; winner by dollars per study of in-container compute."""
    import numpy as np

    _launch()
    names = [_check_gpu(g) for g in gpus.split(",")]
    print("volume:", fetch_data.remote())
    hours = sum(blocks * 120 + studies * 1.2 for _ in names) / 3600  # ~2 min start-up per block, ~1.2 s/study
    _spend(f"GPU comparison {'/'.join(names)}, {studies} studies", hours * max(map(_gpu_rate, names)),
           f"{hours:.2f} GPU-h x ${max(map(_gpu_rate, names)):.2f}/h")
    rows = {g: {"compute_s": [], "startup_and_transfer_s": [], "gpu": None} for g in names}
    features = {g: [] for g in names}
    for b, shard in enumerate(np.array_split(range(studies), blocks)):
        order = names if b % 2 == 0 else names[::-1]
        t0 = time.perf_counter()
        calls = {g: fleet_shard.with_options(gpu=g).spawn(shard.tolist()) for g in order}
        for g in order:
            out = calls[g].get()
            rows[g]["compute_s"].append(out["compute_s"])
            rows[g]["startup_and_transfer_s"].append(time.perf_counter() - t0 - out["in_function_s"])
            rows[g]["gpu"] = out["gpu"]
            features[g].append(out["features"])
        print(f"  block {b}: " + ", ".join(f"{g} {rows[g]['compute_s'][-1]:.1f} s" for g in names))
    for g in names:
        r = rows[g]
        r["total_compute_s"] = sum(r["compute_s"])
        r["s_per_study"] = r["total_compute_s"] / studies
        r["container_usd_per_hour"] = _gpu_rate(g)
        r["usd_per_1000_studies"] = r["s_per_study"] * 1000 / 3600 * _gpu_rate(g)
    ref = np.concatenate(features[names[0]])
    same = all(np.array_equal(np.concatenate(features[g]), ref, equal_nan=True) for g in names)
    fastest = min(names, key=lambda g: rows[g]["total_compute_s"])
    cheapest = min(names, key=lambda g: rows[g]["usd_per_1000_studies"])
    payload = {"kind": "gpu_vs", "studies": studies, "blocks": blocks, "seeds": [0, studies - 1],
               "design": "bench/real_cpu.py DESIGN on Charville, per-study GPU kernel (reimage.run)",
               "rows": rows, "bit_identical_across_gpus": bool(same), "fastest": fastest, "cheapest_per_study": cheapest,
               "speed_ratio": max(r["total_compute_s"] for r in rows.values()) / rows[fastest]["total_compute_s"],
               "rates_source": "bench/results/modal-spend.json rates_usd_per_hour",
               "timing": "compute_s is in-container, after load, adjacency and JIT warm-up; start-up reported apart"}
    _save("-".join(names), payload)
    for g in names:
        print(f"  {g} ({rows[g]['gpu']['model']}): {rows[g]['s_per_study']:.3f} s/study, "
              f"${rows[g]['usd_per_1000_studies']:.2f} per 1,000 studies")
    print(f"  fastest {fastest} x{payload['speed_ratio']:.2f}; cheapest per study {cheapest}; bit-identical {same}")
