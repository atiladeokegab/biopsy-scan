"""Quiet-host rerun of the Numba-vs-NumPy comparison on a dedicated 8-core Modal CPU container.

    MODAL_PROFILE=<profile> uv run --with modal modal run bench/modal_quiet_cpu.py

Same code as bench/numba_negative.py (its worker is reused unchanged), same design and seeds.
Five interleaved rounds; each round starts one fresh process per mode in rotated order. Each process
times seed 0 as warm-up (excluded), then seeds 1 and 2 for both NumPy and Numba, alternating order.
Writes bench/results/modal-quiet-cpu-<utc>.json. Imports only modal and the standard library here.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
APP_DIR, DATA_DIR = "/root/preflight", "/data"
LOCAL_DATA = Path("/home/atilade/code/sota-parallelism-nov21/.worktrees/r14-cpu-baseline/data")
CORES, MEMORY_MIB, TIMEOUT_S, ROUNDS = 8.0, 16384, 2700, 5
MODES = {  # every mode pins BLAS/OMP pools to 1 and asks for 8 Numba threads unless it says otherwise
    "numba-8-omp-default": {"NUMBA_NUM_THREADS": "8", "NUMBA_THREADING_LAYER": "omp"},
    "numba-8-omp-passive": {"NUMBA_NUM_THREADS": "8", "NUMBA_THREADING_LAYER": "omp", "OMP_WAIT_POLICY": "PASSIVE"},
    "numba-8-workqueue": {"NUMBA_NUM_THREADS": "8", "NUMBA_THREADING_LAYER": "workqueue"},
}


def _tracked(rel: Path) -> bool:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    _tracked.files = getattr(_tracked, "files", None) or set(filter(None, out.split("\0")))
    return rel.as_posix() in _tracked.files


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")  # Numba's omp layer needs GNU OpenMP; without it Numba would silently use workqueue
    .uv_sync(str(REPO))
    .env({"DATA_DIR": DATA_DIR, "PYTHONPATH": APP_DIR})
    .add_local_file(LOCAL_DATA / "charville_labels.csv", f"{DATA_DIR}/charville_labels.csv", copy=True)
    .add_local_file(LOCAL_DATA / "charville_cells.npz", f"{DATA_DIR}/charville_cells.npz", copy=True)
    .workdir(APP_DIR)
    .add_local_dir(REPO, APP_DIR, ignore=lambda rel: not _tracked(rel))
)
app = modal.App("preflight-quiet-cpu")


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


@app.function(image=image, cpu=CORES, memory=MEMORY_MIB, timeout=TIMEOUT_S)
def rerun() -> dict:
    import os
    import sys
    import time

    sys.path.insert(0, APP_DIR)
    from bench.numba_negative import build_neighbours

    host = {"cpu_count": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0)),
            "cgroup_cpu_max": _read("/sys/fs/cgroup/cpu.max"), "load_average_start": list(os.getloadavg()),
            "cpu_model": next((line.split(":", 1)[1].strip() for line in (_read("/proc/cpuinfo") or "").splitlines()
                               if line.startswith("model name")), None),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"), "requested_cores": CORES}
    t = time.perf_counter()
    nbr = "/tmp/nbr.pkl"
    build_neighbours(nbr)
    prep_s = time.perf_counter() - t
    runs, names = [], list(MODES)
    for rnd in range(ROUNDS):
        for k in range(len(names)):
            mode = names[(rnd + k) % len(names)]  # rotate so no mode always runs first or last
            env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", **MODES[mode]}
            p = subprocess.run([sys.executable, "-m", "bench.numba_negative", "--worker", nbr], cwd=APP_DIR,
                               env=env, capture_output=True, text=True, timeout=900)
            if p.returncode:
                raise RuntimeError(f"{mode} round {rnd} failed:\n{p.stderr[-3000:]}")
            run = {"mode": mode, "round": rnd, "position": k, **json.loads(p.stdout.strip().splitlines()[-1])}
            runs.append(run)
            st = run["studies"]
            print(mode, rnd, run["threading_layer"],
                  "numpy", round(sum(s["numpy"]["wall_s"] for s in st) / len(st), 3),
                  "numba", round(sum(s["numba"]["wall_s"] for s in st) / len(st), 3), flush=True)
    host["load_average_end"] = list(os.getloadavg())
    return {"host": host, "neighbour_build_s": prep_s, "runs": runs}


@app.local_entrypoint()
def main():
    start = datetime.now(timezone.utc)
    out = rerun.remote()
    end = datetime.now(timezone.utc)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    out.update({"status": "MEASURED", "source_commit": commit, "harness": "bench/numba_negative.py worker, unchanged",
                "modal": {"app": "preflight-quiet-cpu", "cpu": CORES, "memory_mib": MEMORY_MIB,
                          "started_utc": start.isoformat(), "finished_utc": end.isoformat(),
                          "wall_s": (end - start).total_seconds()}})
    path = REPO / "bench" / "results" / f"modal-quiet-cpu-{start:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"Saved {path}")
