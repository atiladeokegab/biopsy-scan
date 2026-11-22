"""Utilisation trace of the CPU fused arm and the batched fused GPU arm on one Modal H100 (bench/modal_gpu_fused.py's design).

    DATA_DIR=<dir> MODAL_PROFILE=fhl-08 uv run --with modal modal run bench/modal_gpu_fused_usage.py

One container, each arm once: CPU fused (8 workers), then GPU fused (every study in one launch). A sampler thread
records every 100 ms: seconds since the trace began, the arm and stage marker, busy CPU cores, GPU utilisation %
and GPU memory used. Samples are raw. busy_cores is CPU time (utime + stime, 10 ms ticks) gained by every process
in the container between samples, over the interval; a worker that exits takes its total with it, so a negative
step is clamped to 0. GPU figures are nvidia-smi's own (-lms 100; its utilisation is averaged over its sample period).
Writes bench/results/gpu-fused-usage-h100-<utc>.json. Refuses a non-H100, a dirty tracked tree and a ceiling over $2.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bench.modal_design import APP_DIR, CORES, MEMORY_MIB, SINGLE, image  # noqa: E402
from bench.modal_gpu_fused import GPU, RATES, USD_PER_HOUR  # noqa: E402

CAP_USD, TIMEOUT_S, PERIOD_S = 2.0, 1200, 0.1
app = modal.App("preflight-gpu-fused-usage")


def _cpu_seconds() -> float:
    total = 0
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                f = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
                total += int(f[11]) + int(f[12])  # utime, stime
            except (OSError, IndexError, ValueError):
                pass
    return total / os.sysconf("SC_CLK_TCK")


@app.function(image=image, gpu=GPU, cpu=CORES, memory=MEMORY_MIB, timeout=TIMEOUT_S, env=SINGLE)
def run(studies: int) -> dict:
    import multiprocessing as mp
    import shutil
    import threading
    from concurrent.futures import ProcessPoolExecutor
    from statistics import median
    from time import perf_counter, sleep

    import numpy as np

    sys.path.insert(0, APP_DIR)
    from bench import design_batch as db, modal_gpu_fused as h
    from core.cohorts import load
    from reimage import gpu_fused as gf

    nsmi = shutil.which("nvidia-smi")
    sku = subprocess.run([nsmi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    if "H100" not in sku or len(sku.splitlines()) != 1:
        raise RuntimeError(f"refusing {sku!r}: this trace is for exactly one H100")
    mark, gpu_now, samples, t_zero = {"arm": "setup", "stage": "prepare"}, [None, None], [], perf_counter()
    smi = subprocess.Popen([nsmi, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", "100"],
                           stdout=subprocess.PIPE, text=True)

    def read_gpu():
        for line in smi.stdout:
            try:
                gpu_now[:] = [float(v) for v in line.split(",")]
            except ValueError:
                pass

    def sample():
        last_t, last_c = perf_counter(), _cpu_seconds()
        while mark["arm"] != "done":
            sleep(PERIOD_S)
            t, c = perf_counter(), _cpu_seconds()
            samples.append({"t_s": round(t - t_zero, 4), "arm": mark["arm"], "stage": mark["stage"],
                            "busy_cores": round(max(0.0, (c - last_c) / (t - last_t)), 3),
                            "gpu_util_pct": gpu_now[0], "gpu_mem_used_mib": gpu_now[1]})
            last_t, last_c = t, c

    threads = [threading.Thread(target=f, daemon=True) for f in (read_gpu, sample)]
    for th in threads:
        th.start()

    def at(arm, stage):
        mark.update(arm=arm, stage=stage)

    fork = mp.get_context("fork")
    cells = load("charville")
    plan = {s: studies for s in db.STREAMS}
    setup = db.prepare("fused", cells, plan)
    jobs = [(stream, i) for stream, n in plan.items() for i in range(n)]
    n = len(jobs)
    at("setup", "cpu_warm")
    db.warm("fused")
    at("cpu_fused", "studies")
    t = perf_counter()
    with ProcessPoolExecutor(int(CORES), mp_context=fork, initializer=db._single_thread) as pool:
        recs = list(pool.map(db.one_study, jobs))
    cpu_wall = perf_counter() - t

    at("setup", "gpu_adjacency_upload")
    src = [db._S["studies"][stream][i] for stream, i in jobs]
    seeds = [s["image_seed"] for s in src]
    time_, event, strata = (np.stack([s[k] for s in src]) for k in ("time", "event", "strata"))
    h._PLAN["st"] = st = gf.static(cells, db.design())
    planners = ProcessPoolExecutor(int(CORES), mp_context=fork)  # forked before the CUDA context exists
    list(planners.map(h._plan_part, [seeds[:1]] * int(CORES) * 2))
    F = gf.Fused(cells, db.design(), st=st)
    chunks = [seeds[k:k + 125] for k in range(0, n, 125)]

    def gpu(keep_X=False, timed=False):
        t = perf_counter()
        pl = gf.merge(list(planners.map(h._plan_part, chunks)))
        if timed:  # only the timed launch carries the arm marker, not the warm-up
            at("gpu_fused", "device")
        out = F.evaluate(pl, time_, event, strata, keep_X=keep_X)
        return out, perf_counter() - t

    at("setup", "gpu_warm")
    gpu()
    at("setup", "gpu_parity_pass")
    ref, _ = gpu(keep_X=True)
    x_same = [db.sha(X) == r["x_sha"] for X, r in zip(ref.pop("X"), recs)]
    sleep(1.0)  # a second of idle samples before the timed launch
    at("gpu_fused", "host_plan")
    out, gpu_wall = gpu(timed=True)
    at("setup", "after")
    sleep(0.5)
    at("done", "done")
    threads[1].join()
    smi.terminate()
    planners.shutdown()
    zc = np.stack([r["z"] for r in recs])
    mc, mg = np.abs(zc).max(1), out["maxT"]
    plant = setup["plant_feature"]
    cuts = {a: db.summarize({"arm": "fused", "records": [{"stream": s, "i": i, "z": z} for (s, i), z in zip(jobs, zz)]},
                            plant)["threshold"] for a, zz in (("cpu_fused", zc), ("gpu_fused", out["z"]))}
    judged = np.array([s != "calibration" for s, _ in jobs])
    agree = ((mc > cuts["cpu_fused"]) == (mg > cuts["gpu_fused"]))[judged] if None not in cuts.values() else judged[judged]
    parity = {"x_bitwise_k": int(sum(x_same)), "x_compared": n, "maxT_max_abs_diff": float(np.abs(mg - mc).max()),
              "thresholds": cuts, "decisions_agree_k": int(agree.sum()), "decisions_compared": int(judged.sum()),
              "timed_z_identical_to_parity_pass": bool(np.array_equal(out["z"], ref["z"]))}
    parity["all"] = bool(sum(x_same) == n and parity["maxT_max_abs_diff"] <= 1e-9 and agree.all()
                         and parity["timed_z_identical_to_parity_pass"])
    arms = {}
    for arm, wall in (("cpu_fused", cpu_wall), ("gpu_fused", gpu_wall)):
        rows = [s for s in samples if s["arm"] == arm]
        util = [s["gpu_util_pct"] for s in rows if s["gpu_util_pct"] is not None]
        arms[arm] = {"wall_s": wall, "wall_per_study_s": wall / n, "samples": len(rows),
                     "busy_cores_median": median(s["busy_cores"] for s in rows) if rows else None,
                     "gpu_util_pct_median": median(util) if util else None, "gpu_util_pct_peak": max(util) if util else None,
                     "gpu_mem_used_mib_peak": max((s["gpu_mem_used_mib"] or 0) for s in rows) if rows else None}
    arms["gpu_fused"].update(studies_per_launch=n, stages=out["stages"])
    return {"host": {"gpu": sku, "cpu_count": os.cpu_count(), "affinity_size": len(os.sched_getaffinity(0))},
            "evaluated_studies": n, "arms": arms, "parity": parity, "sample_period_s": PERIOD_S, "samples": samples}


def _git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


@app.local_entrypoint()
def main(studies: int = 1000):
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit first: the run refuses a dirty tracked tree")
    ceiling = TIMEOUT_S / 3600 * USD_PER_HOUR
    if ceiling > CAP_USD:
        raise SystemExit(f"refused: ${ceiling:.2f} ceiling passes the ${CAP_USD:.0f} cap")
    commit, start = _git("rev-parse", "HEAD"), datetime.now(timezone.utc)
    out, error = None, None
    try:
        out = run.remote(studies)
    except Exception as e:  # noqa: BLE001  a failed run is recorded with its spend
        error = f"{type(e).__name__}: {e}"
    wall = (datetime.now(timezone.utc) - start).total_seconds()
    ok = out is not None and out["parity"]["all"]
    result = {"status": "FAILED" if not ok else "MEASURED" if studies == 1000 else "SMOKE", "source_commit": commit, "error": error,
              "labels": {"timings": "MEASURED", "utilisation": "MEASURED (raw 100 ms samples)", "outcomes": "SYNTHETIC", "spend": "ESTIMATE"},
              "sample_fields": {"t_s": "seconds since the sampler started", "arm": "cpu_fused, gpu_fused or setup (not timed)",
                                "stage": "marker inside the arm", "busy_cores": "CPU seconds gained by all container processes / interval",
                                "gpu_util_pct": "nvidia-smi utilization.gpu, latest reading", "gpu_mem_used_mib": "nvidia-smi memory.used"},
              "design": {"cohort": "charville", "studies_per_stream": studies, "as": "bench/modal_gpu_fused.py"},
              "modal": {"app": "preflight-gpu-fused-usage", "gpu_requested": GPU, "cpu": CORES, "memory_mib": MEMORY_MIB,
                        "started_utc": start.isoformat()},
              "spend": {"cap_usd": CAP_USD, "ceiling_usd": round(ceiling, 2), "est_usd": round(wall / 3600 * USD_PER_HOUR, 3),
                        "modal_wall_s": round(wall, 1), "rates_usd_per_hour": RATES,
                        "basis": "ESTIMATE at list price: one container x local wall (an upper bound: includes image build and queueing)"}}
    result.update(out or {})
    path = REPO / "bench" / "results" / f"gpu-fused-usage-h100-{start:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(result["status"], error, json.dumps(result.get("arms")), json.dumps(result.get("parity")), json.dumps(result["spend"]))
    print(f"Saved {path}")
