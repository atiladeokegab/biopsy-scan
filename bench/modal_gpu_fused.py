"""The batched fused GPU kernel (reimage.gpu_fused) on one Modal H100, against the CPU fused arm in the same container.

    DATA_DIR=<dir> MODAL_PROFILE=fhl-08 uv run --with modal modal run bench/modal_gpu_fused.py --spent-usd <task total>

The archived design point (bench/design_batch.py: --studies per stream of calibration, null and alternative).
One container: the CPU fused arm first (bench.design_batch.one_study in 8 forked single-thread workers, timed
--repeats times), then the GPU path on the same studies and outcomes. A parity pass downloads X and checks every
study's sha256 and z against the CPU arm; the timed passes do not download X. GPU wall = host plan (patient,
region and crop draws) + upload + kernels + z download, per --batches size (studies per launch).
Refuses any GPU that is not an H100, a dirty tracked tree, and a run whose ceiling would pass the $6 task cap.
Writes bench/results/gpu-fused-h100-<utc>.json with the estimated spend.
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
from bench.modal_design import APP_DIR, CORES, MEMORY_MIB, SINGLE, image  # noqa: E402  the design point's image

GPU, CAP_USD, TIMEOUT_S = "H100!", 6.0, 1500  # "!": no automatic upgrade to another SKU
RATES = {"H100": 3.9492, "cpu_core": 0.04716, "memory_gib": 0.007992,
         "source": "bench/results/modal-spend.json (modal.com/pricing read 2026-11-21T17:36Z)"}
USD_PER_HOUR = RATES["H100"] + CORES * RATES["cpu_core"] + MEMORY_MIB / 1024 * RATES["memory_gib"]
app = modal.App("preflight-gpu-fused")
_PLAN: dict = {}


def _plan_part(seeds):
    from reimage import gpu_fused as gf
    return gf.plan(_PLAN["st"], seeds)


def _spread(v):
    from statistics import median
    return {"median": median(v), "min": min(v), "max": max(v), "runs": list(v)}


@app.function(image=image, gpu=GPU, cpu=CORES, memory=MEMORY_MIB, timeout=TIMEOUT_S, env=SINGLE)
def run(studies: int, batches: list[int], repeats: int) -> dict:
    import hashlib
    import multiprocessing as mp
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    from time import perf_counter

    import numpy as np

    sys.path.insert(0, APP_DIR)
    from bench import design_batch as db
    from core.cohorts import load
    from reimage import gpu_fused as gf

    smi = subprocess.run([shutil.which("nvidia-smi"), "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    if "H100" not in smi or len(smi.splitlines()) != 1:
        raise RuntimeError(f"refusing {smi!r}: this benchmark is for exactly one H100")
    host = {"gpu": smi, "cpu_count": os.cpu_count(), "affinity_size": len(os.sched_getaffinity(0)),
            "load_average_start": list(os.getloadavg()), "thread_environment": {k: os.environ.get(k) for k in SINGLE}}
    fork = mp.get_context("fork")
    cells = load("charville")
    plan = {s: studies for s in db.STREAMS}
    setup = db.prepare("fused", cells, plan)
    jobs = [(stream, i) for stream, n in plan.items() for i in range(n)]
    n = len(jobs)

    # ---- CPU fused arm: bench.design_batch.run_arm's timed region, before CUDA starts (the pool forks) ----
    cpu_warm_s = db.warm("fused")
    cpu_walls = []
    for _ in range(repeats):
        t = perf_counter()
        with ProcessPoolExecutor(int(CORES), mp_context=fork, initializer=db._single_thread) as pool:
            recs = list(pool.map(db.one_study, jobs))
        cpu_walls.append(perf_counter() - t)
    host["load_average_after_cpu"] = list(os.getloadavg())

    # ---- GPU fused: the plan pool forks before the CUDA context exists ----
    src = [db._S["studies"][stream][i] for stream, i in jobs]
    seeds = [s["image_seed"] for s in src]
    time_, event, strata = (np.stack([s[k] for s in src]) for k in ("time", "event", "strata"))
    t = perf_counter()
    _PLAN["st"] = st = gf.static(cells, db.design())
    static_s = perf_counter() - t
    planners = ProcessPoolExecutor(int(CORES), mp_context=fork)
    list(planners.map(_plan_part, [seeds[:1]] * int(CORES) * 2))  # start every worker
    from numba import cuda
    from reimage.test_gpu_fused import check
    check("gpu")  # the unit parity case (X bit for bit, NaN bits included) on this device
    t = perf_counter()
    F = gf.Fused(cells, db.design(), st=st)
    cuda.synchronize()
    upload_s = perf_counter() - t

    def gpu(batch: int, keep_X: bool = False) -> dict:
        cuda.current_context().deallocations.clear()
        t = perf_counter()
        c = max(1, min(125, batch // int(CORES)))  # studies per plan task
        parts = list(planners.map(_plan_part, [seeds[k:k + c] for k in range(0, n, c)]))
        plan_s = perf_counter() - t
        per, at, z, shas, stages, peak, looks = max(1, batch // c), 0, [], [], {}, 0, 0
        for k in range(0, len(parts), per):
            t1 = perf_counter()
            pl = gf.merge(parts[k:k + per])
            stages["host_merge_s"] = stages.get("host_merge_s", 0) + perf_counter() - t1
            S = len(pl["patients"])
            out = F.evaluate(pl, time_[at:at + S], event[at:at + S], strata[at:at + S], keep_X=keep_X)
            at += S
            z.append(out["z"])
            looks += out["looks"]
            peak = max(peak, out["stages"].pop("device_bytes_in_use"))
            for name, v in out["stages"].items():
                stages[name] = stages.get(name, 0) + v
            if keep_X:
                shas += [db.sha(X) for X in out["X"]]
        wall = perf_counter() - t
        return {"wall_s": wall, "stages": {"host_plan_s": plan_s, **stages}, "z": np.concatenate(z), "x_sha": shas,
                "studies_per_launch": c * per, "launches": -(-len(parts) // per), "peak_device_bytes": peak, "looks": looks}

    t = perf_counter()
    gpu(min(batches))  # JIT and first-launch costs, excluded like the CPU arm's warm-up
    gpu_warm_s = perf_counter() - t
    ref = gpu(min(batches), keep_X=True)  # the parity pass
    zc, zg = np.stack([r["z"] for r in recs]), ref["z"]
    mc, mg = np.abs(zc).max(1), np.abs(zg).max(1)
    plant = setup["plant_feature"]
    summary = {"cpu_fused": db.summarize({"arm": "fused", "records": recs}, plant),
               "gpu_fused": db.summarize({"arm": "fused", "records": [{"stream": s, "i": i, "z": z} for (s, i), z in zip(jobs, zg)]}, plant)}
    cuts = {a: v["threshold"] for a, v in summary.items()}
    calls = {a: (m > cuts[a] if cuts[a] is not None else np.zeros(n, bool)) for a, m in (("cpu_fused", mc), ("gpu_fused", mg))}
    judged = np.array([s != "calibration" for s, _ in jobs])
    x_same = [a == r["x_sha"] for a, r in zip(ref["x_sha"], recs)]
    parity = {"x_bitwise": all(x_same), "x_bitwise_k": int(sum(x_same)), "x_compared": n,
              "x_sha_digest": hashlib.sha256("".join(ref["x_sha"]).encode()).hexdigest(),
              "z_max_abs_diff": float(np.abs(zg - zc).max()),
              "maxT_max_abs_diff": float(np.abs(mg - mc).max()), "maxT_within_1e-9": bool(np.abs(mg - mc).max() <= 1e-9),
              "thresholds": cuts, "decisions_agree_k": int((calls["cpu_fused"] == calls["gpu_fused"])[judged].sum()),
              "decisions_compared": int(judged.sum()),
              "rng": "the device steps NumPy's PCG64 stream itself: the same uniforms, so X is compared bit for bit"}
    parity["all"] = bool(parity["x_bitwise"] and parity["maxT_within_1e-9"]
                         and parity["decisions_agree_k"] == parity["decisions_compared"])
    timed = {}
    for b in batches:
        runs = [gpu(b) for _ in range(repeats)]
        same = all(np.array_equal(r["z"], zg) for r in runs)  # every timed pass reproduces the parity pass's z
        timed[str(b)] = {"wall_s": _spread([r["wall_s"] for r in runs]),
                         "wall_per_study_s": _spread([r["wall_s"] / n for r in runs]),
                         "studies_per_launch": runs[0]["studies_per_launch"], "launches": runs[0]["launches"],
                         "peak_device_bytes": max(r["peak_device_bytes"] for r in runs), "looks": runs[0]["looks"],
                         "z_identical_to_parity_pass": bool(same),
                         "stages_median_run": sorted(runs, key=lambda r: r["wall_s"])[len(runs) // 2]["stages"]}
    planners.shutdown()
    host["load_average_end"] = list(os.getloadavg())
    return {"host": host, "evaluated_studies": n, "setup": setup,
            "cpu_fused": {"warm_s": cpu_warm_s, "workers": int(CORES), "wall_s": _spread(cpu_walls),
                          "wall_per_study_s": _spread([w / n for w in cpu_walls]), "summary": summary["cpu_fused"]},
            "gpu_fused": {"static_s": static_s, "adjacency_upload_s": upload_s, "warm_s": gpu_warm_s,
                          "parity_pass_stages": ref["stages"], "by_batch": timed, "summary": summary["gpu_fused"]},
            "parity": parity, "maxT": {"cpu_fused": mc.tolist(), "gpu_fused": mg.tolist()}}


def _git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


@app.local_entrypoint()
def main(studies: int = 1000, batches: str = "1000,3000", repeats: int = 3, spent_usd: float = -1.0):
    from bench import design_batch as db

    if spent_usd < 0:
        raise SystemExit("--spent-usd: this task's Modal spend so far (0 for the first run)")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit first: the run refuses a dirty tracked tree, so source_commit is the code that ran")
    ceiling = TIMEOUT_S / 3600 * USD_PER_HOUR
    if spent_usd + ceiling > CAP_USD:
        raise SystemExit(f"refused: ${spent_usd:.2f} spent + ${ceiling:.2f} ceiling passes the ${CAP_USD:.0f} cap")
    commit, start = _git("rev-parse", "HEAD"), datetime.now(timezone.utc)
    sizes = sorted({min(int(b), 3 * studies) for b in batches.split(",")})
    out, error = None, None
    try:
        out = run.remote(studies, sizes, repeats)
    except Exception as e:  # noqa: BLE001  a failed run is recorded with its spend, not hidden
        error = f"{type(e).__name__}: {e}"
    wall = (datetime.now(timezone.utc) - start).total_seconds()
    spend = {"cap_usd": CAP_USD, "spent_before_usd": spent_usd, "ceiling_usd": round(ceiling, 2),
             "est_usd": round(wall / 3600 * USD_PER_HOUR, 3), "modal_wall_s": round(wall, 1), "rates_usd_per_hour": RATES,
             "basis": "ESTIMATE at list price: one container x local wall (an upper bound: includes image build and queueing)"}
    spend["task_total_usd"] = round(spent_usd + spend["est_usd"], 3)
    ok = out is not None and out["parity"]["all"] and studies == 1000
    result = {"status": "FAILED" if out is None or not out["parity"]["all"] else "MEASURED" if ok else "SMOKE",
              "source_commit": commit, "error": error,
              "labels": {"timings": "MEASURED", "outcomes": f"SYNTHETIC (atlas/simulation.py at {db.ATLAS_PIN[0][:7]})",
                         "rates": "ESTIMATE (FWER and power: binomial rates with Wilson 95% intervals)", "spend": "ESTIMATE"},
              "design": {**db.design(), "cohort": "charville", "alpha": db.ALPHA, "studies_per_stream": studies,
                         "streams": db.STREAMS, "root_seed": db.ROOT},
              "scope": ("wall = re-imaging and scoring every evaluated study (CPU: the pool's wall; GPU: host plan, upload, "
                        "kernels and z download); excludes data load, outcome prep, adjacency, warm-up and the threshold "
                        "summary, as bench/design_batch.py's compute_wall_s does. Both arms ran in one container."),
              "modal": {"app": "preflight-gpu-fused", "gpu_requested": GPU, "cpu": CORES, "memory_mib": MEMORY_MIB,
                        "started_utc": start.isoformat()},
              "spend": spend}
    if out is not None:
        g, c = out["gpu_fused"]["by_batch"], out["cpu_fused"]["wall_s"]["median"]
        best = min(g, key=lambda b: g[b]["wall_s"]["median"])
        result["headline"] = {"gpu_sku": out["host"]["gpu"], "best_studies_per_launch": g[best]["studies_per_launch"],
                              "gpu_wall_s_median": g[best]["wall_s"]["median"],
                              "gpu_wall_per_study_s_median": g[best]["wall_per_study_s"]["median"],
                              "cpu_fused_wall_s_median": c, "cpu_over_gpu": c / g[best]["wall_s"]["median"],
                              "peak_device_gib": g[best]["peak_device_bytes"] / 2**30}
        result.update(out)
    path = REPO / "bench" / "results" / f"gpu-fused-h100-{start:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["status"], json.dumps(result.get("headline")), json.dumps(result.get("parity")), json.dumps(spend), error)
    if out is not None:
        print(json.dumps(out["gpu_fused"]["by_batch"], indent=1))
    print(f"Saved {path}")
