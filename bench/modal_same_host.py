"""The 583x ratio on one host: the fused path and the off-the-shelf baseline back to back in each container.

    MODAL_PROFILE=fhl-08 DATA_DIR=<dir> uv run --with modal modal run bench/modal_same_host.py --studies 24

--studies is per stream (null and alternative, seeds as #113): both arms run exactly these studies, each over 8
single-thread pool workers, timed as before: fused = compute_wall_s / studies (bench/design_batch.py), external =
pool wall / studies (bench/external_baseline.py). --repeats fresh containers (single_use_containers, distinct task ids
checked) of the class bench/modal_external.py uses, in its image (the fused arm needs only numpy, scipy, pandas and
numba, all pinned there); arm order alternates per repeat. Commit first; a ceiling over --cap-usd is refused.
Writes bench/results/same-host-<utc>.json and logs the spend in bench/results/modal-spend.json.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import modal

from bench.modal_design import (CLASS, CORES, GPU, MEMORY_MIB, RATES, REPO, SINGLE, _git, _log_spend)
from bench.modal_external import _finite, image

APP_DIR = "/root/preflight"
# gpu="A100" has landed on both A100 sizes (spend ledger: 40 GB; #120's fused host: 80 GB): ceiling at the 80 GB rate
USD_PER_HOUR_CEILING = 2.4984 + CORES * RATES["cpu_core"] + MEMORY_MIB / 1024 * RATES["memory_gib"]
ARMS = ("fused", "external")
app = modal.App("preflight-same-host")


def _host() -> dict:
    def read(path):
        try:
            return open(path).read()
        except OSError:
            return ""
    cpu = next((ln.split(":", 1)[1].strip() for ln in read("/proc/cpuinfo").splitlines() if ln.startswith("model name")), None)
    mem = next((int(ln.split()[1]) * 1024 for ln in read("/proc/meminfo").splitlines() if ln.startswith("MemTotal")), None)
    return {"cpu_model": cpu, "cpu_count": os.cpu_count(), "affinity_size": len(os.sched_getaffinity(0)),
            "cgroup_cpu_max": read("/sys/fs/cgroup/cpu.max").strip() or None,
            "cgroup_memory_max": read("/sys/fs/cgroup/memory.max").strip() or None, "host_mem_total_bytes": mem,
            "modal_task_id": os.environ.get("MODAL_TASK_ID"), "modal_region": os.environ.get("MODAL_REGION"),
            "load_average_start": list(os.getloadavg()), "thread_environment": {k: os.environ.get(k) for k in SINGLE}}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_both(plan: dict, order: list[str], workers: int) -> dict:
    """Both arms on the same plan in this process, in order; a fresh cohort load per arm (prepare mutates it)."""
    from bench import design_batch as db
    from bench import external_baseline as eb
    from core.cohorts import load

    arms = {}
    for arm in order:
        db._S.pop("kernel", None)  # the fused kernel is cached per process; the external arm builds none
        started = _now()
        if arm == "fused":
            out = db.run_arm("fused", load("charville"), plan, workers=workers)
            st = db.stage_times(out)
            arms[arm] = {"s_per_study": st["compute_wall_s"] / st["evaluated_studies"], "stages": st,
                         "warm_s": out["warm_s"], "workers": out["workers"],
                         "x_sha": {f"{r['stream']}/{r['i']}": r["x_sha"] for r in out["records"]}}
        else:
            out = eb.run(load("charville"), plan, workers=workers)
            n = len(out["records"])
            arms[arm] = {"s_per_study": out["pool_wall_s"] / n, "pool_wall_s": out["pool_wall_s"], "studies": n,
                         "warm_s": out["warm_s"], "prep_s": out["prep_s"], "workers": out["workers"],
                         "worker_study_s": eb.spread([r["study_s"] for r in out["records"]]),
                         "x_sha": {f"{r['stream']}/{r['i']}": r["x_sha_ours"] for r in out["records"]}}
        arms[arm] |= {"started_utc": started, "finished_utc": _now()}
    return arms


@app.function(image=image, gpu=GPU, cpu=CORES, memory=MEMORY_MIB, timeout=6 * 3600, env=SINGLE, single_use_containers=True)
def both(plan: dict, order: list[str], commit: str) -> dict:
    import logging
    import sys
    import warnings
    from importlib.metadata import version

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)  # squidpy logs one INFO line per graph
    sys.path.insert(0, APP_DIR)
    host = _host() | {"versions": {p: version(p) for p in ("squidpy", "lifelines", "pandas", "numpy", "numba")}}
    started = _now()
    arms = run_both(plan, order, int(CORES))
    host["load_average_end"] = list(os.getloadavg())
    return {"source_commit": commit, "host": host, "order": order, "arms": arms, "started_utc": started,
            "finished_utc": _now()}


@app.local_entrypoint()
def main(studies: int = 24, repeats: int = 3, cap_usd: float = 7.0, timeout_s: int = 2400):
    from bench import design_batch as db
    from bench import external_baseline as eb

    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes first: source_commit must be the code that runs")
    commit = _git("rev-parse", "HEAD")
    ceiling = round(repeats * timeout_s / 3600 * USD_PER_HOUR_CEILING, 2)
    if ceiling > cap_usd:
        raise SystemExit(f"ceiling ${ceiling} > cap ${cap_usd}: ask the lead first")
    plan = {"null": studies, "alternative": studies}
    start = datetime.now(timezone.utc)
    entry = {"utc": start.strftime("%Y-%m-%dT%H:%MZ"), "run": f"same-host fused + external {studies}/stream x {repeats}",
             "containers": repeats, "gpu": GPU, "cores": CORES, "memory_gib": MEMORY_MIB / 1024, "cap_usd": cap_usd,
             "est_usd": ceiling, "basis": f"{repeats} x {timeout_s} s ceiling x ${USD_PER_HOUR_CEILING:.6f}/h (A100 80 GB rate)",
             "source_commit": commit, "container_class": CLASS}
    _log_spend(entry)
    calls = [both.with_options(timeout=timeout_s).spawn(plan, list(ARMS[::1 if k % 2 == 0 else -1]), commit)
             for k in range(repeats)]
    runs, errors = [], []
    for k, call in enumerate(calls):
        try:
            runs.append(call.get())
        except Exception as e:  # noqa: BLE001 - a failed repeat makes the run FAILED, with its spend
            errors.append({"repeat": k, "error": repr(e)[-2000:]})
    end = datetime.now(timezone.utc)
    wall = (end - start).total_seconds()
    tasks = [r["host"]["modal_task_id"] for r in runs]
    same_cells = all(r["arms"]["fused"].pop("x_sha") == r["arms"]["external"].pop("x_sha") for r in runs)
    clean = not [f for f in _git("diff", "--name-only", commit).splitlines() if f != "bench/results/modal-spend.json"]
    ok = not errors and len(runs) == repeats and len(set(tasks)) == repeats and same_cells and clean
    summary = eb.same_host_summary([{"external_s_per_study": r["arms"]["external"]["s_per_study"],
                                     "fused_s_per_study": r["arms"]["fused"]["s_per_study"]} for r in runs]) if runs else None
    result = {
        "status": "MEASURED" if ok else "FAILED", "errors": errors, "source_commit": commit, "clean_tree": clean,
        "labels": {"timing": "MEASURED", "outcomes": "SYNTHETIC (atlas/simulation.py, as #113)",
                   "ratio": "MEASURED, same host: both arms in each container, on the same studies"},
        "headline": summary and (
            f"Same host, {2 * studies} studies, {int(CORES)} workers per arm, {len(runs)} fresh containers: off the shelf "
            f"(squidpy + lifelines) {summary['external_s_per_study']['median']:.2f} s/study "
            f"(range {summary['external_s_per_study']['min']:.2f}-{summary['external_s_per_study']['max']:.2f}), fused "
            f"{summary['fused_s_per_study']['median']:.4f} s/study (range {summary['fused_s_per_study']['min']:.4f}-"
            f"{summary['fused_s_per_study']['max']:.4f}); ratio median {summary['ratio']['median']:.0f}x "
            f"(range {summary['ratio']['min']:.0f}-{summary['ratio']['max']:.0f}x). {eb.COMPARISON}"),
        "summary": summary,
        "checks": {"distinct_containers": len(set(tasks)) == len(tasks), "same_cells_both_arms": same_cells,
                   "same_cells_basis": "per study, the fused arm's X sha256 equals the external arm's own recount of "
                                       "the cells it gave squidpy"},
        "basis": {"fused": "compute_wall_s / studies (pool wall of re-imaging and scoring, as #120's 0.0419)",
                  "external": "pool wall / studies (as external-baseline's 24.443)",
                  "excluded": "container start, data load, outcome prep, adjacency and warm-up in both arms",
                  "small_set": f"{2 * studies} studies: the fused pool's fork and start-up weigh more than at #120's "
                               "3,000 studies, so its s/study is an upper bound against 0.0419"},
        "design": {**db.design(), "cohort": "charville", "studies_per_stream": studies,
                   "streams": {k: db.STREAMS[k] for k in plan}, "root_seed": db.ROOT, "n_perms": eb.N_PERMS},
        "repeats": runs,
        "modal": {"app": "preflight-same-host", "class": CLASS, "started_utc": start.isoformat(),
                  "finished_utc": end.isoformat(), "wall_s": wall,
                  "note": "repeats run concurrently, one per fresh container; wall_s includes start and image pull"},
    }
    path = REPO / "bench" / "results" / f"same-host-{start:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(_finite(result), indent=2, allow_nan=False) + "\n")
    entry["actual"] = {"modal_wall_s": round(wall, 1), "est_usd": round(repeats * wall / 3600 * USD_PER_HOUR_CEILING, 3),
                       "basis": "each container x local wall at the 80 GB rate (an upper bound: includes queueing)",
                       "result": str(path.relative_to(REPO))}
    _log_spend(entry)
    print(result["status"], result["headline"])
    print(f"Saved {path}")
