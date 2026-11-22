"""The 583x ratio on one host: the fused path and the off-the-shelf baseline back to back in each container.

    MODAL_PROFILE=fhl-08 DATA_DIR=<dir> uv run --with modal modal run bench/modal_same_host.py --studies 24

Per container, in an order that alternates per repeat: the fused arm on --full-studies per stream (all three streams,
its normal size, #120), the fused arm again on the matched set, and squidpy + lifelines on the matched set
(--studies per stream, null and alternative, seeds as #113): its pipeline alone in one pool, then our reference
recount and agreement in a second, so the headline ratio (pipeline only / fused full) has no checks in it. 8
single-thread workers everywhere. --repeats fresh containers (single_use_containers, distinct task ids checked) of the
class bench/modal_external.py uses, in its image (the fused arm needs only numpy, scipy, pandas and numba, all pinned
there). Commit first; a ceiling over --cap-usd is refused. Writes bench/results/same-host-<utc>.json and logs the
spend in bench/results/modal-spend.json.
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


def _fused(plan: dict, workers: int) -> dict:
    from bench import design_batch as db
    from core.cohorts import load
    db._S.pop("kernel", None)  # cached per process; rebuilt on this run's prepared CSR
    out = db.run_arm("fused", load("charville"), plan, workers=workers)
    st = db.stage_times(out)
    return {"s_per_study": st["compute_wall_s"] / st["evaluated_studies"], "stages": st, "warm_s": out["warm_s"],
            "workers": out["workers"], "plan": plan,
            "x_sha": {f"{r['stream']}/{r['i']}": r["x_sha"] for r in out["records"]}}


def run_both(plan: dict, full: dict, order: list[str], workers: int) -> dict:
    """In this process, in order: "fused" = the fused arm on its full plan, then on the matched plan; "external" =
    squidpy + lifelines on the matched plan, the pipeline in one pool and our checks in a second. A fresh cohort
    load per run (prepare mutates it)."""
    from bench import external_baseline as eb
    from core.cohorts import load

    arms = {}
    for arm in order:
        started = _now()
        if arm == "fused":
            arms["fused_full"] = _fused(full, workers) | {"started_utc": started, "finished_utc": _now()}
            arms["fused_full"].pop("x_sha")
            started = _now()
            arms["fused_matched"] = _fused(plan, workers) | {"started_utc": started, "finished_utc": _now()}
        else:
            out = eb.run(load("charville"), plan, workers=workers, split=True)
            n = len(out["records"])
            arms["external"] = {
                "pipeline_s_per_study": out["pipeline_wall_s"] / n, "with_checks_s_per_study": out["pool_wall_s"] / n,
                "pipeline_wall_s": out["pipeline_wall_s"], "checks_wall_s": out["checks_wall_s"], "studies": n,
                "warm_s": out["warm_s"], "prep_s": out["prep_s"], "workers": out["workers"], "started_utc": started,
                "worker_study_s": eb.spread([r["study_s"] for r in out["records"]]),
                "worker_checks_s": eb.spread([r["checks_s"] for r in out["records"]]),
                "x_sha": {f"{r['stream']}/{r['i']}": r["x_sha_ours"] for r in out["records"]}, "finished_utc": _now()}
    return arms


@app.function(image=image, gpu=GPU, cpu=CORES, memory=MEMORY_MIB, timeout=6 * 3600, env=SINGLE, single_use_containers=True)
def both(plan: dict, full: dict, order: list[str], commit: str) -> dict:
    import logging
    import sys
    import warnings
    from importlib.metadata import version

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)  # squidpy logs one INFO line per graph
    sys.path.insert(0, APP_DIR)
    host = _host() | {"versions": {p: version(p) for p in ("squidpy", "lifelines", "pandas", "numpy", "numba")}}
    started = _now()
    arms = run_both(plan, full, order, int(CORES))
    host["load_average_end"] = list(os.getloadavg())
    return {"source_commit": commit, "host": host, "order": order, "arms": arms, "started_utc": started,
            "finished_utc": _now()}


@app.local_entrypoint()
def main(studies: int = 24, full_studies: int = 1000, repeats: int = 3, cap_usd: float = 7.0, timeout_s: int = 2400):
    from bench import design_batch as db
    from bench import external_baseline as eb

    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes first: source_commit must be the code that runs")
    commit = _git("rev-parse", "HEAD")
    ceiling = round(repeats * timeout_s / 3600 * USD_PER_HOUR_CEILING, 2)
    if ceiling > cap_usd:
        raise SystemExit(f"ceiling ${ceiling} > cap ${cap_usd}: ask the lead first")
    plan = {"null": studies, "alternative": studies}
    full = {s: full_studies for s in db.STREAMS}  # the fused arm's normal size, as #120 (0.0419 s over 3,000)
    start = datetime.now(timezone.utc)
    entry = {"utc": start.strftime("%Y-%m-%dT%H:%MZ"), "run": f"same-host fused {full_studies}/stream + matched {studies}/stream, external {studies}/stream x {repeats}",
             "containers": repeats, "gpu": GPU, "cores": CORES, "memory_gib": MEMORY_MIB / 1024, "cap_usd": cap_usd,
             "est_usd": ceiling, "basis": f"{repeats} x {timeout_s} s ceiling x ${USD_PER_HOUR_CEILING:.6f}/h (A100 80 GB rate)",
             "source_commit": commit, "container_class": CLASS}
    _log_spend(entry)
    calls = [both.with_options(timeout=timeout_s).spawn(plan, full, list(ARMS[::1 if k % 2 == 0 else -1]), commit)
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
    same_cells = all([r["arms"]["fused_matched"].pop("x_sha") == r["arms"]["external"].pop("x_sha") for r in runs])
    clean = not [f for f in _git("diff", "--name-only", commit).splitlines() if f != "bench/results/modal-spend.json"]
    ok = not errors and len(runs) == repeats and len(set(tasks)) == repeats and same_cells and clean
    cols = [{"external_pipeline_s_per_study": r["arms"]["external"]["pipeline_s_per_study"],
             "external_with_checks_s_per_study": r["arms"]["external"]["with_checks_s_per_study"],
             "fused_full_s_per_study": r["arms"]["fused_full"]["s_per_study"],
             "fused_matched_s_per_study": r["arms"]["fused_matched"]["s_per_study"]} for r in runs]
    summary = eb.same_host_summary(cols) if runs else None
    m = lambda k: summary[k] if k.endswith("_s_per_study") else summary["ratios"][k]  # noqa: E731
    rng = lambda k, f: f"{m(k)['median']:{f}} (range {m(k)['min']:{f}}-{m(k)['max']:{f}})"  # noqa: E731
    result = {
        "status": "MEASURED" if ok else "FAILED", "errors": errors, "source_commit": commit, "clean_tree": clean,
        "labels": {"timing": "MEASURED", "outcomes": "SYNTHETIC (atlas/simulation.py, as #113)",
                   "ratio": "MEASURED, same host: every column from each container, ratios per repeat"},
        "headline_column": "ratios.external_pipeline_vs_fused_full",
        "headline": summary and (
            f"Same host, {len(runs)} fresh containers, {int(CORES)} workers per arm: off the shelf (squidpy + lifelines, "
            f"pipeline only) {rng('external_pipeline_s_per_study', '.2f')} s/study over {2 * studies} studies; fused "
            f"{rng('fused_full_s_per_study', '.4f')} s per evaluated study over {3 * full_studies}; "
            f"ratio {rng('external_pipeline_vs_fused_full', '.0f')}x. "
            f"With our reference checks in the external time: {rng('external_with_checks_vs_fused_full', '.0f')}x. "
            f"On the identical {2 * studies} studies: fused {rng('fused_matched_s_per_study', '.4f')} s/study, "
            f"ratio {rng('external_pipeline_vs_fused_matched', '.0f')}x. {eb.COMPARISON}"),
        "summary": summary,
        "checks": {"distinct_containers": len(set(tasks)) == len(tasks), "same_cells_both_arms": same_cells,
                   "same_cells_basis": "per matched study, the fused arm's X sha256 equals the external checks' "
                                       "recount of the cells the pipeline gave squidpy"},
        "basis": {
            "external_pipeline": "pipeline pool wall / studies: cells, squidpy, lifelines and the Bonferroni call only",
            "external_with_checks": "pipeline pool wall + checks pool wall (our recount, score and agreement on the "
                                    "same cells, a second pool) / studies; the external-baseline runs' 24.443 basis "
                                    "ran both in one pool",
            "fused_full": f"compute_wall_s / {3 * full_studies} evaluated studies (calibration, null, alternative), "
                          "as #120's 0.0419",
            "fused_matched": f"compute_wall_s / the same {2 * studies} studies the external arm ran; its pool start-up "
                             "weighs more on so few studies",
            "excluded": "container start, data load, outcome prep, adjacency and warm-up in every column"},
        "design": {**db.design(), "cohort": "charville", "studies_per_stream": studies,
                   "full_studies_per_stream": full_studies, "streams": db.STREAMS, "root_seed": db.ROOT, "n_perms": eb.N_PERMS},
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
