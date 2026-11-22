"""The external baseline (bench/external_baseline.py) on Modal, in the design point's container class.

    MODAL_PROFILE=fhl-08 DATA_DIR=<dir> uv run --with modal modal run bench/modal_external.py --studies 25

--studies is per stream (null and alternative; no calibration: Bonferroni needs none), seeds as #113.
One container of the class bench/modal_design.py uses (A100 + 8 cores + 64 GiB), 8 single-thread workers.
squidpy, lifelines and pandas<3 live only in this image: lifelines 0.30 needs pandas<3, which the project
env does not take (#59). Commit first (source_commit is the code that runs); a ceiling over --cap-usd is
refused. Writes bench/results/external-baseline-<utc>.json and logs the spend in bench/results/modal-spend.json.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import modal

from bench.modal_design import (CLASS, CORES, DATA_DIR, GPU, LOCAL_DATA, MEMORY_MIB, REPO, SINGLE, USD_PER_HOUR,
                                _git, _log_spend, _tracked)

APP_DIR = "/root/preflight"
PINS = ("numpy==2.4.6", "scipy==1.17.1", "pandas==2.3.3", "numba==0.68.1", "statsmodels==0.15.0",
        "pyarrow==26.0.1", "matplotlib==3.11.2", "anndata==0.12.19", "squidpy==1.8.2", "lifelines==0.30.3")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*PINS)
    .env({"DATA_DIR": DATA_DIR, "PYTHONPATH": APP_DIR})
    .add_local_file(LOCAL_DATA / "charville_labels.csv", f"{DATA_DIR}/charville_labels.csv", copy=True)
    .add_local_file(LOCAL_DATA / "charville_cells.npz", f"{DATA_DIR}/charville_cells.npz", copy=True)
    .workdir(APP_DIR)
    .add_local_dir(REPO, APP_DIR, ignore=lambda rel: not _tracked(rel))
)
app = modal.App("preflight-external-baseline")


@app.function(image=image, gpu=GPU, cpu=CORES, memory=MEMORY_MIB, timeout=6 * 3600, env=SINGLE)
def baseline(plan: dict, n_jobs: int = 1) -> dict:
    import logging
    import sys
    import time
    import warnings
    from importlib.metadata import version

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)  # squidpy logs one INFO line per graph
    sys.path.insert(0, APP_DIR)
    from bench import external_baseline as eb
    from core.cohorts import load

    host = {"cpu_count": os.cpu_count(), "affinity_size": len(os.sched_getaffinity(0)),
            "load_average_start": list(os.getloadavg()), "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "thread_environment": {k: os.environ.get(k) for k in SINGLE},
            "versions": {p: version(p) for p in ("squidpy", "lifelines", "pandas", "anndata", "numpy", "numba")}}
    t = time.perf_counter()
    cells = load("charville")
    load_s = time.perf_counter() - t
    out = eb.run(cells, plan, workers=int(CORES) // n_jobs, n_jobs=n_jobs)
    host["load_average_end"] = list(os.getloadavg())
    out["summary"] = eb.summarize(out)
    out["host"], out["load_s"] = host, load_s
    return out


def _finite(o):
    if hasattr(o, "item") and not isinstance(o, (dict, list)):  # numpy scalars
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    return o


@app.local_entrypoint()
def main(studies: int = 25, cap_usd: float = 5.0, timeout_s: int = 3600, n_jobs: int = 1):
    from bench import design_batch as db
    from bench import external_baseline as eb

    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes first: source_commit must be the code that runs")
    commit = _git("rev-parse", "HEAD")
    ceiling = round(timeout_s / 3600 * USD_PER_HOUR, 2)
    if ceiling > cap_usd:
        raise SystemExit(f"ceiling ${ceiling} > cap ${cap_usd}: ask the lead first")
    plan = {"null": studies, "alternative": studies}
    start = datetime.now(timezone.utc)
    entry = {"utc": start.strftime("%Y-%m-%dT%H:%MZ"),
             "run": f"build-9 external-baseline {studies}/stream, squidpy n_jobs={n_jobs}",
             "containers": 1, "gpu": GPU, "cores": CORES, "memory_gib": MEMORY_MIB / 1024, "cap_usd": cap_usd,
             "est_usd": ceiling, "basis": f"{timeout_s} s ceiling x ${USD_PER_HOUR:.6f}/h", "source_commit": commit,
             "container_class": CLASS}
    _log_spend(entry)
    error, out = None, None
    try:
        out = baseline.with_options(timeout=timeout_s).remote(plan, n_jobs)
    except Exception as e:  # noqa: BLE001 - a failed run is recorded as FAILED, with its spend
        error = repr(e)[-2000:]
    end = datetime.now(timezone.utc)
    wall = (end - start).total_seconds()
    s = out["summary"] if out else None
    result = {
        "status": "MEASURED" if out and s["studies"] == 2 * studies else "FAILED", "error": error,
        "source_commit": commit,
        "labels": {"timing": "MEASURED", "outcomes": f"SYNTHETIC (atlas/simulation.py, as #113)",
                   "agreement": "MEASURED (same re-imaged cells in both pipelines)",
                   "rates": "ESTIMATE (binomial rates with Wilson 95% intervals)"},
        "headline": s and eb.headline(s),
        "method": eb.__doc__.strip(),
        "design": {**db.design(), "cohort": "charville", "alpha": eb.ALPHA, "n_perms": eb.N_PERMS,
                   "studies_per_stream": studies, "streams": {k: db.STREAMS[k] for k in plan}, "root_seed": db.ROOT,
                   "correction": "Bonferroni over 630"},
        "summary": s, "records": out and out["records"],
        "setup": out and {k: out[k] for k in ("prep_s", "warm_s", "load_s", "host", "expected_event_fraction")},
        "modal": {"app": "preflight-external-baseline", "class": CLASS, "started_utc": start.isoformat(),
                  "finished_utc": end.isoformat(), "wall_s": wall,
                  "note": "wall_s includes container start and image pull; pool_wall_s is inside the container"},
    }
    path = REPO / "bench" / "results" / f"external-baseline-{start:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(_finite(result), indent=2, allow_nan=False) + "\n")
    entry["actual"] = {"modal_wall_s": round(wall, 1), "est_usd": round(wall / 3600 * USD_PER_HOUR, 3),
                       "basis": "container x local wall (an upper bound: includes queueing)",
                       "result": str(path.relative_to(REPO))}
    entry["counted"] = "actual (container x wall)"
    _log_spend(entry)
    print(result["status"], result["headline"], json.dumps(_finite(s), default=str))
    print(f"Saved {path}")
