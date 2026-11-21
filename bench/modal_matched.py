"""The matched lab vs Preflight comparison (bench/matched_e2e.py) in one 8-core, 64 GiB Modal CPU container.

    DATA_DIR=<dir> MODAL_PROFILE=<profile> uv run --with modal modal run bench/modal_matched.py

Preflight runs --calibration, then --per-stream null and alternative studies on 8 workers; the lab then runs the
first --lab-per-stream null and alternative studies of the same streams on one pinned core. Same container,
so the same host class and the same studies, seeds and design for both. Commit first: the run refuses a dirty
tracked tree, so source_commit is the code that ran. A run whose ceiling (timeout x rate) would take the spend
log over --cap-usd (default 8) is refused before it starts. Logs the spend in
bench/results/modal-spend-matched.json before and after; writes bench/results/matched-e2e-<utc>.json.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
APP_DIR, DATA_DIR = "/root/preflight", "/data"
LOCAL_DATA = Path(os.environ.get("DATA_DIR", REPO / "data"))
CORES, MEMORY_MIB, TIMEOUT_S = 8.0, 65536, 3600
RATES = {"cpu_core": 0.04716, "memory_gib": 0.007992,
         "source": "bench/results/modal-spend-numba.json (modal.com/pricing read 2026-11-21T17:36Z)"}
USD_PER_HOUR = CORES * RATES["cpu_core"] + MEMORY_MIB / 1024 * RATES["memory_gib"]
SPEND = REPO / "bench" / "results" / "modal-spend-matched.json"
# workqueue: both arms fork workers after the parent imported Numba, and GNU OpenMP aborts a fork (as build-9's runner)
SINGLE = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
          "NUMBA_THREADING_LAYER": "workqueue"}


def _tracked(rel: Path) -> bool:
    if not getattr(_tracked, "files", None):  # lazy: the container re-imports this module and has no .git
        out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True).stdout
        _tracked.files = set(filter(None, out.split("\0")))
    return rel.as_posix() in _tracked.files


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .uv_sync(str(REPO))
    .env({"DATA_DIR": DATA_DIR, "PYTHONPATH": APP_DIR})
    .add_local_file(LOCAL_DATA / "charville_labels.csv", f"{DATA_DIR}/charville_labels.csv", copy=True)
    .add_local_file(LOCAL_DATA / "charville_cells.npz", f"{DATA_DIR}/charville_cells.npz", copy=True)
    .workdir(APP_DIR)
    .add_local_dir(REPO, APP_DIR, ignore=lambda rel: not _tracked(rel))
)
app = modal.App("preflight-matched-e2e")


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


@app.function(image=image, cpu=CORES, memory=MEMORY_MIB, timeout=TIMEOUT_S, env=SINGLE)
def matched(plan: dict) -> dict:
    import sys

    sys.path.insert(0, APP_DIR)
    from bench import b9_design_batch as db
    from bench import matched_e2e as m
    from core.cohorts import load

    host = {"cpu_count": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0)),
            "cgroup_cpu_max": _read("/sys/fs/cgroup/cpu.max"), "load_average_start": list(os.getloadavg()),
            "cpu_model": next((line.split(":", 1)[1].strip() for line in (_read("/proc/cpuinfo") or "").splitlines()
                               if line.startswith("model name")), None),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"), "thread_environment": {k: os.environ.get(k) for k in SINGLE}}
    out = m.run(lambda: load("charville"), plan, workers=int(CORES))
    host["load_average_end"] = list(os.getloadavg())
    res = m.compare(out["lab"], out["preflight"])
    plant = out["preflight"]["setup"]["plant_feature"]
    res["arms"] = {a: {"workers": o["workers"], "affinity": o.get("affinity"), "setup": o["setup"], "warm_s": o["warm_s"],
                       "stages": db.stage_times(o), "summary": db.summarize(o, plant)} for a, o in out.items()}
    res["host"] = host
    return json.loads(json.dumps(res, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def _git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _log_spend(entry: dict) -> float:
    log = json.loads(SPEND.read_text()) if SPEND.exists() else {
        "workspace_profile": os.environ.get("MODAL_PROFILE"), "rates_usd_per_hour": RATES, "entries": []}
    log["entries"] = [e for e in log["entries"] if e["utc"] != entry["utc"]] + [entry]
    log["running_total_usd"] = round(sum(e.get("actual", {}).get("est_usd", e["est_usd"]) for e in log["entries"]), 3)
    SPEND.write_text(json.dumps(log, indent=2) + "\n")
    return log["running_total_usd"]


@app.local_entrypoint()
def main(lab_per_stream: int = 50, calibration: int = 1000, per_stream: int = 1000, cap_usd: float = 8.0):
    from bench import matched_e2e as m

    _git("ls-files", "--error-unmatch", "bench/modal_matched.py")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes first: source_commit must be the code that runs")
    plan = m.plans(lab_per_stream, calibration, per_stream)
    spent = json.loads(SPEND.read_text())["running_total_usd"] if SPEND.exists() else 0.0
    ceiling = round(TIMEOUT_S / 3600 * USD_PER_HOUR, 3)
    if spent + ceiling > cap_usd:
        raise SystemExit(f"refused: spent {spent} + ceiling {ceiling} > cap {cap_usd} USD")
    commit, start = _git("rev-parse", "HEAD"), datetime.now(timezone.utc)
    utc = f"{start:%Y-%m-%dT%H:%MZ}"
    entry = {"utc": utc, "run": "matched lab vs Preflight end to end, 8 cores, 64 GiB, no GPU", "plan": plan,
             "containers": 1, "cap_usd": cap_usd, "est_hours": TIMEOUT_S / 3600, "est_usd": ceiling,
             "basis": f"8 cores + 64 GiB at the {TIMEOUT_S} s function-timeout ceiling", "source_commit": commit}
    _log_spend(entry)
    try:
        res = matched.remote(plan)
    finally:
        end = datetime.now(timezone.utc)
        wall = (end - start).total_seconds()
        entry["actual"] = {"modal_wall_s": round(wall), "est_usd": round(wall / 3600 * USD_PER_HOUR, 3)}
        _log_spend(entry)  # a failed run is ledgered too
    path = REPO / "bench" / "results" / f"matched-e2e-{start:%Y%m%dT%H%M%SZ}.json"
    entry["actual"]["result"] = str(path.relative_to(REPO))
    total = _log_spend(entry)
    w = res["wall"]
    doc = {
        "status": res.pop("status"), "source_commit": commit, "harness": m.B9,
        "why": "bench/results/cpu-ladder.json: lab_today (7.52 s/study) is features + 1,000-permutation inference, "
               "its ladder rows are feature-only, so it marks matched_full_pipeline_speedup UNMEASURED. "
               "Here both arms do features and the per-study decision on the same studies.",
        "scope": {"lab": "cKDTree re-imaging + stats.permutation_test (1,000 within-strata shuffles), call p <= 0.05; "
                         "one study at a time, one pool worker pinned to one core; fixed sample of the first "
                         f"{lab_per_stream} null and {lab_per_stream} alternative studies",
                  "preflight": "Numba CSR re-imaging + Cox score screen, call maxT > design threshold from the "
                               f"{calibration} calibration nulls; 8 single-thread workers; calibration counted",
                  "both": "same container, design bench.real_cpu.DESIGN, same seeds (SeedSequence([20261121, stream, i])), "
                          "outcomes atlas.simulation; walls exclude outcome simulation, adjacency and JIT warm-up "
                          "(wall.with_setup includes them); the lab ran a sample, so compare per-study wall only"},
        "plan": plan,
        "headline": (f"Per study, end to end: lab {w['lab_s_per_study']:.3f} s on 1 core "
                     f"({w['studies']['lab']} studies) vs Preflight {w['preflight_s_per_decided_study']:.4f} s on 8 cores "
                     f"({w['studies']['preflight_decided']} decided studies, {w['studies']['preflight_calibration']} "
                     f"calibration nulls paid for): {w['ratio']:.1f}x. Features bitwise on "
                     f"{res['parity']['x_compared']} shared studies: {res['parity']['x_bitwise']}; decisions agree "
                     f"{res['decisions']['agree']}/{res['decisions']['compared']}."),
        **res, "projection": m.project(w),
        "modal": {"app": "preflight-matched-e2e", "cpu": CORES, "memory_mib": MEMORY_MIB, "gpu": None,
                  "started_utc": start.isoformat(), "finished_utc": end.isoformat(), "wall_s": wall,
                  "est_usd": entry["actual"]["est_usd"], "spend_log_total_usd": total},
    }
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(doc["headline"])
    print(f"Saved {path}; spend log total {total} USD")
