"""Strong-scaling curve of build-9's Numba arm on Modal CPU containers (helpers and gates: bench/scaling.py).

    DATA_DIR=<dir with charville_cells.npz, charville_labels.csv> MODAL_PROFILE=fhl-08 \
        uv run modal run bench/modal_scaling.py --studies 1000 --cores 1,2,4,8,16,32,64

One container per core count, all launched at once, each cpu=(n, n) (a hard limit, no burst), workers = n,
the same 1,000 null studies (bench/design_batch.py seeds). The run refuses a dirty tracked tree and a
single-attempt estimate (each container once, to its timeout, at list price; not a billing ceiling) over the $5 cap, logs the estimate in bench/results/modal-spend.json
before launch and the actual after, and writes a new bench/results/modal-scaling-<utc>.json: MEASURED only
when every count covers exactly the planned studies with X identical to the 8-core point and z within 1e-9
relative (bitwise z reported, not required), else FAILED with no timings. Limits (scaling.CAVEATS) go into the file:
one concurrent cross-host sweep, hidden CPU model, RAM growing with cores, possible preemption restarts.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

from bench import scaling

REPO = Path(__file__).resolve().parents[1]
APP_DIR, DATA_DIR = "/root/preflight", "/data"
LOCAL_DATA = Path(os.environ.get("DATA_DIR", REPO / "data"))
SPEND = REPO / "bench" / "results" / "modal-spend.json"
BUILD9 = "bench/results/modal-design-20261121T185537Z.json"  # build-9's 8-core numba arm, merged with #113
SINGLE = scaling.SINGLE
BASE_IMAGE = "modal.Image.debian_slim(python_version='3.11') + uv_sync of this repo"


def _tracked(rel: Path) -> bool:
    if not getattr(_tracked, "files", None):  # lazy: the container re-imports this module and has no .git
        out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True).stdout
        _tracked.files = set(filter(None, out.split("\0")))
    return rel.as_posix() in _tracked.files


image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(str(REPO))
    .env({"DATA_DIR": DATA_DIR, "PYTHONPATH": APP_DIR})
    .add_local_file(LOCAL_DATA / "charville_labels.csv", f"{DATA_DIR}/charville_labels.csv", copy=True)
    .add_local_file(LOCAL_DATA / "charville_cells.npz", f"{DATA_DIR}/charville_cells.npz", copy=True)
    .workdir(APP_DIR)
    .add_local_dir(REPO, APP_DIR, ignore=lambda rel: not _tracked(rel))
)
app = modal.App("preflight-modal-scaling")


def _host() -> dict:
    def read(path):
        try:
            return Path(path).read_text()
        except OSError:
            return ""
    model = next((ln.split(":", 1)[1].strip() for ln in read("/proc/cpuinfo").splitlines()
                  if ln.startswith("model name")), None)
    return {"cpu_model": model, "cpu_count": os.cpu_count(), "affinity_size": len(os.sched_getaffinity(0)),
            "cgroup_cpu_max": read("/sys/fs/cgroup/cpu.max").strip() or None,
            "load_average_start": list(os.getloadavg()), "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "thread_environment": {k: os.environ.get(k) for k in SINGLE}}


def _point(cores: int, studies: int) -> dict:
    import time

    import numpy as np

    from bench import design_batch as db
    from core.cohorts import load

    host = _host()
    t = time.perf_counter()
    cells = load("charville")
    load_s = time.perf_counter() - t
    out = db.run_arm("numba", cells, {"null": studies}, workers=cores)
    host["load_average_end"] = list(os.getloadavg())
    recs = [{"i": r["i"], "x_sha": r["x_sha"], "z_sha": scaling.z_sha(r["z"]), "maxT": float(np.abs(r["z"]).max()),
             "reimage_s": r["reimage_s"], "score_s": r["score_s"], "z": np.asarray(r["z"], np.float64)}
            for r in out["records"]]  # z comes home for the tolerance check, then only its sha256 is written
    return {"cores": cores, "workers": out["workers"], "studies_wall_s": out["studies_wall_s"], "warm_s": out["warm_s"],
            "load_s": load_s, "prep_s": out["setup"]["prep_s"], "adjacency_s": out["setup"]["adjacency_s"],
            "reimage_worker_wall_s": sum(r["reimage_s"] for r in recs),
            "scoring_worker_wall_s": sum(r["score_s"] for r in recs),
            "host": host, "records": recs}


@app.function(image=image, cpu=1.0, memory=16384, timeout=3600, env=SINGLE)  # Modal resets image-level OMP_* to the core count
def point(cores: int, studies: int) -> dict:
    return _point(cores, studies)


def _git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _ledger(update: dict | None = None, new: dict | None = None) -> float:
    """Append `new` (running total carried on) or merge `update` into this task's last entry; returns USD spent
    by earlier modal-scaling runs (their actual when known, else their estimate)."""
    log = json.loads(SPEND.read_text())
    mine = [e for e in log["entries"] if e["run"].startswith("modal-scaling")]
    spent = sum(e.get("actual", {}).get("est_usd", e["est_usd"]) for e in mine)
    if new:
        new["running_total_usd"] = round(log["entries"][-1].get("running_total_usd", 0) + new["est_usd"], 2)
        log["entries"].append(new)
    if update:
        mine[-1]["actual"] = update
    SPEND.write_text(json.dumps(log, indent=2) + "\n")
    return spent


def class_name(cores: int) -> str:
    return (f"Modal CPU container, cpu=({cores}, {cores}) physical cores as request and hard limit, "
            f"{scaling.memory_gib(cores)} GiB, no GPU, {BASE_IMAGE}")


@app.local_entrypoint()
def main(studies: int = 1000, cores: str = ",".join(map(str, scaling.CORES))):
    counts = scaling.parse_cores(cores)
    if studies < 1:
        raise SystemExit("--studies must be at least 1")
    _git("ls-files", "--error-unmatch", "bench/modal_scaling.py", "bench/scaling.py", "bench/design_batch.py")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes first: source_commit must be the code that runs")
    commit = _git("rev-parse", "HEAD")
    spent = _ledger()
    ceiling = scaling.check_cap(counts, studies, spent)
    start = datetime.now(timezone.utc)
    _ledger(new={"utc": start.strftime("%Y-%m-%dT%H:%MZ"), "run": f"modal-scaling {studies} null studies, cores {list(counts)}",
                 "containers": len(counts), "est_usd": ceiling, "source_commit": commit,
                 "basis": "single-attempt estimate (not a billing ceiling): each container's timeout x (cores x $0.0000131 + GiB x $0.00000222) per s; "
                          f"task cap ${scaling.CAP_USD}, ${round(spent, 2)} spent before"})
    calls = {c: point.with_options(cpu=(float(c), float(c)), memory=scaling.memory_gib(c) * 1024,
                                   timeout=scaling.timeout_s(c, studies)).spawn(c, studies) for c in counts}
    points, errors = {}, {}
    for c, call in calls.items():
        try:
            points[c] = call.get()
        except Exception as e:  # noqa: BLE001 - a failed container makes the run FAILED, not a crash with no record
            errors[c] = repr(e)[-2000:]
    end = datetime.now(timezone.utc)
    try:
        prior = json.loads(_git("show", f"{commit}:{BUILD9}"))
    except subprocess.CalledProcessError:
        prior = None
    clean = not [f for f in _git("diff", "--name-only", commit).splitlines() if f != str(SPEND.relative_to(REPO))]
    res = scaling.result(points, counts, errors if clean else {**errors, "tree": "changed during the run"}, prior,
                         studies)
    from bench import design_batch as db
    out = {
        "status": res["status"], "failures": res["failures"], "source_commit": commit,
        "labels": {"compute": "MEASURED", "sweep": "one sweep, every core count launched at once",
                   "outcomes": "SYNTHETIC (atlas/simulation.py)",
                   "claims": "times and ratios only: speedup_vs_1_core = studies wall at 1 core / at n cores"},
        "workload": {"arm": "numba (bench/design_batch.py at source_commit)", "stream": "null",
                     "studies": studies, "root_seed": db.ROOT, "stream_seed": db.STREAMS["null"], "design": db.design(),
                     "timed": "studies_wall_s: the pool's re-imaging and Cox scoring of every study; excludes data "
                              "load, study preparation, adjacency and JIT warm-up (each recorded per point)"},
        "container_class": {c: class_name(c) for c in counts},
        "parity": res["parity"], "caveats": res["caveats"], "fields": res["fields"],
        "curve": res.get("curve"),
        "points": {c: {k: v for k, v in p.items() if k != "records"} | {"container_class": class_name(c)}
                   for c, p in points.items()} if res["status"] == "MEASURED" else
                  {c: {"host": p["host"]} for c, p in points.items()},
        "studies_by_cores": scaling.study_rows(points),
        "errors": {c: e for c, e in errors.items()},
        "modal": {"app": "preflight-modal-scaling", "started_utc": start.isoformat(), "finished_utc": end.isoformat(),
                  "wall_s": (end - start).total_seconds(), "build9_reference": BUILD9 if prior else None},
    }
    path = REPO / "bench" / "results" / f"modal-scaling-{start:%Y%m%dT%H%M%SZ}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; results are never overwritten")
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    wall = (end - start).total_seconds()
    _ledger(update={"modal_wall_s": round(wall, 1), "result": str(path.relative_to(REPO)),
                    "est_usd": round(sum(wall * scaling.usd_per_s(c) for c in counts), 3),
                    "basis": "single-attempt estimate at list price: each container x local wall (includes image build "
                             "and queueing); invoice not verified"})
    print(res["status"], json.dumps(res["failures"]), json.dumps(res.get("curve")), json.dumps(res["parity"], default=str))
    print(f"Saved {path}")
