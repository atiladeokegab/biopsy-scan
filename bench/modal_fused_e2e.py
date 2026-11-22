"""Run the paired Charville CPU benchmarks on Modal.

DATA_DIR=<public Charville file directory> MODAL_PROFILE=fhl-08 \
    uv run --with modal modal run bench/modal_fused_e2e.py --spent-usd <project total>
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import modal

REPO = Path(__file__).resolve().parents[1]
LOCAL_DATA = Path(os.environ.get("DATA_DIR", REPO / "data"))
DATA_FILES = ("charville_labels.csv", "charville_raw_data.zip")
RESULTS = REPO / "bench" / "results"
CORES, MEMORY_MIB, TIMEOUT_S = 8, 65536, 3600
REPEATS, KEEP_WARM_S = 3, 60
CAP_USD = 150.0
RATES = {"cpu_core_s": 0.0000131, "memory_gib_s": 0.00000222,
         "source": "https://modal.com/pricing", "checked_utc": "2026-11-22"}
USD_S = CORES * RATES["cpu_core_s"] + MEMORY_MIB / 1024 * RATES["memory_gib_s"]
SINGLE = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
          "NUMBA_NUM_THREADS": "1", "NUMBA_THREADING_LAYER": "workqueue"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def ignore_untracked(rel):
    if not hasattr(ignore_untracked, "tracked"):
        ignore_untracked.tracked = set(git("ls-files", "-z").split("\0"))
    return Path(rel).as_posix() not in ignore_untracked.tracked


image = (modal.Image.debian_slim(python_version="3.11").apt_install("libgomp1")
         .uv_sync(str(REPO)).env({"PYTHONPATH": "/root/preflight"})
         .workdir("/root/preflight").add_local_dir(REPO, "/root/preflight", ignore=ignore_untracked))
for name in DATA_FILES:
    image = image.add_local_file(LOCAL_DATA / name, f"/data/{name}", copy=True)
app = modal.App("preflight-fused-e2e")


def host():
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    model = next((line.partition(":")[2].strip() for line in (read("/proc/cpuinfo") or "").splitlines()
                  if line.startswith("model name")), None)
    affinity = sorted(os.sched_getaffinity(0))
    available = []
    physical = set()
    for cpu in affinity:
        siblings = read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        if not siblings:
            break
        if siblings not in physical:
            physical.add(siblings)
            available.append(cpu)
    topology_visible = len(available) >= CORES
    mask = (available if topology_visible else affinity)[:CORES]
    if len(mask) != CORES:
        raise RuntimeError(f"need {CORES} available CPUs, got {affinity}")
    return {"cpu_model": model, "cpu_count": os.cpu_count(), "available_affinity": affinity,
            "paired_cpu_mask": mask, "physical_topology_visible": topology_visible,
            "cgroup_cpu_max": read("/sys/fs/cgroup/cpu.max"), "task_id": os.environ.get("MODAL_TASK_ID"),
            "thread_environment": {k: os.environ.get(k) for k in SINGLE}, "load_average": os.getloadavg()}


def timed_arm(arm, workers, mask, plan, data_dir=Path("/data")):
    from bench import design_batch as db
    from core.cohorts import load

    before = os.sched_getaffinity(0)
    os.sched_setaffinity(0, set(mask))
    try:
        if os.sched_getaffinity(0) != set(mask):
            raise RuntimeError(f"failed to apply requested CPU affinity {mask}")
        db._S.clear()
        with TemporaryDirectory(prefix="preflight-raw-") as directory:
            raw = Path(directory)
            for name in DATA_FILES:
                (raw / name).symlink_to(data_dir / name)
            start = perf_counter()
            cells = load("charville", raw)
            load_s = perf_counter() - start
            out = db.run_arm(arm, cells, plan, workers=workers)
            scored = perf_counter()
            summary = db.summarize(out, out["setup"]["plant_feature"])
            elapsed = perf_counter() - start
        out["load_s"] = load_s
        return out, {"arm": arm, "workers": workers, "affinity": mask, "raw_load_s": load_s,
                     "setup": out["setup"], "warm_s": out["warm_s"],
                     "compute_through_scoring_s": out["studies_wall_s"],
                     "threshold_and_calls_s": elapsed - (scored - start),
                     "raw_to_answer_s": elapsed, "summary": summary, "stages": db.stage_times(out)}
    finally:
        db._S.clear()
        os.sched_setaffinity(0, before)


def scale_pair(plan, mask, order):
    from bench.matched_e2e import _coverage

    outputs, timings = {}, {}
    for workers in order:
        out, timing = timed_arm("lab", workers, mask[:workers], plan)
        _coverage(out["records"], plan, f"lab {workers} workers")
        outputs[workers] = {(r["stream"], r["i"]): (r["x_sha"], r["p"], r["maxT"]) for r in out["records"]}
        timings[str(workers)] = timing
    if outputs[1] != outputs[CORES]:
        raise ValueError("lab process-count change altered features, p-values or maxT")
    return {"order": order, "arms": timings, "identical_studies": len(outputs[1]),
            "scoring_ratio": timings["1"]["compute_through_scoring_s"] / timings["8"]["compute_through_scoring_s"],
            "raw_to_answer_ratio": timings["1"]["raw_to_answer_s"] / timings["8"]["raw_to_answer_s"]}


def fair_pair(plan, mask, order):
    from bench.matched_e2e import compare

    outputs, timings = {}, {}
    for arm in order:
        out, timings[arm] = timed_arm(arm, CORES, mask, plan["lab" if arm == "lab" else "preflight"])
        outputs[arm] = out
    checked = compare(outputs["lab"], outputs["fused"], plan, lab_workers=CORES)
    if checked["status"] != "MEASURED":
        raise ValueError("fused features differ from lab features")
    if checked["threshold"] != timings["fused"]["summary"]["threshold"]:
        raise ValueError("timed answer and parity check used different thresholds")
    decided = sum(plan["lab"].values())
    return {"order": order, "arms": timings, "parity": checked["parity"], "decisions": checked["decisions"],
            "scoring_ratio": checked["wall"]["ratio"],
            "raw_to_answer_ratio": timings["lab"]["raw_to_answer_s"] / timings["fused"]["raw_to_answer_s"],
            "usd_per_1000_decided_estimate": {arm: o["raw_to_answer_s"] * USD_S * 1000 / decided
                                             for arm, o in timings.items()}}


@app.function(image=image, cpu=(CORES, CORES), memory=(MEMORY_MIB, MEMORY_MIB), timeout=TIMEOUT_S,
              scaledown_window=KEEP_WARM_S, retries=0, env=SINGLE)
def measure(kind: str, repeat: int):
    from bench.matched_e2e import plans

    started = perf_counter()
    machine = host()
    mask = machine["paired_cpu_mask"]
    if kind == "fair":
        plan = plans(200, 1000, 200)
        order = ("lab", "fused") if repeat % 2 else ("fused", "lab")
        rows = [fair_pair(plan, mask, order)]
    elif kind == "lab-scaling":
        plan = {"null": 16, "alternative": 16}
        rows = [scale_pair(plan, mask, (1, CORES) if i % 2 else (CORES, 1))
                for i in range(1, REPEATS + 1)]
    else:
        raise ValueError(f"unknown benchmark {kind}")
    return {"kind": kind, "repeat": repeat, "plan": plan, "host": machine, "rows": rows,
            "function_wall_s": perf_counter() - started}


def spread(values):
    values = list(values)
    return {"n": len(values), "median": median(values), "min": min(values), "max": max(values)}


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@app.local_entrypoint()
def main(spent_usd: float):
    from bench.matched_e2e import check_cap

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes before starting the measured run")
    commit = git("rev-parse", "HEAD")
    if spent_usd < 0:
        raise SystemExit("project spend estimate must be nonnegative")
    jobs = [("lab-scaling", 1)] + [("fair", i) for i in range(1, REPEATS + 1)]
    # Allow $1.25 per invocation for its one-hour timeout, retention and startup.
    reservation = 1.25 * len(jobs)
    check_cap(spent_usd, reservation, CAP_USD)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    spend_path = RESULTS / f"modal-spend-fused-e2e-{run_id}.json"
    result_path = RESULTS / f"fused-e2e-{run_id}.json"
    spend = {"status": "ESTIMATE", "source_commit": commit, "prior_project_usd": spent_usd,
             "project_cap_usd": CAP_USD, "reserved_usd": reservation, "rates": RATES,
             "basis": "requested CPU and memory at list rates, using function wall plus 60 s retention per success; "
                      "failed or unresolved jobs retain their $1.25 reservation. Startup, credits and invoices differ.",
             "entries": []}
    result = {"status": "PENDING", "source_commit": commit,
              "labels": {"timings": "MEASURED", "outcomes": "SIMULATED", "rates_and_costs": "ESTIMATE"},
              "modal": {"physical_cores_requested_and_capped": CORES, "memory_mib_requested_and_capped": MEMORY_MIB,
                        "gpu": None, "timeout_s": TIMEOUT_S, "profile": os.environ.get("MODAL_PROFILE")},
              "scope": "Raw public Charville CSV ZIP and labels already staged in the container. Each arm builds its own "
                       "cell cache and runs simulation setup, adjacency and JIT as applicable, scoring, threshold and calls. "
                       "Excludes upload, container startup, reporting and deployment. OS file cache uncontrolled. "
                       "Arms run sequentially in a paired container and use single-thread workers. Scaling repeats share "
                       "one container. Fair repeats use separate calls. Costs normalize each arm to 1,000 decided studies "
                       "at this batch size, including fused calibration, excluding startup and retention.",
              "data_source": "https://zenodo.org/records/13179600", "spend_receipt": spend_path.name, "runs": []}
    save(spend_path, spend)
    save(result_path, result)
    running = []
    for kind, repeat in jobs:
        entry = {"kind": kind, "repeat": repeat, "est_usd": 1.25, "state": "submitting"}
        spend["entries"].append(entry)
        save(spend_path, spend)
        call = measure.spawn(kind, repeat)
        entry.update(call_id=call.object_id, state="submitted")
        save(spend_path, spend)
        running.append((call, entry))
        print(f"Submitted {kind} {repeat}: {call.object_id}", flush=True)
    for call, entry in running:
        try:
            row = call.get()
        except Exception as exc:
            entry.update(state="failed", error_type=type(exc).__name__)
            print(f"{entry['kind']} repeat {entry['repeat']} failed: {type(exc).__name__}: {exc}", flush=True)
        else:
            entry.update(state="completed", function_wall_s=row["function_wall_s"],
                         est_usd=(row["function_wall_s"] + KEEP_WARM_S) * USD_S)
            result["runs"].append(row)
            save(result_path, result)
            print(f"Completed {entry['kind']} repeat {entry['repeat']}", flush=True)
        save(spend_path, spend)
    spend["running_total_usd"] = sum(e["est_usd"] for e in spend["entries"])
    save(spend_path, spend)
    if len(result["runs"]) != len(jobs):
        result["status"] = "FAILED"
        save(result_path, result)
        raise SystemExit(f"incomplete benchmark: see {result_path} and {spend_path}")
    fair = [run["rows"][0] for run in result["runs"] if run["kind"] == "fair"]
    scale = next(run["rows"] for run in result["runs"] if run["kind"] == "lab-scaling")
    result["spread"] = {kind: {metric: spread(row[metric] for row in rows)
                               for metric in ("scoring_ratio", "raw_to_answer_ratio")}
                        for kind, rows in (("fair", fair), ("lab_scaling", scale))}
    result["spread"]["fair"].update({
        "raw_to_answer_s": {arm: spread(row["arms"][arm]["raw_to_answer_s"] for row in fair)
                            for arm in ("lab", "fused")},
        "usd_per_1000_decided_estimate": {arm: spread(row["usd_per_1000_decided_estimate"][arm] for row in fair)
                                           for arm in ("lab", "fused")}})
    result["status"] = "MEASURED"
    save(result_path, result)
    print(json.dumps(result["spread"], indent=2), flush=True)
