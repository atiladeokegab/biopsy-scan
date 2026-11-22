"""Measured CPU and GPU utilisation of three arms on one fixed study set, for the demo's replay (demo/usage/).

    DATA_DIR=<dir> MODAL_PROFILE=fhl-08 uv run --with modal modal run bench/modal_usage.py --studies 4

Arms, each in its own container of the class bench/modal_design.py uses (A100 + 8 cores + 64 GiB), in parallel:
  current  squidpy + lifelines (bench/external_baseline.run, 8 single-thread workers), the industry baseline
  fused    Preflight's fused CPU kernel (bench/design_batch.run_arm "fused", 8 workers)
  gpu      Preflight's A100 path (bench/design_batch.run_arm "gpu": GPU re-imaging, batched GPU scoring)
--studies is per stream (null and alternative), the same seeds for every arm. A sampler thread reads /proc/stat
(what psutil.cpu_percent(percpu=True) reads) and streams `nvidia-smi -lms 100` (utilisation.gpu, memory.used)
every 100 ms; each study's completion time is stamped in its worker. Commit first; a ceiling over --cap-usd is
refused. Writes bench/results/usage-traces-<utc>.json and demo/usage/traces.json, and logs the spend.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone

from bench.modal_design import CLASS, CORES, GPU, MEMORY_MIB, REPO, SINGLE, USD_PER_HOUR, _git, _log_spend
from bench.modal_design import image as design_image
from bench.modal_external import image as external_image

import modal

DT = 0.1
ARMS = ("current", "fused", "gpu")
app = modal.App("preflight-usage-traces")
_ONE = {}


def _proc_stat() -> list[tuple[int, int]]:
    """(busy, total) jiffies per core from /proc/stat; busy excludes idle and iowait, as psutil does."""
    rows = []
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                v = [int(x) for x in line.split()[1:]]
                rows.append((sum(v[:8]) - v[3] - v[4], sum(v[:8])))
    return rows


class Sampler:
    """Per-core CPU % and GPU util % / memory every DT seconds, on time.monotonic() from t0."""

    def __init__(self):
        self.t0, self.cpu, self.gpu, self.stop = time.monotonic(), [], [], threading.Event()
        self.gpu_error = None

    def t(self) -> float:
        return round(time.monotonic() - self.t0, 3)

    def _cpu(self):
        prev = _proc_stat()
        while not self.stop.wait(DT):
            cur = _proc_stat()
            self.cpu.append([self.t(), [round(100 * (cb - pb) / (ct - pt), 1) if ct > pt else 0.0
                                        for (pb, pt), (cb, ct) in zip(prev, cur)]])
            prev = cur

    def _gpu(self):
        try:
            p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                                  f"-lms={int(DT * 1000)}"], stdout=subprocess.PIPE, text=True)
        except OSError as e:
            self.gpu_error = repr(e)
            return
        self.proc = p
        for line in p.stdout:
            try:
                util, mem = (float(x) for x in line.split(","))
            except ValueError:
                continue
            self.gpu.append([self.t(), util, mem])

    def __enter__(self):
        self.threads = [threading.Thread(target=f, daemon=True) for f in (self._cpu, self._gpu)]
        for th in self.threads:
            th.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if getattr(self, "proc", None):
            self.proc.terminate()
        for th in self.threads:
            th.join(timeout=5)


def _timed(job):
    """The arm's own one_study, plus its completion time (CLOCK_MONOTONIC is shared by forked workers)."""
    r = _ONE["fn"](job)
    r["done_mono"] = time.monotonic()
    return r


def _host() -> dict:
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except OSError:
        smi = None
    return {"cpu_count": os.cpu_count(), "affinity_size": len(os.sched_getaffinity(0)), "cores_reserved": CORES,
            "modal_task_id": os.environ.get("MODAL_TASK_ID"), "gpu": smi,
            "thread_environment": {k: os.environ.get(k) for k in SINGLE}}


def _measure(arm: str, plan: dict) -> dict:
    import sys
    import warnings
    sys.path.insert(0, "/root/preflight")
    warnings.filterwarnings("ignore")
    from core.cohorts import load
    if arm == "current":
        import logging
        logging.disable(logging.WARNING)
        from bench import external_baseline as mod
    else:
        from bench import design_batch as mod
    _ONE["fn"], mod.one_study = mod.one_study, _timed  # run()/run_arm() look one_study up at call time
    cells = load("charville")
    with Sampler() as s:
        out = mod.run(cells, plan, workers=int(CORES)) if arm == "current" else mod.run_arm(arm, cells, plan, workers=int(CORES))
        end = s.t()
    done = sorted(round(r["done_mono"] - s.t0, 3) for r in out["records"])
    if arm == "current":
        studies_wall, score = out["pool_wall_s"], 0.0
    else:
        studies_wall, score = out["studies_wall_s"], out["batched_score_wall_s"] or 0.0
    # DERIVED from the arm's own perf_counter walls: the timed compute window (prep and warm-up before it)
    compute_end = end
    compute_start = round(end - score - studies_wall, 3)
    return {"cpu": s.cpu, "gpu": s.gpu, "gpu_error": s.gpu_error, "done_s": done, "sampler_end_s": end,
            "compute_window_s": [compute_start, compute_end], "studies_wall_s": studies_wall,
            "batched_score_wall_s": score or None, "studies": len(out["records"]), "workers": out["workers"],
            "host": _host()}


@app.function(image=external_image, gpu=GPU, cpu=CORES, memory=MEMORY_MIB, timeout=3600, env=SINGLE)
def current(plan: dict) -> dict:
    return _measure("current", plan)


@app.function(image=design_image, gpu=GPU, cpu=CORES, memory=MEMORY_MIB, timeout=3600, env=SINGLE)
def preflight(arm: str, plan: dict) -> dict:
    return _measure(arm, plan)


def mean(xs) -> float | None:
    return round(sum(xs) / len(xs), 2) if xs else None


def summarise(a: dict) -> dict:
    lo, hi = a["compute_window_s"]
    cpu = [sum(c) / len(c) for t, c in a["cpu"] if lo <= t <= hi]
    gpu = [u for t, u, _ in a["gpu"] if lo <= t <= hi]
    return {"wall_s": round(hi - lo, 3), "studies": a["studies"], "mean_cpu_pct_all_cores": mean(cpu),
            "mean_cpu_cores_busy": mean([c * a["host"]["cpu_count"] / 100 for c in cpu]),
            "mean_gpu_util_pct": mean(gpu), "cpu_samples_in_window": len(cpu), "gpu_samples_in_window": len(gpu)}


def replay(arms: dict, source: str) -> dict:
    """The compute window of each arm, times from its start: what demo/usage/ plays back. No smoothing."""
    out = {"schema": "preflight-usage/1", "source": source, "label": "MEASURED", "sample_interval_s": DT, "arms": {}}
    for k, a in arms.items():
        lo, hi = a["compute_window_s"]
        pad = 0.5  # half a second either side, so the idle-to-busy edge shows
        win = lambda t: lo - pad <= t <= hi + pad  # noqa: E731
        out["arms"][k] = {"wall_s": round(hi - lo, 3), "studies": a["studies"], "cores": a["host"]["cpu_count"],
                          "cores_reserved": CORES, "gpu_name": (a["host"]["gpu"] or "").split(",")[0] or None,
                          "done_s": [round(t - lo, 3) for t in a["done_s"]],
                          "cpu": [[round(t - lo, 2), [round(x) for x in c]] for t, c in a["cpu"] if win(t)],
                          "gpu": [[round(t - lo, 2), u, m] for t, u, m in a["gpu"] if win(t)],
                          "summary": summarise(a)}
    return out


@app.local_entrypoint()
def main(studies: int = 4, cap_usd: float = 4.0, timeout_s: int = 1200):
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes first: source_commit must be the code that runs")
    commit = _git("rev-parse", "HEAD")
    ceiling = round(len(ARMS) * timeout_s / 3600 * USD_PER_HOUR, 2)
    if ceiling > cap_usd:
        raise SystemExit(f"ceiling ${ceiling} > cap ${cap_usd}: ask the lead first")
    plan = {"null": studies, "alternative": studies}
    start = datetime.now(timezone.utc)
    entry = {"utc": start.strftime("%Y-%m-%dT%H:%MZ"), "run": f"usage-traces {studies}/stream, arms {list(ARMS)}",
             "containers": len(ARMS), "gpu": GPU, "cores": CORES, "memory_gib": MEMORY_MIB / 1024, "cap_usd": cap_usd,
             "est_usd": ceiling, "basis": f"{timeout_s} s ceiling x {len(ARMS)} x ${USD_PER_HOUR:.6f}/h",
             "source_commit": commit, "container_class": CLASS}
    _log_spend(entry)
    calls = {"current": current.with_options(timeout=timeout_s).spawn(plan),
             **{a: preflight.with_options(timeout=timeout_s).spawn(a, plan) for a in ARMS[1:]}}
    arms, errors = {}, {}
    for a, c in calls.items():
        try:
            arms[a] = c.get()
        except Exception as e:  # noqa: BLE001 - a failed arm is recorded, never faked
            errors[a] = repr(e)[-2000:]
    end = datetime.now(timezone.utc)
    wall = (end - start).total_seconds()
    usd = round(wall / 3600 * USD_PER_HOUR * len(ARMS), 3)
    stamp = f"{start:%Y%m%dT%H%M%SZ}"
    path = REPO / "bench" / "results" / f"usage-traces-{stamp}.json"
    result = {
        "status": "MEASURED" if not errors and all(a["studies"] == 2 * studies for a in arms.values()) else "FAILED",
        "errors": errors, "source_commit": commit,
        "labels": {"traces": "MEASURED", "compute_window": "DERIVED (sampler end minus the arm's own perf_counter "
                   "studies and batched-score walls)", "outcomes": "SYNTHETIC (atlas/simulation.py, as #113)",
                   "gpu_note": "squidpy and lifelines never touch the GPU; the A100 is only the container class",
                   "cpu_note": "per-core % from /proc/stat of the container (gVisor shows every visible core; "
                               "8 are reserved, the rest can be lent); nvidia-smi utilization.gpu is NVML's "
                               "busy-kernel share over its own sample period"},
        "design": {"studies_per_stream": studies, "streams": list(plan), "same_seeds_every_arm": True,
                   "arms": {"current": "bench/external_baseline.run, 8 workers",
                            "fused": "bench/design_batch.run_arm('fused'), 8 workers",
                            "gpu": "bench/design_batch.run_arm('gpu'), one process on the A100"}},
        "sampling": {"interval_s": DT, "cpu": "/proc/stat per core (psutil's source)",
                     "gpu": "nvidia-smi --query-gpu=utilization.gpu,memory.used -lms=100"},
        "summary": {a: summarise(v) for a, v in arms.items()},
        "arms": arms,
        "cost": {"usd": usd, "basis": f"{len(ARMS)} containers x local wall {wall:.0f} s x ${USD_PER_HOUR:.4f}/h "
                 "(upper bound: includes image pull and queueing)", "cap_usd": cap_usd},
        "modal": {"app": "preflight-usage-traces", "class": CLASS, "started_utc": start.isoformat(),
                  "finished_utc": end.isoformat(), "wall_s": wall},
    }
    path.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    if result["status"] == "MEASURED":
        demo = REPO / "demo" / "usage" / "traces.json"
        demo.write_text(json.dumps(replay(arms, str(path.relative_to(REPO))), separators=(",", ":")) + "\n")
    entry["actual"] = {"modal_wall_s": round(wall, 1), "est_usd": usd, "basis": result["cost"]["basis"],
                       "result": str(path.relative_to(REPO))}
    _log_spend(entry)
    print(result["status"], json.dumps(result["summary"]), errors or "")
    print(f"Saved {path}")
