"""Measure the archived three-stream Charville design on Modal H100s.

    uv run modal run bench/modal_h100_design.py::run_h100 --prior-project-usd ...

One GPU per worker. Compute windows include re-imaging and GPU scoring. Local
wall includes worker preparation, transfer and the single global summary.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import modal
import numpy as np

from bench import design_batch as db
from gpu.modal_app import app, gpu_function, gpu_info, overlap

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench/results"
ARCHIVE = RESULTS / "modal-design-20261121T205536Z.json"
COUNTS = (1, 2, 4, 8)
PLAN = {stream: 1000 for stream in db.STREAMS}
MODEL = "NVIDIA H100 80GB HBM3"
TIMEOUTS = {1: 2900, 2: 1600, 4: 1000, 8: 700}
USD_PER_SECOND = 0.001097 + 8 * 0.0000131 + 32 * 0.00000222
THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"), "1")


def expected_ids(plan: dict) -> list[tuple[str, int]]:
    return [(stream, i) for stream, count in plan.items() for i in range(count)]


def archived_reference(receipt: dict) -> dict:
    design = receipt["design"]
    if (receipt["status"] != "MEASURED" or design["root_seed"] != db.ROOT
            or design["streams"] != db.STREAMS or design["studies_per_stream"] != 1000
            or any(design.get(key) != value for key, value in db.design().items())
            or design["hr_per_source_sd"] != db.HR or design["alpha"] != db.ALPHA):
        raise ValueError("archived receipt does not match the current design and seed streams")
    rows = receipt["arms"]["fused"]["studies"]
    keyed = {(r["stream"], r["i"]): r for r in rows}
    if len(rows) != len(keyed) or set(keyed) != set(expected_ids(PLAN)):
        raise ValueError("archived CPU fused receipt lacks exact study coverage")
    return keyed


def validate_records(records: list[dict], reference: dict) -> dict:
    keyed = {(r["stream"], r["i"]): r for r in records}
    if len(records) != len(keyed) or keyed.keys() != reference.keys():
        raise ValueError("GPU result has missing, duplicated or unexpected studies")
    deltas = [abs(r["maxT"] - reference[k]["maxT"]) for k, r in keyed.items()]
    if any(not math.isfinite(x) or x > 1e-9 for x in deltas):
        raise ValueError("GPU maxT differs from the archive beyond absolute 1e-9")
    if any(r["x_sha"] != reference[k]["x_sha"] for k, r in keyed.items()):
        raise ValueError("GPU feature hash differs from the CPU fused archive")
    return {"studies_compared": len(keyed), "x_bitwise": True,
            "max_abs_maxT_delta": max(deltas), "absolute_tolerance": 1e-9}


def _device() -> dict:
    info = gpu_info()
    info["uuid"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10
    ).strip()
    if info["count"] != 1 or info["model"] != MODEL or not info["uuid"]:
        raise ValueError(f"expected a single {MODEL}, got {info['all']}")
    return info


def _rendezvous(ready, uuid: str, count: int) -> float:
    if count == 1:
        return 0.0
    begun = time.perf_counter()
    ready[uuid] = True
    while ready.len() < count:
        if time.perf_counter() - begun > 100:
            raise TimeoutError(f"only {ready.len()} of {count} H100s reached the start gate")
        time.sleep(.5)
    return time.perf_counter() - begun


@app.function(**(gpu_function | {"gpu": "H100!", "timeout": TIMEOUTS[1]}),
              single_use_containers=True, max_containers=16, retries=0, env=THREADS)
def evaluate_shard(jobs: list[tuple[str, int]], ready, count: int) -> dict:
    from core.cohorts import load
    from stats.screen import screen_studies

    entered = time.perf_counter()
    device = _device()
    t = time.perf_counter()
    cells = load("charville")
    load_s = time.perf_counter() - t
    prep = db.prepare("gpu", cells, PLAN)
    warm = db.warm("gpu")
    wait = _rendezvous(ready, device["uuid"], count)
    start = time.time()
    ref_ns = time.perf_counter()
    records, reimage_s, scoring_s = [], 0.0, 0.0
    for i in range(0, len(jobs), 128):
        block = jobs[i:i + 128]
        t = time.perf_counter()
        results = [db.one_study(j) for j in block]
        reimage_s += time.perf_counter() - t
        t = time.perf_counter()
        samples = [db._S["studies"][stream][index] for stream, index in block]
        Z = screen_studies(np.stack([r.pop("X") for r in results]),
                           *(np.stack([s[k] for s in samples]) for k in ("time", "event", "strata")),
                           backend="gpu", block=128)["z"]
        scoring_s += time.perf_counter() - t
        for result, z in zip(results, Z, strict=True):
            result.update(z=z, maxT=float(np.abs(z).max()))
        records.extend(results)
    compute = time.perf_counter() - ref_ns
    return {"records": records, "gpu": device, "compute_window": [start, time.time()],
            "reimage_wall_s": reimage_s, "scoring_wall_s": scoring_s, "compute_wall_s": compute,
            "setup_s": {"load": load_s, "outcome_prep": prep["prep_s"], "adjacency_upload": prep["adjacency_s"],
                        "warm_up": warm, "rendezvous": wait}, "plant_feature": prep["plant_feature"],
            "in_function_s": time.perf_counter() - entered}


def measure(n: int, reference: dict) -> tuple[dict, float]:
    starts = time.perf_counter()
    jobs = expected_ids(PLAN)
    with modal.Dict.ephemeral() as state:
        parts = list(evaluate_shard.with_options(timeout=TIMEOUTS[n]).map(
            [jobs[i::n] for i in range(n)], kwargs={"ready": state, "count": n}, order_outputs=True))
    transferred = time.perf_counter()
    hardware = [p["gpu"] for p in parts]
    if len(parts) != n or len({h["uuid"] for h in hardware}) != n or any(h["model"] != MODEL for h in hardware):
        raise ValueError(f"expected {n} distinct {MODEL} devices")
    plants = {p["plant_feature"] for p in parts}
    if len(plants) != 1:
        raise ValueError("workers disagree about the planted feature")
    all_records = [r for p in parts for r in p["records"]]
    windows = [p | {"seeds": [db.STREAMS[r["stream"]] * 1000 + r["i"] for r in p["records"]]} for p in parts]
    timing = overlap(windows, [db.STREAMS[s] * 1000 + i for s, i in jobs])
    if not all(timing[k] for k in ("all_overlapped", "seed_coverage_exact", "distinct_task_ids")):
        raise ValueError(f"fleet timing/coverage checks failed: {timing}")
    t = time.perf_counter()
    answer = db.summarize({"arm": "gpu", "records": all_records}, plants.pop())
    summary_s = time.perf_counter() - t
    wall = time.perf_counter() - starts  # feature and maxT parity runs after the timed workflow
    parity = validate_records(all_records, reference)
    return {"status": "MEASURED", **timing, "gpus": n, "evaluated_studies": len(jobs),
            "local_threshold_and_calls_s": summary_s, "modal_call_wall_s": transferred - starts,
            "end_to_end_wall_s": wall, "wall_s_per_evaluated_study": wall / len(jobs),
            "studies_per_hour_compute_span": len(jobs) * 3600 / timing["compute_span_s"],
            "summary": answer, "parity": parity,
            "workers": [{k: v for k, v in p.items() if k != "records"} for p in parts]}, (
            sum(p["in_function_s"] for p in parts) * USD_PER_SECOND)


def save(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def spread(samples: list[float]) -> dict:
    return {"n": len(samples), "median": median(samples), "min": min(samples), "max": max(samples)}


@app.local_entrypoint()
def run_h100(prior_project_usd: float, cap_usd: float = 25.0, repeats: int = 2):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes before measuring")
    if (not math.isfinite(prior_project_usd) or not math.isfinite(cap_usd) or not 0 <= prior_project_usd <= 150
            or not 0 < cap_usd <= 25 or prior_project_usd + cap_usd > 150 or repeats not in (1, 2)):
        raise SystemExit("invalid project or task reservation")
    prior = json.loads(ARCHIVE.read_text())
    reference = archived_reference(prior)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS / f"modal-h100-design-scaling-{stamp}.json"
    spend = RESULTS / f"modal-spend-h100-design-scaling-{stamp}.json"
    source = git("rev-parse", "HEAD")
    report = {"status": "INCOMPLETE", "source_commit": source, "reference": str(ARCHIVE.relative_to(ROOT)),
              "reference_commit": prior["source_commit"], "synthetic_outcome_plan": PLAN, "seed_root": db.ROOT,
              "labels": {"timings": "MEASURED", "outcomes": "SIMULATED", "rates_and_costs": "ESTIMATE"},
              "hardware_request": {"gpu": "H100!", "cpus": 8, "memory_gib": 32},
              "scope": "Each H100 worker loads staged Charville inputs, prepares draws and warms up. Compute spans "
                       "re-imaging and GPU scoring across shard windows (host clock skew unmeasured). End-to-end wall "
                       "includes Modal call, setup, transfer and one driver threshold/calls summary. Parity checks separate.",
              "repeats": [], "failures": []}
    ledger = {"status": "ESTIMATE", "source_commit": source, "prior_project_usd": prior_project_usd,
              "task_cap_usd": cap_usd, "usd_per_worker_second": USD_PER_SECOND, "entries": [],
              "basis": "Modal list rates. Successful points use function runtime. Failed points retain reservations."}
    used = 0.0
    for repeat in range(1, repeats + 1):
        points = {}
        report["repeats"].append(points)
        reserved = 0.0
        with ThreadPoolExecutor(max_workers=len(COUNTS)) as executor:
            active = {}
            for n in COUNTS:
                ceiling = n * TIMEOUTS[n] * USD_PER_SECOND
                if used + reserved + ceiling > cap_usd:
                    report["failures"].append({"repeat": repeat, "gpus": n, "reason": "task allocation"})
                    continue
                entry = {"repeat": repeat, "gpus": n, "reserved_usd": ceiling, "status": "SUBMITTED"}
                reserved += ceiling
                ledger["entries"].append(entry)
                save(spend, ledger)
                active[executor.submit(copy_context().run, measure, n, reference)] = entry
                print(f"repeat {repeat} {n} H100s reserved ${ceiling:.2f}", flush=True)
            for future in as_completed(active):
                entry = active[future]
                n = entry["gpus"]
                try:
                    points[str(n)], est = future.result()
                    entry.update(status="MEASURED", estimated_usd=est)
                    used += est
                    print(f"repeat {repeat} {n} H100s wall {points[str(n)]['end_to_end_wall_s']:.2f} s", flush=True)
                except Exception as exc:
                    entry.update(status="UNMEASURED", error=str(exc))
                    used += entry["reserved_usd"]
                    report["failures"].append({"repeat": repeat, "gpus": n, "reason": str(exc)})
                ledger.update(task_estimated_and_retained_usd=used, running_total_usd=prior_project_usd + used)
                save(spend, ledger)
                save(path, report)
    report["status"] = "PARTIAL" if report["failures"] else "MEASURED"
    report["spread"] = {}
    for n in COUNTS:
        paired = [(r["1"], r[str(n)]) for r in report["repeats"] if "1" in r and str(n) in r]
        if paired:
            report["spread"][str(n)] = {"end_to_end_wall_s": spread([p["end_to_end_wall_s"] for _, p in paired]),
                 "studies_per_hour_compute_span": spread([p["studies_per_hour_compute_span"] for _, p in paired]),
                 "compute_efficiency": spread([b["compute_span_s"] / (n * p["compute_span_s"]) for b, p in paired])}
    save(path, report)
    print(path, flush=True)
