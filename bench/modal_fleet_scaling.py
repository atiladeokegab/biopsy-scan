"""Strong and weak scaling of the same Charville re-imaging workload on Modal A100s.

Run from a committed checkout: uv run modal run bench/modal_fleet_scaling.py
Each point uses fresh one-GPU workers. Strong and weak batches share setup. The
16-GPU point has the same seeds and shard sizes in both curves, so it runs once.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import modal

from gpu.modal_app import _charville, app, gpu_function, gpu_info, overlap

FLEET = (1, 2, 4, 8, 16)
STRONG, WEAK_PER_GPU = 256, 16
GPU = "A100-80GB"
MODEL = "NVIDIA A100-SXM4-80GB"
RATE = 2.4984 + 8 * 0.04716 + 32 * 0.007992  # Modal list rates, USD/h, 2026-11-22
ROOT = Path(__file__).resolve().parents[1]


def workload(n: int, seed: int) -> dict[str, list[int]]:
    batches = {"strong": list(range(seed, seed + STRONG))}
    if n * WEAK_PER_GPU != STRONG:
        batches["weak"] = list(range(seed, seed + n * WEAK_PER_GPU))
    return batches


def _timeout(n: int) -> int:
    return 400 if n == 1 else 270


def split_batches(batches: dict[str, list[int]], n: int) -> list[dict[str, list[int]]]:
    split = {mode: np.array_split(seeds, n) for mode, seeds in batches.items()}
    return [{mode: parts[i].tolist() for mode, parts in split.items()} for i in range(n)]


def wait_for_fleet(ready, mode: str, count: int, uuid: str, expected_ready: int) -> float:
    """Wait for distinct prepared GPUs before starting a timed batch."""
    if count == 1:
        return 0.0
    start = time.perf_counter()
    ready[(mode, uuid)] = True
    while ready.len() < expected_ready:
        if time.perf_counter() - start > 65:
            raise TimeoutError(f"{mode} fleet did not rendezvous with {count} GPUs")
        time.sleep(.25)
    return time.perf_counter() - start


@app.function(**(gpu_function | {"gpu": GPU, "timeout": 400}),
              single_use_containers=True, max_containers=16, retries=0)
def scaling_shard(batches: dict[str, list[int]], ready, count: int) -> dict:
    from bench.real_cpu import DESIGN
    from reimage import gpu, run

    entered = time.perf_counter()
    device = gpu_info()
    device["uuid"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10
    ).strip()
    if device["count"] != 1 or device["model"] != MODEL or not device["uuid"]:
        raise ValueError(f"expected one {MODEL}, got {device['all']}")
    cells, layout, prep = _charville()
    design = {**DESIGN, "cells": cells, "backend": "gpu"}
    t0 = time.perf_counter()
    prepared = {("gpu", "v1", id(nbr), layout["K"]): gpu.prepare(nbr, layout["K"])
                for nbr in cells["_nbr"].values()}
    prep["gpu_upload"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    run(design, np.array(batches["strong"][:1]), _prepared=prepared)
    prep["warm_up"] = time.perf_counter() - t0
    results = {}
    for phase, (mode, seeds) in enumerate(batches.items(), start=1):
        wait = wait_for_fleet(ready, mode, count, device["uuid"], phase * count)
        wall_start, t0 = time.time(), time.perf_counter()
        features = run(design, np.array(seeds), _prepared=prepared)
        compute, wall_end = time.perf_counter() - t0, time.time()
        results[mode] = {"seeds": seeds, "features": features, "compute_s": compute,
                         "compute_window": [wall_start, wall_end], "rendezvous_wait_s": wait}
    return {"batches": results, "gpu": device, "setup_s": prep,
            "in_function_s": time.perf_counter() - entered}


def validate_point(parts: list[dict], n: int, expected: dict[str, list[int]], reference=None):
    """Check complete, distinct real devices and exact feature parity before publishing metrics."""
    hw = [p["gpu"] for p in parts]
    if (len(hw) != n or any(g["count"] != 1 or g["model"] != MODEL for g in hw)
            or any(not g["uuid"] for g in hw) or len({g["uuid"] for g in hw}) != n):
        models = sorted({g["model"] for g in hw})
        uuids = len({g["uuid"] for g in hw if g["uuid"]})
        raise ValueError(f"expected {n} distinct {MODEL} devices, got {uuids} UUIDs from {len(hw)} workers: {models}")
    rows = {}
    for mode, seeds in expected.items():
        shards = [p["batches"][mode] | {"gpu": p["gpu"]} for p in parts]
        windows = np.asarray([s["compute_window"] for s in shards])
        durations = np.asarray([s["compute_s"] for s in shards])
        if (not np.isfinite(windows).all() or not np.isfinite(durations).all()
                or not (windows[:, 1] > windows[:, 0]).all() or not (durations > 0).all()):
            raise ValueError(f"invalid {mode} timing")
        ov = overlap(shards, seeds)
        if not ov["seed_coverage_exact"] or not ov["distinct_task_ids"] or not ov["all_overlapped"]:
            raise ValueError(f"{mode} coverage, distinct task or common overlap check failed: {ov}")
        if any(s["features"].ndim != 3 or len(s["features"]) != len(s["seeds"]) for s in shards):
            raise ValueError(f"invalid {mode} feature shape")
        order = np.argsort([seed for s in shards for seed in s["seeds"]])
        features = np.concatenate([s["features"] for s in shards])[order]
        if reference is None:
            reference = features  # mode order is strong then weak, with one GPU first
        if not np.array_equal(features, reference[:len(seeds)], equal_nan=True):
            raise ValueError(f"{mode} features differ from one-GPU strong reference")
        span = ov["compute_span_s"]
        rows[mode] = {**ov, "studies": len(seeds), "studies_per_hour": len(seeds) * 3600 / span,
                      "feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                      "exact_features_vs_one_gpu": True,
                      "shards": [{k: v for k, v in s.items() if k not in {"features", "gpu"}} for s in shards]}
    if "weak" not in rows:
        rows["weak"] = rows["strong"] | {"shared_with_strong": True}
    return rows, reference


def summarize(repeats: list[dict]) -> dict:
    """Use paired, repeat-local baselines. Min/max describe observed spread."""
    summary = {"strong": {}, "weak": {}}
    for mode in summary:
        for n in sorted(FLEET):
            values = []
            for repeat in repeats:
                if str(n) not in repeat:
                    continue
                baseline = repeat["1"][mode]["studies_per_hour"]
                row = repeat[str(n)][mode]
                speedup = row["studies_per_hour"] / baseline
                values.append({"studies_per_hour": row["studies_per_hour"], "throughput_speedup": speedup,
                               "efficiency": speedup / n, "compute_span_s": row["compute_span_s"]})
            if len(values) < 2:
                continue
            summary[mode][str(n)] = {"repeats": len(values)} | {
                key: {"median": median(samples), "min": min(samples), "max": max(samples)}
                for key, samples in {k: [v[k] for v in values] for k in values[0]}.items()
            }
    return summary


def save(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


@app.local_entrypoint()
def scaling(repeats: int = 3, cap_usd: float = 20.0, pilot: bool = False):
    """Write raw validated points and a spend ledger even when the sweep aborts."""
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes before launching")
    if repeats not in (2, 3):
        raise SystemExit("use two or three repeats")
    counts = (1,) if pilot else FLEET
    runs = 1 if pilot else repeats
    ceiling = runs * sum(n * _timeout(n) for n in counts) * RATE / 3600
    if ceiling > min(cap_usd, 20):
        raise SystemExit(f"worker-time reserve ${ceiling:.2f} exceeds allocation")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "bench/results" / f"modal-fleet-scaling-{stamp}.json"
    spend = ROOT / "bench/results" / f"modal-spend-fleet-scaling-{stamp}.json"
    commit = git("rev-parse", "HEAD")
    ledger = {"status": "ESTIMATE", "source_commit": commit, "usd_per_worker_hour": RATE,
              "worker_time_reservation_usd": ceiling, "points": [],
              "basis": "Recorded Modal list rates. Estimates from measured function runtime; excludes platform overhead and is not an invoice."}
    report = {"status": "INCOMPLETE", "source_commit": commit, "pilot": pilot,
              "hardware_request": {"gpu": GPU, "cpu_cores": 8, "memory_gib": 32},
              "labels": {"timings": "MEASURED", "study_design": "SIMULATED on measured Charville tissue",
                         "ideal_line": "ANALYTIC perfect scaling", "spend": "ESTIMATE"},
              "scope": "Re-imaging through feature aggregation only. Strong first, then weak using the same prepared device state. "
                       "Workers rendezvous before each batch. The identical 16-GPU strong and weak work is measured once. "
                       "Compute span is first shard start to last shard finish using host clocks (clock skew unmeasured). "
                       "Modal wall measures the whole point, including setup, both batches and transfer. "
                       "Weak efficiency is one-GPU compute span divided by N-GPU compute span for N times the workload.",
              "strong_studies": STRONG, "weak_studies_per_gpu": WEAK_PER_GPU, "repeats": [], "failures": []}
    for j in range(runs):
        repeat, reference = {}, None
        report["repeats"].append(repeat)
        for n in counts:
            expected = workload(n, j * STRONG)
            shards = split_batches(expected, n)
            point = {"repeat": j + 1, "gpus": n, "reserved_usd": n * _timeout(n) * RATE / 3600}
            ledger["points"].append(point)
            save(spend, ledger)
            print(f"repeat {j + 1}, {n} GPUs, reserve ${point['reserved_usd']:.2f}", flush=True)
            t0 = time.perf_counter()
            try:
                with modal.Dict.ephemeral() as ready:
                    parts = list(scaling_shard.with_options(timeout=_timeout(n)).map(
                        shards, kwargs={"ready": ready, "count": n}, order_outputs=True))
                wall = time.perf_counter() - t0
                point["estimated_usd"] = sum(p["in_function_s"] for p in parts) * RATE / 3600
                rows, reference = validate_point(parts, n, expected, reference)
                repeat[str(n)] = {**rows, "modal_wall_s": wall,
                                  "workers": [{k: v for k, v in p.items() if k != "batches"} for p in parts]}
            except Exception as exc:
                point["failure"] = str(exc)
                report["failures"].append({"repeat": j + 1, "gpus": n, "reason": str(exc)})
                print(f"repeat {j + 1}, {n} GPUs: UNMEASURED ({exc})", flush=True)
                if n == 1:
                    break  # No verified baseline to compare any later point against.
            finally:
                save(spend, ledger)
                save(out, report)
    report["status"] = "PARTIAL" if report["failures"] else "MEASURED"
    if not pilot:
        report["summary"] = summarize(report["repeats"])
    save(out, report)
    print(out, flush=True)
