"""Measure the legacy Charville v2 permutation screen on one H100.

DATA_DIR=<prepared Charville directory> MODAL_PROFILE=fhl-08 \
    uv run --with modal modal run bench/modal_charville_v2_gpu.py --project-spent-usd <estimate excluding this task>

The launcher adds prior Charville replay receipts in this checkout to that project baseline.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from time import perf_counter

import modal

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
FILES = ("charville_labels.csv", "charville_cells.npz", "charville_channels.npz")
GPU, CORES, MEM_MIB, TIMEOUT = "H100", 8, 32768, 1500
RETENTION_S, RESERVATION_USD = 60, 2.50
RATES = {"gpu_s": 0.001097, "physical_core_s": 0.0000131, "gib_s": 0.00000222,
         "source": "https://modal.com/pricing", "checked_utc": "2026-11-22"}
USD_S = RATES["gpu_s"] + CORES * RATES["physical_core_s"] + MEM_MIB / 1024 * RATES["gib_s"]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).rstrip("\n")


def ignore_untracked(path):
    if not hasattr(ignore_untracked, "tracked"):
        ignore_untracked.tracked = set(git("ls-files", "-z").split("\0"))
    return Path(path).as_posix() not in ignore_untracked.tracked


image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .apt_install("libgomp1").uv_sync(str(ROOT)).uv_pip_install("cupy-cuda12x")
         .env({"PYTHONPATH": "/root/preflight", "CUDA_HOME": "/usr/local/cuda"})
         .workdir("/root/preflight"))
for filename in FILES:
    image = image.add_local_file(LOCAL_DATA / filename, f"/data/{filename}")
image = image.add_local_dir(ROOT, "/root/preflight", ignore=ignore_untracked)
app = modal.App("preflight-charville-v2-permutation")


@app.function(image=image, gpu=GPU, cpu=(CORES, CORES), memory=(MEM_MIB, MEM_MIB), timeout=TIMEOUT,
              scaledown_window=RETENTION_S, retries=0,
              env={"OPENBLAS_NUM_THREADS": str(CORES), "OMP_NUM_THREADS": str(CORES), "MKL_NUM_THREADS": str(CORES)})
def measure():
    launch_start = perf_counter()
    import cupy as cp
    import numpy as np

    from audit.charville_v2 import _source_fingerprint
    from audit.runner import conditional_permutation_cutoff
    from bench.charville_v2_gpu import BATCH, TOLERANCE, permutation_screen, same_outputs
    from bench.screen import _sha, cpu_config
    from core.cohorts import load
    from core.grammar import feature_names, grammar_hash, grammar_v2
    from core.truth import truth_v2

    baseline = json.loads(Path("/root/preflight/results/charville-v2.json").read_text())
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError("expected exactly one CUDA device")
    device = cp.cuda.runtime.getDeviceProperties(0)
    name = device["name"].decode()
    if "H100" not in name or device["totalGlobalMem"] < 75 * 1024 ** 3:
        raise RuntimeError(f"wrong GPU for this benchmark: {name}, {device['totalGlobalMem']} bytes")
    started = perf_counter()
    cells = load("charville", data_dir=Path("/data"), include_channels=True)
    X = np.asarray(truth_v2(cells), dtype=np.float64)
    args = X, cells["time"], cells["event"], cells["strata"]
    grammar = grammar_v2(list(cells["type_names"]))
    names = feature_names(grammar)
    fingerprint = _source_fingerprint(cells, cells.get("channel_stats"))
    if (fingerprint != baseline["source_fingerprint"] or grammar_hash(grammar) != baseline["grammar_v2_hash"]
            or X.shape != (159, 3780)):
        raise ValueError(f"Charville inputs did not match the original receipt: {fingerprint}, {X.shape}")
    prep_s = perf_counter() - started
    n_resamples, seed = baseline["maxT"]["n_resamples"], baseline["seed"]
    if (n_resamples, seed) != (9999, 0):
        raise ValueError("legacy receipt does not declare 9,999 resamples and seed zero")
    print(f"Verified {X.shape[0]} patients and {X.shape[1]} features. Data and feature preparation {prep_s:.2f}s.", flush=True)

    # Warm CUDA and BLAS before the two prepared-matrix timers.
    for xp in (cp, np):
        permutation_screen(X[:, :8], *args[1:], n_resamples=2, seed=seed, xp=xp, block=2)
    cp.cuda.Device(0).synchronize()
    memory = {"baseline_device_used_bytes": 0, "peak_sampled_device_used_bytes": 0,
              "peak_pool_reserved_bytes": 0, "samples": 0,
              "basis": "device used via total minus free and CuPy pool reservation after each synchronized score block"}

    def sample():
        free, total = cp.cuda.runtime.memGetInfo()
        memory["peak_sampled_device_used_bytes"] = max(memory["peak_sampled_device_used_bytes"], total - free)
        pool = cp.get_default_memory_pool().total_bytes()
        memory["peak_pool_reserved_bytes"] = max(memory["peak_pool_reserved_bytes"], pool)
        memory["samples"] += 1

    sample()
    memory["baseline_device_used_bytes"] = memory["peak_sampled_device_used_bytes"]
    runs, outputs = {}, {}
    for label, xp in (("gpu", cp), ("cpu", np)):
        start = perf_counter()
        out = permutation_screen(*args, n_resamples=n_resamples, seed=seed, xp=xp, block=BATCH,
                                 sample_memory=sample if label == "gpu" else None)
        cutoff = conditional_permutation_cutoff(out["null_z"])
        if label == "gpu":
            cp.cuda.Device(0).synchronize()
        wall = perf_counter() - start
        outputs[label] = out
        runs[label] = {"label": "MEASURED", "backend": "cupy_float64" if label == "gpu" else "numpy_float64",
                       "screen_wall_s": wall, "synchronized_score_products_wall_s": out["product_wall_s"],
                       "maxT": out["maxT"], "p": out["p"], "conditional_cutoff_alpha_05": cutoff,
                       "omnibus": out["omnibus"]}
        print(f"{label}: {wall:.3f}s, maxT {out['maxT']:.12f}, p {out['p']:.4f}", flush=True)

    differences = same_outputs(outputs["gpu"], outputs["cpu"])
    gpu_out = outputs["gpu"]
    order = np.lexsort((np.arange(len(names)), -np.abs(gpu_out["z"])))[:len(baseline["maxT"]["top_hits"])]
    top = [{"grammar_index": int(i), "feature": names[i], "z": float(gpu_out["z"][i]),
            "adjusted_p": float(gpu_out["adjusted_p"][i])} for i in order]
    target = baseline["maxT"]
    if (gpu_out["p"] != target["p"] or not np.isclose(gpu_out["maxT"], target["statistic"], **TOLERANCE)
            or gpu_out["omnibus"]["p"] != baseline["omnibus"]["p"]
            or not np.isclose(runs["gpu"]["conditional_cutoff_alpha_05"], target["conditional_cutoff_alpha_05"], **TOLERANCE)
            or any(a["grammar_index"] != b["grammar_index"] or a["feature"] != b["feature"]
                   or a["adjusted_p"] != b["adjusted_p"] or not np.isclose(a["z"], b["z"], **TOLERANCE)
                   for a, b in zip(top, target["top_hits"], strict=True))):
        raise ValueError("H100 results differ from the original Charville v2 receipt")
    return {"status": "MEASURED", "cohort": "Charville", "exploratory": True,
            "outcomes_already_seen": True, "clinical_claims": False,
            "scope": "Prepared feature matrix through legacy maxT, adjusted p-values, cutoff and omnibus. CPU generates "
                     "paired within-stratum shuffles, recalculates masked Breslow residuals and reduces the final outputs. "
                     "GPU computes batched float64 score numerators and variances. Timers include host/device transfers "
                     "and exclude data and feature preparation, warmup, parity checks and leave-one-patient-out audit.",
            "hardware": {"gpu": name, "gpu_count": 1, "gpu_memory_bytes": device["totalGlobalMem"],
                         "modal_request": GPU, "cpu_physical_cores_requested": CORES, "memory_mib": MEM_MIB,
                         "cpu": cpu_config(), "modal_task_id": os.environ.get("MODAL_TASK_ID"),
                         "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(), "cupy": cp.__version__,
                         "numpy": np.__version__},
            "input": {"patients": X.shape[0], "features": X.shape[1], "resamples": n_resamples, "seed": seed,
                      "source_fingerprint": fingerprint, "grammar_v2_hash": grammar_hash(grammar), "matrix_sha256": _sha(X),
                      "nan_cells": int(np.isnan(X).sum()), "source": "https://zenodo.org/records/13179600"},
            "preparation_wall_s": prep_s, "runs": runs,
            "same_container_speedup": {"label": "MEASURED", "cpu_over_gpu": runs["cpu"]["screen_wall_s"] / runs["gpu"]["screen_wall_s"]},
            "gpu_batches": gpu_out["score_batches"], "gpu_memory": memory,
            "parity": {"tolerance": TOLERANCE, "same_container_null_and_observed_max_abs_difference": differences,
                       "same_container_p_exact": True, "historical_receipt": "results/charville-v2.json",
                       "historical_source_commit": baseline["code_revision"], "historical_maxT": target["statistic"],
                       "historical_p": target["p"], "historical_top_hits_identical": True,
                       "historical_wall_s_different_scope": baseline["runtime"]["wall_s"]},
            "maxT": {"statistic": gpu_out["maxT"], "p": gpu_out["p"], "top_hits": top},
            "function_wall_s": perf_counter() - launch_start}


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


@app.local_entrypoint()
def main(project_spent_usd: float):
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes before launching")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    ledger_path = ROOT / "bench" / "results" / f"modal-spend-charville-v2-gpu-{stamp}.json"
    receipt = ROOT / "results" / f"charville-v2-gpu-{stamp}.json"
    prior = sum(json.loads(p.read_text())["run_estimated_usd"] for p in ledger_path.parent.glob("modal-spend-charville-v2-gpu-*.json"))
    if prior + RESERVATION_USD > 5:
        raise SystemExit("Charville H100 task spend plus reservation exceeds $5")
    if (not math.isfinite(project_spent_usd) or project_spent_usd < 0 or
            project_spent_usd + prior + RESERVATION_USD > 150):
        raise SystemExit("project baseline, prior Charville receipts and reservation must fit the $150 ceiling")
    source = git("rev-parse", "HEAD")
    ledger = {"label": "ESTIMATE", "source_commit": source, "state": "reserved", "rates": RATES,
              "run_estimated_usd": RESERVATION_USD, "task_cap_usd": 5.0, "project_cap_usd": 150.0,
              "prior_task_estimate_usd": prior, "prior_project_estimate_usd": project_spent_usd,
              "prior_project_estimate_scope": "Project baseline excluding Charville replay receipts in this ledger.",
              "running_project_estimate_usd": project_spent_usd + prior + RESERVATION_USD,
              "reservation_usd": RESERVATION_USD, "timeout_s": TIMEOUT,
              "basis": "GPU plus capped CPU and memory list rates, charged for measured function wall plus retention. "
                       "An unresolved or failed call retains its reservation. Startup and credits can differ."}
    write(ledger_path, ledger)
    call = measure.spawn()
    ledger.update(state="submitted", call_id=call.object_id)
    write(ledger_path, ledger)
    print(f"Submitted {call.object_id}", flush=True)
    try:
        result = call.get()
    except Exception:
        ledger["state"] = "failed"
        write(ledger_path, ledger)
        raise
    result.update(source_commit=source, spend_receipt=str(ledger_path.relative_to(ROOT)))
    ledger.update(state="completed", function_wall_s=result["function_wall_s"],
                  run_estimated_usd=(result["function_wall_s"] + RETENTION_S) * USD_S)
    ledger["running_project_estimate_usd"] = project_spent_usd + prior + ledger["run_estimated_usd"]
    write(receipt, result)
    write(ledger_path, ledger)
    print(f"Saved {receipt.name} and {ledger_path.name}", flush=True)
