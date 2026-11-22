"""R survival and permGPU baselines on the Charville v2 screen (bench/r_baselines.py), one Modal container.

    MODAL_PROFILE=fhl-08 DATA_DIR=<dir> uv run --with modal modal run bench/modal_r_baselines.py

The design point's container class (A100 + 8 cores + 64 GiB). R 4.1 (Ubuntu 22.04 r-base), survival and Biobase
from apt; permGPU 0.15 from its CRAN mirror, built in the container so a failed build is recorded, not hidden.
Each arm runs 3 times on the same inputs; times exclude loading data and building X (equal for every arm).
Commit first (source_commit is the code that runs); a ceiling over --cap-usd is refused. Writes
bench/results/r-baselines-<utc>.json and logs the spend in bench/results/modal-spend-rbaselines.json.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

from bench.modal_design import (CLASS, CORES, DATA_DIR, GPU, LOCAL_DATA, MEMORY_MIB, RATES, REPO, USD_PER_HOUR,
                                _git, _tracked)

APP_DIR = "/root/preflight"
PERMGPU = ("https://github.com/cran/permGPU", "3c64fbba3c0611640586ff64ecf0eca3882c8998")  # CRAN 0.15 mirror
SPEND = REPO / "bench" / "results" / "modal-spend-rbaselines.json"
THREADS = {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}  # ours: 8 BLAS threads
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})  # tzdata (pulled in by R) otherwise prompts
    .apt_install("git", "r-base-core", "r-cran-survival", "r-bioc-biobase", "r-cran-runit", "r-cran-foreach",
                 "autoconf")
    .run_commands(f"git clone -q {PERMGPU[0]} /root/permGPU && git -C /root/permGPU checkout -q {PERMGPU[1]}")
    .uv_sync(str(REPO))
    .env({"DATA_DIR": DATA_DIR, "PYTHONPATH": APP_DIR, "CUDA_HOME": "/usr/local/cuda", **THREADS})
    .add_local_file(LOCAL_DATA / "charville_labels.csv", f"{DATA_DIR}/charville_labels.csv", copy=True)
    .add_local_file(LOCAL_DATA / "charville_cells.npz", f"{DATA_DIR}/charville_cells.npz", copy=True)
    # the v2 channels: core.cohorts' own cache of charville_raw_data.zip (50 MB vs 243 MB), validated on load
    .add_local_file(LOCAL_DATA / "charville_channels.npz", f"{DATA_DIR}/charville_channels.npz", copy=True)
    .workdir(APP_DIR)
    .add_local_dir(REPO, APP_DIR, ignore=lambda rel: not _tracked(rel))
)
app = modal.App("preflight-r-baselines")
# permGPU's configure hard-codes -arch=sm_20 (removed from CUDA 9+) and links -lcuda from lib64, where CUDA 12
# ships only a stub; under nvcc 12's default C++17, ranker.h's global rank() clashes with std::rank (the source
# has `using namespace std`), so the call sites are qualified as ::rank. These three edits are the whole port.
PORT = ["sed -i 's/-arch=sm_20/-arch=sm_80/' configure",
        "sed -i 's|LDFLAGS=\"-L${CUDA_HOME}/lib64\"|LDFLAGS=\"-L${CUDA_HOME}/lib64 -L${CUDA_HOME}/lib64/stubs\"|' configure",
        "sed -i 's/^\\([[:space:]]*\\)rank(a, ranks, method);/\\1::rank(a, ranks, method);/' src/permGPU.cu"]


def _run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _host() -> dict:
    def out(cmd):
        p = _run(cmd, timeout=60)
        return (p.stdout + p.stderr).strip()
    return {"cpu_model": next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                               if l.startswith("model name")), None),
            "cpu_count": os.cpu_count(), "affinity_size": len(os.sched_getaffinity(0)),
            "load_average_start": list(os.getloadavg()), "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "gpu": out(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
            "nvcc": out(["/usr/local/cuda/bin/nvcc", "--version"]).splitlines()[-1],
            "r": out(["R", "--version"]).splitlines()[0],
            "r_packages": out(["Rscript", "-e", "for (p in c('survival','Biobase')) cat(p, format(packageVersion(p)), '\\n')"]),
            "r_blas": out(["Rscript", "-e", "cat(extSoftVersion()[['BLAS']], La_library())"]),
            "threads": {k: os.environ.get(k) for k in THREADS}}


@app.function(image=image, gpu=GPU, cpu=CORES, memory=MEMORY_MIB, timeout=3600)
def baselines(repeats: int = 3, n_resamples: int = 9999, seed: int = 0) -> dict:
    import sys
    import time

    import numpy as np

    sys.path.insert(0, APP_DIR)
    from bench import r_baselines as rb
    from core import cohorts
    from core.grammar import feature_names, grammar_v2
    from core.truth import truth_v2

    host = _host()
    t = time.perf_counter()
    cells = cohorts.load("charville", include_channels=True)
    X = np.asarray(truth_v2(cells), np.float64)
    tm, ev = np.asarray(cells["time"], np.float64), np.asarray(cells["event"], np.float64)
    st = np.asarray(cells["strata"], np.int32)
    names = feature_names(grammar_v2(list(cells["type_names"])))
    prep_s = time.perf_counter() - t
    n, C = X.shape
    P = rb.draws(st, n_resamples, seed)
    P0 = rb.draws(np.zeros(n, np.int32), n_resamples, seed)  # unstratified: permGPU's workload, for ours

    d = Path("/tmp/w")
    d.mkdir(exist_ok=True)
    rb.write_inputs(d, X, tm, ev, st, P)

    out = {"prep_s": prep_s, "shape": {"patients": n, "patterns": C, "events": int(ev.sum()),
                                       "nan_cells": int(np.isnan(X).sum()), "strata": int(len(np.unique(st))),
                                       "shuffles": n_resamples, "seed": seed}, "errors": {}}

    # ours, stratified (the audit's workload)
    times, res = [], None
    for _ in range(repeats):
        t = time.perf_counter()
        res = rb.ours(X, tm, ev, st, P)
        times.append(time.perf_counter() - t)
    out["ours"] = {"timing": rb.spread(times), **{k: res[k] for k in ("maxT", "p", "top")},
                   "top_name": names[res["top"]]}
    z_ours, null_ours = res["z"], res["null_maxT"]

    # R survival, same X, same P
    times, wall, log = [], [], ""
    rscript = ["Rscript", f"{APP_DIR}/bench/r/survival_perm.R", str(d), str(int(CORES))]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}  # 8 forks, one thread each
    for _ in range(repeats):
        t = time.perf_counter()
        p = _run(rscript, env=env, timeout=1800)
        wall.append(time.perf_counter() - t)
        log = (p.stdout + p.stderr)[-3000:]
        if p.returncode:
            out["errors"]["r"] = log
            break
        times.append(float((d / "elapsed.txt").read_text()))
    if "r" not in out["errors"]:
        z_r, null_r = np.fromfile(d / "z.f64"), np.fromfile(d / "null.f64")
        s = rb.summary(z_r, null_r)
        out["r"] = {"timing": rb.spread(times), "rscript_wall": rb.spread(wall), **s, "top_name": names[s["top"]],
                    "parity_vs_ours": {"z_max_abs_diff": float(np.max(np.abs(z_r - z_ours))),
                                       "null_maxT_max_abs_diff": float(np.max(np.abs(null_r - null_ours))),
                                       "same_p": s["p"] == res["p"], "same_top": s["top"] == res["top"]}}

    # permGPU: build, then the closest workload it expresses (no strata, its own shuffles)
    build = _run(["bash", "-c", "set -e; rm -rf /tmp/pg; cp -r /root/permGPU /tmp/pg; cd /tmp/pg; "
                  + "; ".join(PORT) + "; R CMD INSTALL --configure-args='--with-cuda-home=/usr/local/cuda' ."],
                 timeout=1200)
    out["permgpu_build"] = {"returncode": build.returncode, "port": PORT, "source": PERMGPU,
                            "log_tail": (build.stdout + build.stderr)[-4000:]}
    times0, res0 = [], None
    for _ in range(repeats):
        t = time.perf_counter()
        res0 = rb.ours(X, tm, ev, np.zeros(n, np.int32), P0)
        times0.append(time.perf_counter() - t)
    out["ours_unstratified"] = {"timing": rb.spread(times0), **{k: res0[k] for k in ("maxT", "p", "top")},
                                "top_name": names[res0["top"]]}
    if build.returncode == 0:
        times, wall = [], []
        for _ in range(repeats):
            t = time.perf_counter()
            p = _run(["Rscript", f"{APP_DIR}/bench/r/permgpu_cox.R", str(d), str(n_resamples), str(seed)], timeout=1800)
            wall.append(time.perf_counter() - t)
            if p.returncode:
                out["errors"]["permgpu"] = (p.stdout + p.stderr)[-3000:]
                break
            times.append(float((d / "permgpu_elapsed.txt").read_text()))
        if "permgpu" not in out["errors"]:
            stat = np.fromfile(d / "permgpu_stat.f64")
            fwer = np.fromfile(d / "permgpu_fwerp.f64")
            ref = rb.permgpu_stat(X, tm, ev)
            full = np.isfinite(X).all(0)
            rel = np.abs(stat - ref) / np.maximum(np.abs(ref), 1e-12)
            top = int(np.nanargmax(stat))
            out["permgpu"] = {
                "timing": rb.spread(times), "rscript_wall": rb.spread(wall),
                "top": top, "top_name": names[top], "top_stat_chi2": float(stat[top]),
                "top_fwer_p": float(fwer[top]),
                "parity_vs_transcription": {"max_rel_diff_all": float(np.nanmax(rel)),
                                            "max_rel_diff_complete_columns": float(np.nanmax(rel[full])),
                                            "within_tol": bool(np.nanmax(rel) <= rb.PERMGPU_TOL),
                                            "tol": rb.PERMGPU_TOL, "same_top": top == int(np.argmax(ref)),
                                            "nan_stats": int(np.isnan(stat).sum())},
                "ours_unstratified_score_chi2_top": float(np.max(ref))}
    host["load_average_end"] = list(os.getloadavg())
    out["host"] = host
    return out


def _log(entry: dict) -> None:
    log = json.loads(SPEND.read_text()) if SPEND.exists() else {
        "workspace_profile": os.environ.get("MODAL_PROFILE", "fhl-08"), "rates_usd_per_hour": RATES, "entries": []}
    es = [e for e in log["entries"] if e["utc"] != entry["utc"]] + [entry]
    log["entries"] = es
    log["total_usd"] = round(sum(e.get("actual", {}).get("est_usd", e["est_usd"]) for e in es), 3)
    SPEND.write_text(json.dumps(log, indent=2) + "\n")


@app.local_entrypoint()
def main(repeats: int = 3, cap_usd: float = 5.0, timeout_s: int = 3600):
    from bench import r_baselines as rb

    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes first: source_commit must be the code that runs")
    commit = _git("rev-parse", "HEAD")
    ceiling = round(timeout_s / 3600 * USD_PER_HOUR, 2)
    if ceiling > cap_usd:
        raise SystemExit(f"ceiling ${ceiling} > cap ${cap_usd}: ask the lead first")
    start = datetime.now(timezone.utc)
    entry = {"utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "run": f"R survival + permGPU baselines, {repeats} repeats",
             "containers": 1, "container_class": CLASS, "cap_usd": cap_usd, "est_usd": ceiling,
             "basis": f"{timeout_s} s ceiling x ${USD_PER_HOUR:.4f}/h", "source_commit": commit}
    _log(entry)
    error, out = None, None
    try:
        out = baselines.with_options(timeout=timeout_s).remote(repeats)
    except Exception as e:  # noqa: BLE001 - a failed run is recorded as FAILED, with its spend
        error = repr(e)[-2000:]
    end = datetime.now(timezone.utc)
    wall = (end - start).total_seconds()
    entry["actual"] = {"modal_wall_s": round(wall, 1), "est_usd": round(wall / 3600 * USD_PER_HOUR, 3),
                       "basis": "container x local wall (an upper bound: includes image build and queueing)"}
    speedups = {}
    if out:
        for arm, base in (("r", "ours"), ("permgpu", "ours_unstratified")):
            if arm in out:
                speedups[f"ours_vs_{arm}"] = rb.ratio(out[arm]["timing"], out[base]["timing"])
    result = {"status": "MEASURED" if out and "r" in out else "FAILED", "error": error, "source_commit": commit,
              "method": rb.__doc__.strip(), "audit": rb.AUDIT, "container_class": CLASS,
              "speedup": speedups, "result": out,
              "modal": {"app": "preflight-r-baselines", "started_utc": start.isoformat(),
                        "finished_utc": end.isoformat(), "wall_s": wall}}
    path = REPO / "bench" / "results" / f"r-baselines-{start:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    entry["actual"]["result"] = str(path.relative_to(REPO))
    _log(entry)
    print(result["status"], json.dumps(speedups), error or "")
    print(f"Saved {path}")
