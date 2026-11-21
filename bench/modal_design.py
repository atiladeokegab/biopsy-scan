"""The batched design point on Modal: each arm in its own container of one class, same studies (bench/design_batch.py).

    DATA_DIR=<dir> MODAL_PROFILE=fhl-08 uv run --with modal modal run bench/modal_design.py --studies 20
    ... --arms hybrid --timeout-s 1800 --compare-with bench/results/modal-design-<utc>.json

--arms picks arms (default lab,numba,gpu). --compare-with adds a committed MEASURED result's arms to the
file, side by side, after checking every new study's X sha256 and batched maxT against them.

--studies is per stream: calibration, null and alternative each get that many studies (the lab arm
skips calibration). Each arm runs in its own A100 container with 8 cores and 64 GiB, in parallel.
Commit first: the run refuses a dirty tracked tree, so source_commit is the code that ran. A run whose
ceiling (3 containers x timeout x rate) is over --cap-usd (default 20) is refused before it starts.
Writes bench/results/modal-design-<utc>.json, MEASURED only when its parity gate passes (parity.scope says
what it covered: --compare-with checks z only through maxT), a clean tree and one
container class (SMOKE when --cpu-arms-on-cpu put the CPU arms in GPU-less containers), else FAILED; logs the spend in the workspace ledger bench/results/modal-spend.json before and after the run.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
APP_DIR, DATA_DIR, PIN = "/root/preflight", "/data", "/root/pins/atlas_simulation.py"
LOCAL_DATA = Path(os.environ.get("DATA_DIR", REPO / "data"))
GPU, CORES, MEMORY_MIB = "A100", 8.0, 65536
ARMS = ("lab", "numba", "gpu", "hybrid", "fused")
CLASS = f"Modal {GPU} + {CORES:g} cores + {MEMORY_MIB // 1024} GiB"
RATES = {"A100": 2.0988, "cpu_core": 0.04716, "memory_gib": 0.007992,
         "source": "bench/results/modal-spend.json on modal-gpu (modal.com/pricing read 2026-11-21T17:36Z)"}
USD_PER_HOUR = RATES["A100"] + CORES * RATES["cpu_core"] + MEMORY_MIB / 1024 * RATES["memory_gib"]
SPEND = REPO / "bench" / "results" / "modal-spend.json"  # the one workspace ledger
# workqueue: the CPU arms fork workers after the parent's JIT warm-up, and GNU OpenMP aborts a fork
SINGLE = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
          "NUMBA_THREADING_LAYER": "workqueue"}
RAW = "https://raw.githubusercontent.com/atiladeokegab/sota-parallelism-nov21"
ATLAS_SHA = "79a77926aa0c07fc2c85574736f0f1396767ded0"  # must equal design_batch.ATLAS_PIN[0]; main() checks


def _tracked(rel: Path) -> bool:
    if not getattr(_tracked, "files", None):  # lazy: the container re-imports this module and has no .git
        out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True).stdout
        _tracked.files = set(filter(None, out.split("\0")))
    return rel.as_posix() in _tracked.files


def _timeout_s(studies: int) -> int:
    # shortcut: sized from the lab arm (~8 s/study/core, 2 x studies over 8 workers) plus 30 min of headroom.
    return int(min(6 * 3600, 1800 + 2 * studies * 8 / 8 * 1.5))


# build-modal's image (gpu/modal_app.py, #75): CUDA devel for numba's CUDA target, uv-synced project.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "git")
    .uv_sync(str(REPO))
    .env({"DATA_DIR": DATA_DIR, "PYTHONPATH": APP_DIR, "CUDA_HOME": "/usr/local/cuda", "ATLAS_SIMULATION": PIN})
    .run_commands("mkdir -p /root/pins", f"curl -fsSL {RAW}/{ATLAS_SHA}/atlas/simulation.py -o {PIN}")
    .add_local_file(LOCAL_DATA / "charville_labels.csv", f"{DATA_DIR}/charville_labels.csv", copy=True)
    .add_local_file(LOCAL_DATA / "charville_cells.npz", f"{DATA_DIR}/charville_cells.npz", copy=True)
    .workdir(APP_DIR)
    .add_local_dir(REPO, APP_DIR, ignore=lambda rel: not _tracked(rel))
)
app = modal.App("preflight-design-batch")


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _arm(name: str, plan: dict) -> dict:
    import shutil
    import sys
    import time

    sys.path.insert(0, APP_DIR)
    from bench import design_batch as db
    from core.cohorts import load

    smi = shutil.which("nvidia-smi")
    host = {"cpu_count": os.cpu_count(), "affinity_size": len(os.sched_getaffinity(0)),
            "load_average_start": list(os.getloadavg()), "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "gpu": smi and subprocess.run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                          capture_output=True, text=True, timeout=30, check=False).stdout.strip(),
            "thread_environment": {k: os.environ.get(k) for k in SINGLE}}
    t = time.perf_counter()
    cells = load("charville")
    load_s = time.perf_counter() - t
    out = db.run_arm(name, cells, plan, workers=int(CORES))
    host["load_average_end"] = list(os.getloadavg())
    out["host"], out["load_s"] = host, load_s
    for r in out["records"]:
        if "z" in r:
            r["z"] = r["z"].tolist()
    return out


@app.function(image=image, gpu=GPU, cpu=CORES, memory=MEMORY_MIB, timeout=6 * 3600, env=SINGLE)
def arm(name: str, plan: dict) -> dict:
    return _arm(name, plan)


@app.function(image=image, cpu=CORES, memory=MEMORY_MIB, timeout=6 * 3600, env=SINGLE)
def arm_cpu(name: str, plan: dict) -> dict:
    """The same container without a GPU: only for a cheap smoke (--cpu-arms-on-cpu), never a headline."""
    return _arm(name, plan)


def _git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _counted(e: dict) -> float:
    return e.get("actual", {}).get("est_usd", e["est_usd"])


def _log_spend(entry: dict) -> None:
    """One workspace ledger: append the ceiling before a run; after it, swap in the actual and shift later totals."""
    log = json.loads(SPEND.read_text())
    es = log["entries"]
    at = next((k for k, e in enumerate(es) if (e["utc"], e["run"]) == (entry["utc"], entry["run"])), None)
    if at is None:
        es.append(dict(entry))  # a copy: the caller adds "actual" to entry later
        at = len(es) - 1
        es[at]["running_total_usd"] = round((es[at - 1].get("running_total_usd", 0) if at else 0) + _counted(entry), 3)
    else:
        delta = _counted(entry) - _counted(es[at])
        es[at] = entry | {"running_total_usd": es[at]["running_total_usd"]}
        for e in es[at:]:
            e["running_total_usd"] = round(e["running_total_usd"] + delta, 3)
    SPEND.write_text(json.dumps(log, indent=2) + "\n")


@app.local_entrypoint()
def main(studies: int = 20, cap_usd: float = 20.0, cpu_arms_on_cpu: bool = False, arms: str = "lab,numba,gpu",
         timeout_s: int = 0, compare_with: str = ""):
    import numpy as np

    from bench import design_batch as db

    if ATLAS_SHA != db.ATLAS_PIN[0]:
        raise SystemExit("ATLAS_SHA and design_batch.ATLAS_PIN disagree")
    chosen = tuple(a for a in arms.split(",") if a)
    if not chosen or set(chosen) - set(ARMS) or len(set(chosen)) != len(chosen):
        raise SystemExit(f"--arms must be distinct names from {ARMS}")
    prior = json.loads((REPO / compare_with).read_text()) if compare_with else None
    if prior and (prior["status"] != "MEASURED" or prior["design"]["studies_per_stream"] != studies
                  or prior["design"]["root_seed"] != db.ROOT or prior["design"]["streams"] != db.STREAMS
                  or set(prior["arms"]) & set(chosen)):
        raise SystemExit("--compare-with needs a MEASURED result at the same design and seeds, with other arms")
    if studies < 4:
        raise SystemExit("need at least 4 studies per stream (the calibration rank rule)")
    _git("ls-files", "--error-unmatch", "bench/modal_design.py", "bench/design_batch.py")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes first: source_commit must be the code that runs")
    commit = _git("rev-parse", "HEAD")
    timeout = timeout_s or _timeout_s(studies)
    cpu_rate = USD_PER_HOUR - RATES["A100"]
    rates = {a: cpu_rate if cpu_arms_on_cpu and a not in db.GPU_SCORE else USD_PER_HOUR for a in chosen}
    ceiling = round(timeout / 3600 * sum(rates.values()), 2)
    if ceiling > cap_usd:
        raise SystemExit(f"ceiling ${ceiling} > cap ${cap_usd}: ask the lead first")
    plan = {s: studies for s in db.STREAMS}
    start = datetime.now(timezone.utc)
    entry = {"utc": start.strftime("%Y-%m-%dT%H:%MZ"), "run": f"modal-design {studies}/stream, arms {list(chosen)}",
             "containers": len(chosen), "gpu": GPU, "cores": CORES, "memory_gib": MEMORY_MIB / 1024, "cap_usd": cap_usd,
             "est_usd": ceiling, "basis": f"{timeout} s ceiling x per-arm $/h {rates}", "source_commit": commit,
             "container_class": "lab and numba without a GPU (smoke only)" if cpu_arms_on_cpu else "all arms identical"}
    _log_spend(entry)
    fn = {a: arm_cpu if cpu_arms_on_cpu and a not in db.GPU_SCORE else arm for a in chosen}
    calls = {a: fn[a].with_options(timeout=timeout).spawn(a, plan) for a in chosen}
    arms, errors = {}, {}
    for a, call in calls.items():
        try:
            arms[a] = call.get()
        except Exception as e:  # noqa: BLE001 - a failed arm makes the run FAILED, not a crash with no record
            errors[a] = repr(e)[-2000:]
    end = datetime.now(timezone.utc)
    wall = (end - start).total_seconds()
    for o in arms.values():
        for r in o["records"]:
            if "z" in r:
                r["z"] = np.asarray(r["z"])
    plant = next(iter(arms.values()))["setup"]["plant_feature"] if arms else None
    if len(arms) != len(chosen):
        par = {"all": False, "reason": "an arm failed"}
    else:  # one new arm has nothing to match in this run: its parity is against the prior run's arms
        par = db.parity(arms) if len(arms) > 1 else {"all": bool(prior), "this_run": "one arm"}
    if prior and arms:
        par["against_prior"] = {a: db.prior_parity(o["records"], prior["arms"], arm=a, studies_per_stream=studies)
                              for a, o in arms.items()}
        par["all"] = bool(par["all"] and all(v["all"] for v in par["against_prior"].values()))
        par["scope"] = ("against the prior run: X sha256 on every shared study, and z only through each study's "
                        "maxT (the prior file stores maxT, not z); not full z parity")
    # the code that ran is unchanged since start (the spend log is the only tracked file this run writes)
    clean = not [f for f in _git("diff", "--name-only", commit).splitlines() if f != str(SPEND.relative_to(REPO))]
    same_class = not cpu_arms_on_cpu
    status = ("MEASURED" if same_class else "SMOKE") if par["all"] and clean and not errors else "FAILED"
    summaries, summary_s = {}, {}
    for a, o in arms.items():
        t = time.perf_counter()
        summaries[a] = db.summarize(o, plant)
        summary_s[a] = time.perf_counter() - t
    new = {a: {"summary": summaries[a], "stages": db.stage_times(o) | {"summary_local_s": summary_s[a]},
               "warm_s": o["warm_s"],
               "warm_score_s": o.get("warm_score_s"), "workers": o["workers"], "load_s": o["load_s"],
               "host": o["host"], "container_class": CLASS, "source": "this run",
               "expected_event_fraction": o["setup"]["expected_event_fraction"],
               "studies": [{k: r[k] for k in ("stream", "i", "x_sha", "maxT", "p") if k in r}
                           | ({"maxT": float(np.abs(r["z"]).max())} if "z" in r else {}) for r in o["records"]]}
           for a, o in arms.items()}
    old = {a: {**v, "container_class": CLASS, "source": f"{compare_with} (source_commit {prior['source_commit']})",
               "stages": db.complete(v["stages"], v.get("load_s"), v.get("warm_s"), v.get("warm_score_s"))}
           for a, v in prior["arms"].items()} if prior else {}
    side = {a: (new.get(a) or old[a]) for a in [*ARMS] if a in new or a in old}
    for a, v in side.items():
        v["split"] = db.split(a, v["stages"])
        if a in db.GPU_SCORE and "numba" in side:  # the z check that covers this arm
            if a in old:  # a four-arm prior records it per arm; a three-arm prior in its parity block
                pp = prior["parity"]
                v["z_vs_numba"] = v.get("z_vs_numba") or (
                    {**pp[f"z_{a}_vs_numba"], "basis": "every z, prior run"} if f"z_{a}_vs_numba" in pp else
                    {"max_rel_diff": pp["z_max_rel_diff"], "basis": "every z, prior run"} if a == "gpu" and "z_max_rel_diff" in pp
                    else {"max_rel_diff": None, "basis": "not recorded in the prior run"})
            elif "numba" in new:
                v["z_vs_numba"] = {**par.get(f"z_{a}_vs_numba", {}), "basis": "every z, this run"}
            else:
                v["z_vs_numba"] = {"max_rel_diff": par["against_prior"][a]["numba"].get("maxT_max_rel_diff"),
                                   "basis": "maxT per study against the prior run's numba arm"}
    result = {
        "status": status, "source_commit": commit, "clean_tree": clean, "errors": errors,
        "labels": {"compute": "MEASURED", "outcomes": f"SYNTHETIC (atlas/simulation.py at {db.ATLAS_PIN[0][:7]})",
                   "rates": "ESTIMATE (FWER and power: binomial rates with Wilson 95% intervals)"},
        "headline": db.headline(side) if side else None,
        "same_container_class": same_class,
        "design": {**db.design(), "cohort": "charville", "alpha": db.ALPHA, "hr_per_source_sd": db.HR,
                   "plant_feature": plant, "studies_per_stream": studies, "streams": db.STREAMS, "root_seed": db.ROOT,
                   "outcome_model": f"atlas/simulation.py at {db.ATLAS_PIN[0]}"},
        "parity": par,
        "arms": side,
        "modal": {"app": "preflight-design-batch", "gpu": GPU, "cpu": CORES, "memory_mib": MEMORY_MIB,
                  "started_utc": start.isoformat(), "finished_utc": end.isoformat(), "wall_s": wall,
                  "note": "wall_s includes container start and image pull; stage times are inside the container"},
    }
    path = REPO / "bench" / "results" / f"modal-design-{start:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    entry["actual"] = {"modal_wall_s": round(wall, 1), "est_usd": round(wall / 3600 * sum(rates.values()), 3),
                       "basis": "each container x local wall (an upper bound: includes queueing)", "result": str(path.relative_to(REPO))}
    _log_spend(entry)
    print(status, result["headline"], json.dumps(par), json.dumps({a: v["summary"] for a, v in side.items()}, default=str))
    print(f"Saved {path}")
