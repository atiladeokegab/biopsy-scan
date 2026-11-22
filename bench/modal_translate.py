"""R-17 Translate (translate/runner.py: Step 1 stain panels + Step 2 formats) on Modal, UPMC discovery split only.

    MODAL_PROFILE=fhl-08 timeout 9h uv run --with modal modal run bench/modal_translate.py --seed 0

One CPU-only container (4 cores, 32 GiB). upmc_labels.csv comes from build-modal's preflight-data volume (mounted
read-only); that volume has no upmc_raw_data.zip, so the container downloads it from Zenodo 13179600, the source
scripts/fetch_data.sh uses. The runner is called unchanged: its fail-closed discovery guard checks the labels
sha256 against core/upmc_locked.json and never reads a locked region. Commit first (code_revision is the commit that
ran; the container has no .git). A ceiling (timeout x rate) over --cap-usd is refused. Writes results/upmc-translate.json
and the spend in bench/results/modal-spend-translate.json.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import modal

from bench.modal_design import _git, _tracked

REPO = Path(__file__).resolve().parents[1]
APP_DIR, DATA_DIR, FOREIGN = "/root/preflight", "/tmp/data", "/foreign"
CORES, MEMORY_MIB = 4.0, 32768
RATES = json.loads((REPO / "bench" / "results" / "modal-spend-power.json").read_text())["rates_usd_per_hour"] \
    if modal.is_local() else {}
USD_PER_HOUR = RATES and CORES * RATES["cpu_core"] + MEMORY_MIB / 1024 * RATES["memory_gib"]
ZIP_URL = "https://zenodo.org/records/13179600/files/upmc_raw_data.zip?download=1"
SPEND = REPO / "bench" / "results" / "modal-spend-translate.json"
OUT = REPO / "results" / "upmc-translate.json"

app = modal.App("preflight-translate")
data = modal.Volume.from_name("preflight-data").read_only()
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync(str(REPO))
    .env({"DATA_DIR": DATA_DIR, "PYTHONPATH": APP_DIR})
    .workdir(APP_DIR)
    .add_local_dir(REPO, APP_DIR, ignore=lambda rel: not _tracked(rel))
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@app.function(image=image, cpu=CORES, memory=MEMORY_MIB, volumes={FOREIGN: data}, timeout=8 * 3600)
def translate(seed: int) -> dict:
    import shutil
    import time
    import urllib.request

    from translate.runner import run_translate

    t0 = time.perf_counter()
    d = Path(DATA_DIR)
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(f"{FOREIGN}/upmc_labels.csv", d / "upmc_labels.csv")
    with urllib.request.urlopen(ZIP_URL, timeout=600) as r, open(d / "upmc_raw_data.zip.part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    (d / "upmc_raw_data.zip.part").rename(d / "upmc_raw_data.zip")
    fetch_s = time.perf_counter() - t0
    out = Path("/tmp/upmc-translate.json")
    run_translate(data_dir=d, out_path=out, seed=seed)  # the runner's CLI defaults, as `python -m translate.runner`
    return {"record": out.read_text(), "fetch_s": fetch_s, "in_function_s": time.perf_counter() - t0,
            "inputs_sha256": {n: _sha256(d / n) for n in ("upmc_labels.csv", "upmc_raw_data.zip")},
            "host": {"cpu_count": os.cpu_count(), "affinity_size": len(os.sched_getaffinity(0)),
                     "mem_total_kib": next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal")),
                     "modal_task_id": os.environ.get("MODAL_TASK_ID")}}


@app.local_entrypoint()
def main(seed: int = 0, cap_usd: float = 10.0, timeout_s: int = 8 * 3600):
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes first: code_revision must be the code that runs")
    commit = _git("rev-parse", "HEAD")
    ceiling = round(timeout_s / 3600 * USD_PER_HOUR, 2)
    if ceiling > cap_usd:
        raise SystemExit(f"ceiling ${ceiling} > cap ${cap_usd}: refused")
    start = datetime.now(timezone.utc)
    entry = {"utc": start.strftime("%Y-%m-%dT%H:%MZ"), "run": f"bench/modal_translate.py --seed {seed}",
             "containers": 1, "cores": CORES, "memory_gib": MEMORY_MIB / 1024, "gpu": None,
             "est_usd": ceiling, "basis": f"ceiling: {timeout_s} s timeout x ${USD_PER_HOUR:.3f}/h "
                                          f"({CORES:g} cores x ${RATES['cpu_core']} + {MEMORY_MIB // 1024} GiB x ${RATES['memory_gib']})",
             "source_commit": commit}
    log = {"workspace_profile": os.environ.get("MODAL_PROFILE", "fhl-08"), "cap_usd": cap_usd,
           "rates_usd_per_hour": RATES, "entries": [entry]}
    SPEND.write_text(json.dumps(log, indent=2) + "\n")
    error, out = None, None
    try:
        out = translate.with_options(timeout=timeout_s).remote(seed)
    except Exception as e:  # noqa: BLE001 - a failed run still records its spend
        error = repr(e)[-2000:]
    wall = (datetime.now(timezone.utc) - start).total_seconds()
    entry["actual"] = {"modal_wall_s": round(wall, 1), "est_usd": round(wall / 3600 * USD_PER_HOUR, 3),
                       "basis": f"local call wall x ${USD_PER_HOUR:.3f}/h (upper bound: includes queueing; image build excluded)",
                       "result": str(OUT.relative_to(REPO)) if out else None, "error": error}
    if out:
        entry["actual"]["in_function_s"] = round(out["in_function_s"], 1)
        rec = json.loads(out["record"])
        rec["code_revision"] = commit  # the container has no .git; the clean-tree check above makes this the code that ran
        rec["modal"] = {"app": "preflight-translate", "cores": CORES, "memory_gib": MEMORY_MIB / 1024, "gpu": None,
                        "host": out["host"], "inputs_sha256": out["inputs_sha256"], "fetch_s": out["fetch_s"],
                        "in_function_s": out["in_function_s"], "modal_wall_s": wall, "repeats": 1,
                        "data": "upmc_labels.csv from Modal volume preflight-data; upmc_raw_data.zip from Zenodo 13179600"}
        OUT.write_text(json.dumps(rec, indent=2) + "\n")
    log["actual_total_usd"] = entry["actual"]["est_usd"]
    log["actual_total_basis"] = "container x local wall time; ESTIMATE, not a Modal invoice"
    SPEND.write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(entry, indent=2))
    if error:
        raise SystemExit(f"FAILED: {error}")
