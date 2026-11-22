"""Head and neck power curve on Modal: bench/hn_power.py's design, one A100 container per patient count.

    DATA_DIR=<dir with upmc_labels.csv, upmc_raw_data.zip> timeout 2h uv run --with modal modal run gpu/hn_power.py

The local side builds the 34 discovery patients' cells with bench.hn_power.source (never a locked region; the
cache and the patient list are checked against core/upmc_locked.json) and checks the arrays once more before
sending them. Only those arrays reach Modal: no labels file, raw zip, cache or outcome. The containers get
DATA_DIR=/nonexistent, so any hidden data read fails instead of finding something.

Budget: 200 calibration, 100 null and 200 planted studies per point (the #83 design but 200 planted, not 300, to keep
every container's timeout inside the $5 cap). Writes the raw bench/results/modal-hn-power-<utc>.json (MEASURED
compute) and bench/results/hn-power-upmc-discovery.json (ESTIMATE); refuses to overwrite either.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
APP_DIR = "/root/preflight"
ALTERNATIVE = 200
TIMEOUT_S = {50: 500, 100: 650, 200: 1000, 400: 1800, 600: 2700}  # 6,650 s x $2.668/h (A100, 8 cores, 24 GiB) = $4.93

app = modal.App("preflight-hn-power")
image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .env({"PYTHONPATH": APP_DIR, "CUDA_HOME": "/usr/local/cuda", "DATA_DIR": "/nonexistent"}))

if modal.is_local():
    src = Path(tempfile.mkdtemp(prefix="hn-power-"))
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD"], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:  # tracked files at HEAD only
        tar.extractall(src, filter="data")
    image = image.uv_sync(str(src)).workdir(APP_DIR).add_local_dir(src, APP_DIR, ignore=["data", "demo", "results"])


def gpu_info() -> dict:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()
    name, memory, driver = (s.strip() for s in q[0].split(","))
    return {"model": name, "memory": memory, "driver": driver, "modal_task_id": os.environ.get("MODAL_TASK_ID")}


@app.function(image=image, gpu="A100-40GB", timeout=max(TIMEOUT_S.values()), memory=24576, cpu=8.0,
              single_use_containers=True)
def power_point(cells: dict, patients: int) -> dict:
    from bench import hn_power as h

    entered = time.perf_counter()
    out = Path(tempfile.mkdtemp())
    h.run(out / "run", out / "cache", cells, (patients,), ["--alternative", str(ALTERNATIVE), "--backend", "gpu"])
    rows, meta = h.read_run(out / "run")
    return {"rows": rows, "metadata": meta, "in_function_s": time.perf_counter() - entered, "gpu": gpu_info()}


def _jsonable(x):
    for attr in ("item", "tolist"):
        if hasattr(x, attr):
            return getattr(x, attr)()
    return str(x)


@app.local_entrypoint()
def run():
    import pandas as pd

    from bench import hn_power as h

    summary_path, stamp = h.RESULT, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = REPO / "bench" / "results" / f"modal-hn-power-{stamp}.json"
    if summary_path.exists() or raw_path.exists():
        raise SystemExit(f"refusing to overwrite {summary_path.name} or {raw_path.name}")
    cells = h.source(Path(os.environ["DATA_DIR"]))
    case = [cells["patient_names"][i] for i in cells["patient_of_region"]]
    h.check_disjoint(pd.DataFrame({"region_id": cells["region_names"], "case": case}), json.loads(h.LOCK.read_text()))
    assert len(cells["patient_names"]) == 34 and len(cells["region_names"]) == 125, "not the 34 discovery patients"
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0 = time.perf_counter()
    calls = {p: power_point.with_options(timeout=t).spawn(cells, p) for p, t in TIMEOUT_S.items()}
    points, rows, meta = [], [], {}
    for p, call in calls.items():
        try:
            r = call.get()
            rows += r["rows"]
            meta = meta or r["metadata"]
            points.append({"patients": p, **r, "wall_s": time.perf_counter() - t0})
        except Exception as e:  # one lost point is reported, the rest still land
            points.append({"patients": p, "error": repr(e)})
        print(json.dumps({"patients": p, "power": (points[-1].get("rows") or [{}])[0].get("power"),
                          "error": points[-1].get("error"), "wall_s": round(time.perf_counter() - t0, 1)}), flush=True)
    raw_path.write_text(json.dumps({"schema": "preflight-modal-hn-power/1", "status": "MEASURED", "runner": "modal",
                                    "commit": commit, "utc": stamp, "timeouts_s": TIMEOUT_S,
                                    "modal_wall_s_total": time.perf_counter() - t0, "points": points},
                                   indent=1, default=_jsonable) + "\n")
    if rows:
        result = h.summarize(rows, meta)
        result.update(raw=f"bench/results/{raw_path.name}", commit=commit, planted_studies_per_point=ALTERNATIVE)
        summary_path.write_text(json.dumps(result, indent=1, default=_jsonable) + "\n")
        print(result["answer"]["statement"])
    print(raw_path)
    if any("error" in p for p in points):
        raise SystemExit(1)
