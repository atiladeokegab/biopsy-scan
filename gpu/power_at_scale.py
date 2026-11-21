"""How many patients for 80% power? The vs-schoenfeld-300 simulation, extended to larger cohorts on Modal GPUs.

    timeout 3h uv run --with modal modal run gpu/power_at_scale.py --patients 400,500,600

One fresh GPU container per patient count, all in parallel. Same code as bench/results/vs-schoenfeld-300.json:
the repo at PIN (core, reimage, stats) plus the frozen atlas snapshot in bench/atlas_snapshot/ whose per-file sha256 that file records
(the run refuses a snapshot that does not match), same design (Charville, 2 regions, 400 um field, 10%
mis-calls, HR 2 per source SD), same budget (200 calibration, 100 null, 300 planted studies per point), same
seed. design_id, and so every study seed, leaves out the backend, so the GPU re-images the same studies.

Each run writes bench/results/modal-power-<gpu>-<utc>.json (raw, MEASURED); bench/power_at_scale.py merges them.
Mounts build-modal's preflight-data volume read-only and copies only the raw Charville inputs out of it.
"""
from __future__ import annotations

import hashlib
import io
import shutil
import json
import os
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

PIN = "1f1bc7457eaf98e5cff600c0e17001d7e2fc44bf"  # 32-vs-schoenfeld head: the core/reimage/stats the 300 run used
REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "bench" / "results" / "vs-schoenfeld-300.json"
SNAPSHOT = REPO / "bench" / "atlas_snapshot" / "atlas"
APP_DIR = "/root/preflight"
DATA_DIR = "/tmp/data"  # private to each container: the pinned core builds its own cells cache here
DESIGN = ["--cohorts", "charville", "--regions", "2", "--fov-um", "400", "--eps", "0.1", "--hr", "2.0",
          "--seed", "20261121", "--calibration", "200", "--null", "100", "--alternative", "300", "--nested", "0",
          "--permutations", "999", "--batch", "8"]

FOREIGN = "/foreign"  # build-modal's preflight-data volume, mounted read-only

app = modal.App("preflight-power")
data = modal.Volume.from_name("preflight-data").read_only()
image = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
         .env({"DATA_DIR": DATA_DIR, "PYTHONPATH": APP_DIR, "CUDA_HOME": "/usr/local/cuda"}))


def stage(dest: Path) -> Path:
    """PIN's tracked files into a fresh dest, verified; nothing else in it."""
    archive = subprocess.run(["git", "-C", str(REPO), "archive", PIN], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return verify(dest)


def verify(dest: Path) -> Path:
    """Every file in dest is one of PIN's tracked files, byte-identical by its git blob hash, and none is missing."""
    tree = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "-z", PIN], capture_output=True, text=True,
                          check=True).stdout.split("\0")
    want = {e.split("\t", 1)[1]: e.split()[2] for e in tree if e and e.split()[0] in ("100644", "100755")}
    got = {str(f.relative_to(dest)): f for f in dest.rglob("*") if f.is_file() or f.is_symlink()}
    if set(got) != set(want):
        raise RuntimeError(f"staged files differ from {PIN[:12]}: {sorted(set(got) ^ set(want))[:5]}")
    for name, f in got.items():
        b = f.read_bytes()
        if hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() != want[name]:
            raise RuntimeError(f"staged {name} does not match {PIN[:12]}")
    return dest


def snapshot_sha256() -> dict:
    """The 7 full sha256 lines in bench/atlas_snapshot/README.md, checked against the #53 record's 12-char prefixes."""
    lines = [ln.split() for ln in (SNAPSHOT.parent / "README.md").read_text().splitlines()
             if len(ln.split()) == 2 and len(ln.split()[0]) == 64 and ln.split()[1].startswith("atlas/")]
    full = {name: digest for digest, name in lines}
    record = json.loads(RECORD.read_text())["simulated"]["atlas_code"]["sha256_12"]
    if {k: v[:12] for k, v in full.items()} != record:
        raise RuntimeError("bench/atlas_snapshot/README.md does not match the #53 record")
    return full


def verify_snapshot(dest: Path, full: dict | None = None) -> Path:
    """dest holds exactly the snapshot's files, each matching its full sha256."""
    full = full or snapshot_sha256()
    got = {f"atlas/{f.name}": hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.iterdir()
           if f.name != "__pycache__"}
    if got != full:
        raise RuntimeError(f"atlas snapshot in {dest} differs from the committed #53 snapshot")
    return dest


def stage_snapshot(dest: Path) -> Path:
    """The snapshot as committed at HEAD (never the working tree), verified."""
    rel = SNAPSHOT.relative_to(REPO).as_posix()
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD", rel], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return verify_snapshot(dest / rel)


if modal.is_local():
    src = stage(Path(tempfile.mkdtemp(prefix="preflight-power-")))
    snap = stage_snapshot(Path(tempfile.mkdtemp(prefix="preflight-power-snap-")))
    # only tracked files: the pinned commit's verified archive and HEAD's committed, verified snapshot
    image = image.uv_sync(str(src)).workdir(APP_DIR).add_local_dir(src, APP_DIR, ignore=["data", "demo", "results"])
    image = image.add_local_dir(snap, f"{APP_DIR}/atlas")


def gpu_info() -> dict:
    """nvidia-smi's model, memory and driver for the device every number belongs to (as gpu/modal_app.py)."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()
    name, memory, driver = (s.strip() for s in q[0].split(","))
    return {"model": name, "memory": memory, "driver": driver, "modal_task_id": os.environ.get("MODAL_TASK_ID")}


@app.function(image=image, volumes={FOREIGN: data}, gpu="A100-40GB", timeout=4 * 3600, memory=32768, cpu=8.0,
              single_use_containers=True)
def power_point(patients: int, backend: str, snapshot_sha256: dict) -> dict:
    """One design point, start to finish, in this container: calibrate, null, planted; the atlas's own summary."""
    import pandas as pd

    entered = time.perf_counter()
    verify_snapshot(Path(APP_DIR, "atlas"), snapshot_sha256)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    for name in ("charville_labels.csv", "charville_raw_data.zip"):  # raw inputs only; never the volume's caches
        shutil.copy(Path(FOREIGN, name), Path(DATA_DIR, name))
    from atlas.__main__ import main

    out = Path(tempfile.mkdtemp())
    main([*DESIGN, "--patients", str(patients), "--backend", backend, "--out", str(out / "run"),
          "--cache", str(out / "cache")])
    summary = pd.read_parquet(out / "run" / "charville" / "summary.parquet").iloc[0].to_dict()
    meta = json.loads((out / "run" / "charville" / "metadata.json").read_text())
    return {"summary": summary, "metadata": {k: meta.get(k) for k in ("source_hash", "grammar_hash", "budget", "seed",
                                                                       "source_n", "hr_per_source_sd", "backend")},
            "in_function_s": time.perf_counter() - entered, "gpu": gpu_info()}


def _jsonable(x):
    for attr in ("item", "tolist"):
        if hasattr(x, attr):
            return getattr(x, attr)()
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")


def preflight(patients: str, backend: str) -> list[int]:
    """Refuse, before any paid dispatch, what would fail or repeat a seeded job, and CPU work billed as GPU."""
    if backend != "gpu":
        raise SystemExit(f"backend {backend!r}: this launcher records the GPU as each point's hardware, so gpu only")
    try:
        counts = [int(p) for p in patients.split(",")]
    except ValueError:
        raise SystemExit(f"--patients must be comma-separated integers, got {patients!r}") from None
    if min(counts) < 2:
        raise SystemExit("each patient count must be at least 2")
    if len(set(counts)) != len(counts):
        raise SystemExit("a patient count appears twice: the same seeds would run twice")
    return counts


@app.local_entrypoint()
def run(patients: str, gpu: str = "A100-40GB", backend: str = "gpu"):
    record = snapshot_sha256()
    counts = preflight(patients, backend)
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    launched = {"commit": git("rev-parse", "HEAD"), "pin": PIN,
                "launched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    fn = power_point.with_options(gpu=gpu)
    t0 = time.perf_counter()
    walls, points = {}, []
    for p, result in zip(counts, fn.map(counts, kwargs={"backend": backend, "snapshot_sha256": record},
                                        return_exceptions=True, order_outputs=True)):
        if isinstance(result, Exception):
            points.append({"patients": p, "error": repr(result)})
        else:
            points.append({"patients": p, **result})
        walls[p] = time.perf_counter() - t0  # ordered outputs: an upper bound per point; in_function_s is exact
        print(json.dumps({"patients": p, "power": points[-1].get("summary", {}).get("power"),
                          "error": points[-1].get("error"), "wall_s": round(walls[p], 1)}), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPO / "bench" / "results" / f"modal-power-{gpu.lower()}-{stamp}.json"
    path.write_text(json.dumps({"schema": "preflight-modal-power/1", "status": "MEASURED", "runner": "modal",
                                "gpu_requested": gpu, "backend": backend, **launched, "utc": stamp,
                                "modal_wall_s_total": time.perf_counter() - t0, "points": points},
                               indent=1, default=_jsonable) + "\n")
    print(path)
    if any("error" in p for p in points):
        raise SystemExit(1)
