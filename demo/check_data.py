"""Check demo/data/{grid,tissue,bench,baseline}.json against the schemas in demo/README.md.

Run: python3 -I demo/check_data.py [data_dir]. bench.json is optional (absent = UNMEASURED).
Exit 0 when every present file is valid; otherwise it names the first problem.
"""

import json
import sys
from pathlib import Path

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
TYPES = ("tumour", "stroma", "immune")


def need(cond, where, msg):
    if not cond:
        sys.exit(f"{where}: {msg}")


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def fov_ok(v):
    return v == "full" or (num(v) and v > 0)


def check_grid(g, w):
    need(g.get("schema") == "preflight-grid/1", w, "schema must be 'preflight-grid/1'")
    need(g.get("source") in ("SYNTHETIC", "MEASURED"), w, "source must be SYNTHETIC or MEASURED")
    need(isinstance(g.get("studies_per_point"), int) and g["studies_per_point"] > 0, w, "studies_per_point: positive int")
    need(num(g.get("effect_hr_per_sd")), w, "effect_hr_per_sd: number")
    c = g.get("cohort", {})
    need(isinstance(c.get("name"), str), w, "cohort.name: string")
    need(isinstance(c.get("patients"), int) and c["patients"] > 0, w, "cohort.patients: positive int")
    need(isinstance(c.get("regions"), int) and c["regions"] > 0, w, "cohort.regions: positive int")
    need(fov_ok(c.get("fov_um")), w, "cohort.fov_um: positive number or 'full'")
    need("regions_mean" not in c or (num(c["regions_mean"]) and c["regions_mean"] > 0), w, "cohort.regions_mean: positive number if given")
    need(num(c.get("miscall")) and 0 <= c["miscall"] <= 1, w, "cohort.miscall: 0..1")
    a = g.get("axes", {})
    for k in ("patients", "regions", "fov_um", "miscall"):
        need(isinstance(a.get(k), list) and a[k], w, f"axes.{k}: non-empty list")
    need(all(fov_ok(v) for v in a["fov_um"]), w, "axes.fov_um: numbers or 'full'")
    pts = g.get("points")
    need(isinstance(pts, list) and pts, w, "points: non-empty list")
    seen = set()
    for i, p in enumerate(pts):
        key = tuple(p.get(k) for k in ("patients", "regions", "fov_um", "miscall"))
        for k, v in zip(("patients", "regions", "fov_um", "miscall"), key):
            need(v in a[k], w, f"points[{i}].{k}={v!r} is not on axes.{k}")
        need(key not in seen, w, f"points[{i}] duplicates design point {key}")
        seen.add(key)
        for k in ("power", "identification_rate"):
            need(num(p.get(k)) and 0 <= p[k] <= 1, w, f"points[{i}].{k}: 0..1")
        need(p["identification_rate"] <= p["power"] + 1e-9, w, f"points[{i}]: identification_rate above power")
    return f"{len(pts)} points, source {g['source']}"


def check_tissue(t, w):
    need(t.get("schema") == "preflight-tissue/1", w, "schema must be 'preflight-tissue/1'")
    need(t.get("source") in ("SYNTHETIC", "MEASURED"), w, "source must be SYNTHETIC or MEASURED")
    side = t.get("region_um")
    need(num(side) and side > 0, w, "region_um: positive number")
    ps = t.get("patients")
    need(isinstance(ps, list) and ps, w, "patients: non-empty list")
    for i, p in enumerate(ps):
        need(isinstance(p.get("id"), str), w, f"patients[{i}].id: string")
        cells = p.get("cells")
        need(isinstance(cells, list) and cells, w, f"patients[{i}].cells: non-empty list")
        for j, c in enumerate(cells):
            need(num(c.get("x")) and num(c.get("y")) and 0 <= c["x"] <= side and 0 <= c["y"] <= side,
                 w, f"patients[{i}].cells[{j}]: x, y must be within 0..region_um")
            need(c.get("type") in TYPES, w, f"patients[{i}].cells[{j}].type must be one of {TYPES}")
    return f"{len(ps)} patients, {sum(len(p['cells']) for p in ps)} cells, source {t['source']}"


def check_bench(b, w):
    need(b.get("schema") == "preflight-bench/1", w, "schema must be 'preflight-bench/1'")
    need(b.get("status") in ("UNMEASURED", "MEASURED"), w, "status must be UNMEASURED or MEASURED")
    need(isinstance(b.get("studies"), int) and b["studies"] > 0, w, "studies: positive int")
    for k in ("cpu", "gpu"):  # a speed-up means nothing without the hardware it was measured on
        need(isinstance(b.get(k), str) and b[k].strip() != "", w, f"{k}: hardware model string, required")
    if b["status"] == "MEASURED":
        for k in ("cpu_seconds", "gpu_seconds"):
            need(num(b.get(k)) and b[k] > 0, w, f"{k}: positive number when MEASURED")
    return f"status {b['status']}"


LABELS = ("MEASURED", "ESTIMATE", "SYNTHETIC", "UNMEASURED", "PENDING")


def check_baseline(b, w):
    need(b.get("schema") == "preflight-baseline/1", w, "schema must be 'preflight-baseline/1'")
    need(isinstance(b.get("task"), str) and b["task"].strip(), w, "task: string, the job every row times")
    rows = b.get("rows")
    need(isinstance(rows, list) and rows, w, "rows: non-empty list")
    for i, r in enumerate(rows):
        at = f"rows[{i}]"
        need(isinstance(r.get("method"), str) and r["method"].strip(), w, f"{at}.method: string")
        need(r.get("role") in ("baseline", "screen"), w, f"{at}.role: 'baseline' or 'screen'")
        # A number never travels without its label and its source.
        need(r.get("label") in LABELS, w, f"{at}.label must be one of {LABELS}")
        need(isinstance(r.get("source"), str) and r["source"].strip(), w, f"{at}.source: string, required")
        s = r.get("seconds")
        if r["label"] in ("PENDING", "UNMEASURED"):
            need(s is None, w, f"{at}: a {r['label']} row has seconds: null, never a number")
            continue
        need(num(s) and s > 0, w, f"{at}.seconds: positive number when {r['label']}")
        need(isinstance(r.get("hardware"), str) and r["hardware"].strip(), w, f"{at}.hardware: string when timed")
        cores, gpu = r.get("cores"), r.get("gpu")
        need((isinstance(cores, int) and cores > 0) or (isinstance(gpu, str) and gpu.strip()), w,
             f"{at}: cores (positive int) or gpu (string) when timed")
    timed = sum(r["seconds"] is not None for r in rows)
    return f"{len(rows)} rows, {timed} timed"


for name, check, required in (("grid.json", check_grid, True), ("tissue.json", check_tissue, True),
                              ("bench.json", check_bench, False), ("baseline.json", check_baseline, True)):
    path = DIR / name
    if not path.exists():
        need(not required, path, "missing")
        print(f"{path}: absent (the page shows UNMEASURED)")
        continue
    print(f"{path}: ok, {check(json.loads(path.read_text()), path)}")
