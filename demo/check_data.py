"""Check demo/data/{grid,tissue,bench,baseline,claims,patch}.json against the schemas in demo/README.md.

Run: python3 -I demo/check_data.py [data_dir]. bench.json is optional (absent = UNMEASURED); so is
baseline.json (absent = no baseline panel; race.test.mjs still requires the committed one).
Exit 0 when every present file is valid; otherwise it names the first problem.
"""

import json
import math
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


def pos(v):
    return num(v) and math.isfinite(v) and v > 0


def text(v):
    return isinstance(v, str) and v.strip() != ""


def check_timing(r, w, at):
    """One timed or untimed entry; every field is checked on its own, whatever the others say."""
    need(text(r.get("method")), w, f"{at}.method: string")
    need(r.get("label") in LABELS, w, f"{at}.label must be one of {LABELS}")  # a number never travels without
    need(text(r.get("source")), w, f"{at}.source: string, required")          # its label and its source
    for k, ok, what in (("cores", lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0, "positive int"),
                        ("gpu", text, "non-empty string"), ("hardware", text, "non-empty string"),
                        ("startup_s", pos, "positive finite number"), ("note", text, "non-empty string")):
        need(r.get(k) is None or ok(r[k]), w, f"{at}.{k}: {what} or null")
    if r["label"] in ("PENDING", "UNMEASURED"):
        need(r.get("seconds") is None and r.get("startup_s") is None, w, f"{at}: a {r['label']} row has seconds: null, never a number")
        return
    need(pos(r.get("seconds")), w, f"{at}.seconds: positive finite number when {r['label']}")
    need(r.get("hardware") is not None, w, f"{at}.hardware: required when timed")
    need(r.get("cores") is not None or r.get("gpu") is not None, w, f"{at}: cores or gpu required when timed")


def check_baseline(b, w):
    need(b.get("schema") == "preflight-baseline/1", w, "schema must be 'preflight-baseline/1'")
    need(text(b.get("task")), w, "task: string, the one unit of work every row times")
    rows = b.get("rows")
    need(isinstance(rows, list) and rows, w, "rows: non-empty list")
    for i, r in enumerate(rows):
        need(text(r.get("id")) and r["id"] not in [q.get("id") for q in rows[:i]], w, f"rows[{i}].id: unique string")
        need(r.get("role") in ("baseline", "screen"), w, f"rows[{i}].role: 'baseline' or 'screen'")
        check_timing(r, w, f"rows[{i}]")
    if b.get("workflow") is not None:  # different work: shown as a footnote, never as a bar or a ratio
        check_timing(b["workflow"], w, "workflow")
    timed = sum(r["seconds"] is not None for r in rows)
    return f"{len(rows)} rows, {timed} timed"


PATCH_KINDS = ("t5", "tumour", "mac", "mac_pd1", "immune", "other")


def check_patch(p, w):
    need(p.get("schema") == "preflight-patch/1", w, "schema must be 'preflight-patch/1'")
    need(p.get("source") in LABELS, w, f"source must be one of {LABELS}")
    for k in ("feature", "patient", "region", "pd1_rule", "window_rule"):
        need(text(p.get(k)), w, f"{k}: string")
    side = p.get("window_um")
    need(pos(side) and pos(p.get("radius_um")), w, "window_um, radius_um: positive numbers")
    cells = p.get("cells")
    need(isinstance(cells, list) and cells, w, "cells: non-empty list")
    for j, c in enumerate(cells):
        need(num(c.get("x")) and num(c.get("y")) and 0 <= c["x"] <= side and 0 <= c["y"] <= side, w, f"cells[{j}]: x, y within the window")
        need(c.get("k") in PATCH_KINDS, w, f"cells[{j}].k must be one of {PATCH_KINDS}")
    km = p.get("km")
    if km is not None:
        need(km.get("label") in LABELS and text(km.get("note")), w, "km: label and note required")
        for arm in ("high", "low"):
            a = km.get(arm, {})
            need(isinstance(a.get("n"), int) and isinstance(a.get("events"), int), w, f"km.{arm}: n, events ints")
            st = a.get("steps")
            need(isinstance(st, list) and st and st[0] == [0, 1], w, f"km.{arm}.steps: start at [0, 1]")
            need(all(u[0] <= v[0] and u[1] >= v[1] for u, v in zip(st, st[1:])), w, f"km.{arm}.steps: monotone")
    return f"{len(cells)} cells, km {'yes' if km else 'none'}"


def check_claims(c, w):
    need(c.get("schema") == "preflight-claims/1", w, "schema must be 'preflight-claims/1'")
    for k in ("power", "false_alarms", "maxT"):  # every number on screen carries a label and a source
        need(c.get(k, {}).get("label") in LABELS, w, f"{k}.label must be one of {LABELS}")
        need(isinstance(c[k].get("source"), str) and c[k]["source"].strip(), w, f"{k}.source: string, required")
    return "power, false_alarms, maxT"


if __name__ == "__main__":
    for name, check, required in (("grid.json", check_grid, True), ("tissue.json", check_tissue, True),
                                  ("bench.json", check_bench, False), ("baseline.json", check_baseline, False),
                                  ("claims.json", check_claims, False), ("patch.json", check_patch, False)):
        path = DIR / name
        if not path.exists():
            need(not required, path, "missing")
            print(f"{path}: absent (" + "the page hides that panel)" if name in ("baseline.json", "claims.json", "patch.json") else "the page shows UNMEASURED)")
            continue
        print(f"{path}: ok, {check(json.loads(path.read_text()), path)}")
