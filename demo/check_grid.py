"""Check demo/data/grid.json (or a given path) against the preflight-grid/1 schema in demo/README.md.

Run: python3 -I demo/check_grid.py [path]. Exit 0 when valid; otherwise it names the first problem.
"""

import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "grid.json"
g = json.loads(path.read_text())


def need(cond, msg):
    if not cond:
        sys.exit(f"{path}: {msg}")


def fov_ok(v):
    return v == "full" or (isinstance(v, (int, float)) and v > 0)


need(g.get("schema") == "preflight-grid/1", "schema must be 'preflight-grid/1'")
need(g.get("source") in ("SYNTHETIC", "MEASURED"), "source must be SYNTHETIC or MEASURED")
need(isinstance(g.get("studies_per_point"), int) and g["studies_per_point"] > 0, "studies_per_point: positive int")
need(isinstance(g.get("effect_hr_per_sd"), (int, float)), "effect_hr_per_sd: number")

c = g.get("cohort", {})
need(isinstance(c.get("name"), str), "cohort.name: string")
need(isinstance(c.get("patients"), int) and c["patients"] > 0, "cohort.patients: positive int")
need(isinstance(c.get("regions"), int) and c["regions"] > 0, "cohort.regions: positive int")
need(fov_ok(c.get("fov_um")), "cohort.fov_um: positive number or 'full'")
need(isinstance(c.get("miscall"), (int, float)) and 0 <= c["miscall"] <= 1, "cohort.miscall: 0..1")

h = g.get("headline", {})
need(h.get("status") in ("UNMEASURED", "MEASURED"), "headline.status must be UNMEASURED or MEASURED")
if h["status"] == "MEASURED":
    for k in ("gpu_seconds", "cpu_seconds"):
        need(isinstance(h.get(k), (int, float)) and h[k] > 0, f"headline.{k}: positive number when MEASURED")

a = g.get("axes", {})
for k in ("patients", "regions", "fov_um", "miscall"):
    need(isinstance(a.get(k), list) and a[k], f"axes.{k}: non-empty list")
need(all(fov_ok(v) for v in a["fov_um"]), "axes.fov_um: numbers or 'full'")

pts = g.get("points")
need(isinstance(pts, list) and pts, "points: non-empty list")
seen = set()
for i, p in enumerate(pts):
    key = tuple(p.get(k) for k in ("patients", "regions", "fov_um", "miscall"))
    for k, v in zip(("patients", "regions", "fov_um", "miscall"), key):
        need(v in a[k], f"points[{i}].{k}={v!r} is not on axes.{k}")
    need(key not in seen, f"points[{i}] duplicates design point {key}")
    seen.add(key)
    for k in ("power", "identification_rate"):
        need(isinstance(p.get(k), (int, float)) and 0 <= p[k] <= 1, f"points[{i}].{k}: 0..1")
    need(p["identification_rate"] <= p["power"] + 1e-9, f"points[{i}]: identification_rate above power")

print(f"{path}: ok, {len(pts)} points, source {g['source']}, headline {h['status']}")
