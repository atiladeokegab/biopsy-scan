"""Check demo/data/{grid,tissue,bench,baseline,claims,biomarker}.json against the schemas in demo/README.md.

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
        if g["source"] == "MEASURED":  # Atlas's simulated grid: every shown rate is its count over its studies, inside its interval
            n = g["studies_per_point"]
            need(isinstance(p.get("power_count"), int) and abs(p["power_count"] / n - p["power"]) < 1e-9, w,
                 f"points[{i}].power: power_count / studies_per_point")
            need(num(p.get("power_lo")) and num(p.get("power_hi")) and p["power_lo"] <= p["power"] <= p["power_hi"], w,
                 f"points[{i}]: power_lo <= power <= power_hi")
            need(num(p.get("fwer")) and 0 <= p["fwer"] <= 1, w, f"points[{i}].fwer: 0..1")
    if g["source"] == "MEASURED":
        note = g.get("note", "")
        need("SYNTHETIC" in note and "legacy R4" in note and "not an R15" in note, w,
             "note: a MEASURED grid says its outcomes are SYNTHETIC, its test is legacy R4 and it is not R15")
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
                        ("startup_s", pos, "positive finite number"), ("note", text, "non-empty string"),
                        ("calls_s", lambda v: num(v) and math.isfinite(v) and v >= 0, "non-negative finite number")):
        need(r.get(k) is None or ok(r[k]), w, f"{at}.{k}: {what} or null")
    if r["label"] in ("PENDING", "UNMEASURED"):
        need(r.get("seconds") is None and r.get("startup_s") is None, w, f"{at}: a {r['label']} row has seconds: null, never a number")
        return
    need(pos(r.get("seconds")), w, f"{at}.seconds: positive finite number when {r['label']}")
    need(r.get("hardware") is not None, w, f"{at}.hardware: required when timed")


def check_baseline(b, w):
    need(b.get("schema") == "preflight-baseline/1", w, "schema must be 'preflight-baseline/1'")
    need(text(b.get("task")), w, "task: string, the one unit of work every row times")
    rows = b.get("rows")
    need(isinstance(rows, list) and rows, w, "rows: non-empty list")
    for i, r in enumerate(rows):
        need(text(r.get("id")) and r["id"] not in [q.get("id") for q in rows[:i]], w, f"rows[{i}].id: unique string")
        need(r.get("role") in ("baseline", "screen"), w, f"rows[{i}].role: 'baseline' or 'screen'")
        check_timing(r, w, f"rows[{i}]")
    for g, grp in enumerate(b.get("interim") or []):  # measured pairs shown until the design point is timed
        need(text(grp.get("title")) and text(grp.get("unit")), w, f"interim[{g}]: title, unit")
        need(grp.get("same_scope") in (None, True, False), w, f"interim[{g}].same_scope: boolean")
        need(isinstance(grp.get("rows"), list) and grp["rows"], w, f"interim[{g}].rows: non-empty list")
        for i, r in enumerate(grp["rows"]):
            check_timing(r, w, f"interim[{g}].rows[{i}]")
    if b.get("workflow") is not None:  # different work: shown as a footnote, never as a bar or a ratio
        check_timing(b["workflow"], w, "workflow")
    timed = sum(r["seconds"] is not None for r in rows)
    sc = b.get("scope") or {}
    need(not timed or all(text(sc.get(k)) for k in ("comparison", "calibration")), w,
         "scope.comparison and scope.calibration: required once a race row is timed (WORKFLOW/EXPLORATORY)")
    return f"{len(rows)} rows, {timed} timed"


def check_biomarker(b, w):
    """build-8's file (#82): two real regions, one per KM arm, and the KM curves; exploratory, never a p-value."""
    need(b.get("schema") == "preflight-biomarker/1", w, "schema must be 'preflight-biomarker/1'")
    need(b.get("source") in LABELS and b.get("exploratory") is True, w, "source label, and exploratory: true")
    need(text(b.get("label")) and "no p-value" in b["label"], w, "label must say: no p-value")
    need(text(b.get("note")), w, "note: string")
    km = b.get("km") or {}
    need(km.get("p_value") is None, w, "km.p_value must be null: the cutoff was chosen on the same cohort")
    need(sorted(a.get("name", "") for a in km.get("arms", [])) == ["high", "low"], w, "km.arms: high and low")
    for a in km["arms"]:
        c = a.get("curve")
        need(isinstance(a.get("n"), int) and isinstance(a.get("events"), int), w, f"km {a['name']}: n, events ints")
        need(isinstance(c, list) and c and c[0] == [0.0, 1.0], w, f"km {a['name']}.curve: starts at [0, 1]")
        need(all(u[0] <= v[0] and u[1] >= v[1] for u, v in zip(c, c[1:])), w, f"km {a['name']}.curve: monotone")
    regions = b.get("regions") or []
    need(sorted(r.get("arm", "") for r in regions) == ["high", "low"], w, "regions: one per arm")
    for r in regions:
        cells = r.get("cells", {})
        n = len(cells.get("x", []))
        need(n > 0 and all(len(cells.get(k, [])) == n for k in ("y", "type", "pd1_pos", "near_tumour")), w,
             f"region {r['arm']}: columnar cells of equal length")
        need(pos(r.get("width_um")) and pos(r.get("height_um")), w, f"region {r['arm']}: width_um, height_um")
        need(r.get("type_names", [])[:2] == ["Tumor 5", "Macrophage"], w, f"region {r['arm']}: type 0 Tumor 5, 1 Macrophage")
    arms = " / ".join(f"{a['events']}/{a['n']}" for a in km["arms"])
    return f"{len(regions)} regions, km {arms}"


CLAIM_NUMBERS = {"power": ("patients", "sim", "lo", "hi", "n", "formula"), "false_alarms": ("hits", "n", "designs", "target"),
                 "maxT": ("z", "p", "tests")}


def check_claims(c, w):
    """Every number the page shows from claims.json is finite, labelled, and traced to a committed file."""
    need(c.get("schema") == "preflight-claims/1", w, "schema must be 'preflight-claims/1'")
    root = Path(__file__).resolve().parents[1]
    for k, fields in CLAIM_NUMBERS.items():
        b = c.get(k)
        need(isinstance(b, dict), w, f"{k}: object")
        need(b.get("label") in LABELS, w, f"{k}.label must be one of {LABELS}")
        need(text(b.get("source")) and (root / b["source"]).is_file(), w, f"{k}.source: a committed file, required")
        for f in fields:
            need(num(b.get(f)) and math.isfinite(b[f]), w, f"{k}.{f}: finite number")
    # The same rules as claims.js claimTexts, which renders them (tests/test_collect_race.py checks they agree).
    pw, fa, mt = c["power"], c["false_alarms"], c["maxT"]
    count = lambda v, lo: isinstance(v, int) and not isinstance(v, bool) and v >= lo
    rate = lambda v: 0 <= v <= 1
    need(count(pw["patients"], 1) and count(pw["n"], 1), w, "power.patients, power.n: positive integers")
    need(all(rate(pw[f]) for f in ("sim", "lo", "hi", "formula")) and pw["lo"] <= pw["sim"] <= pw["hi"], w,
         "power: sim, lo, hi, formula in [0, 1] and lo <= sim <= hi")
    need(count(fa["n"], 1) and count(fa["designs"], 1) and count(fa["hits"], 0) and fa["hits"] <= fa["n"], w,
         "false_alarms: n, designs positive integers; hits an integer in [0, n]")
    need(rate(fa["target"]), w, "false_alarms.target in [0, 1]")
    need(0 < mt["p"] <= 1 and count(mt["tests"], 1), w, "maxT: p in (0, 1], tests a positive integer")
    need(pw.get("formula_label") == "ANALYTIC", w, "power.formula_label: ANALYTIC")
    need(text(pw.get("model")) and "not the PD1" in pw["model"], w, "power.model: says it is not PD1's power")
    return "power, false_alarms, maxT"


if __name__ == "__main__":
    for name, check, required in (("grid.json", check_grid, True), ("tissue.json", check_tissue, True),
                                  ("bench.json", check_bench, False), ("baseline.json", check_baseline, False),
                                  ("claims.json", check_claims, False), ("biomarker.json", check_biomarker, False)):
        path = DIR / name
        if not path.exists():
            need(not required, path, "missing")
            print(f"{path}: absent (" + "the page hides that panel)" if name in ("baseline.json", "claims.json", "biomarker.json") else "the page shows UNMEASURED)")
            continue
        print(f"{path}: ok, {check(json.loads(path.read_text()), path)}")
