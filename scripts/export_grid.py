"""Turn the atlas's design-point results (#5) into the demo's power grid: demo/data/grid.json (preflight-grid/1).

Run from the repo root (cohort counts need `make data`):
  uv run python -I scripts/export_grid.py RESULTS [--cohort charville] [--hr 2.0] [--measured] [--out PATH]

RESULTS is a .parquet, .csv or .json(l) table with one row per design point: the atlas's run_point
summary (cohort, patients, regions_per_patient (null = all), fov_um (null = full core), eps,
hr_per_source_sd, alpha, plant_feature, power with power_n/_lo/_hi, identification_rate, fwer,
wrong_identification_rate, source_n, source_events, source_regions, optional region_um). Other columns are ignored.
Every row must share one model (alpha, plant and the other MODEL settings) and one power_n, and every
value is validated before anything is written; a bad table exits non-zero and leaves the output alone.

The chart's regions lines are 1, 2, 4 and 8, capped by the most regions any usable patient has;
"all regions" points and the per-cohort source facts go to `meta`, which the page does not read.
source stays SYNTHETIC unless --measured is passed, which is for a full atlas run on the real cohort.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # python -I drops the cwd

import pandas as pd  # noqa: E402

NAMES = {"charville": "Charville", "upmc": "UPMC"}
# The power grid describes study designs, not a discovery search, so it reads both sides of the locked
# UPMC split (core/heldout.py); the only caller allowed to, and every grid records it in meta.
SPLIT = "all"
REGION_LINES = (1, 2, 4, 8)
SOURCE = ("source_n", "source_events", "source_regions")  # region_um is kept in meta when supplied
# One curve = one model: rows that differ on any of these must not be merged (alpha and plant_feature
# are required; the others are checked when the table has them).
MODEL = ("alpha", "plant_feature", "plant_mean", "plant_sd", "shape", "target_event_fraction",
         "censoring_tail_policy", "calibration_n")
NEED = ["cohort", "patients", "regions_per_patient", "fov_um", "eps", "hr_per_source_sd", "alpha", "plant_feature",
        "power", "power_n", "power_lo", "power_hi", "identification_rate",
        "fwer", "wrong_identification_rate", *SOURCE]
METHOD = ("Model-based Monte Carlo: each point is the share of simulated studies, re-imaged from real cells with "
          "planted Weibull outcomes, in which the maxT test rejects. A simulation estimate under the stated model, "
          "not an observed rate, whatever the source label.")

def read_results(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    raise ValueError(f"{path}: expected .parquet, .csv, .json or .jsonl")


def cohort_block(cohort: str) -> tuple[dict, int]:
    """Our cohort as core loads it (usable patients with a verified identity, and their regions),
    and the most regions any of those patients has."""
    from core.cohorts import load_patients, load_regions

    pts = load_patients(load_regions(cohort, split=SPLIT))
    pts = pts[pts.usable]
    n_pat, n_reg = len(pts), int(pts.n_regions.sum())
    block = {"name": NAMES[cohort], "patients": n_pat, "regions": max(1, round(n_reg / n_pat)),
             "regions_mean": round(n_reg / n_pat, 2), "fov_um": "full", "miscall": 0.0}
    return block, int(pts.n_regions.max())


def _missing(v):
    return v is None or pd.isna(v)  # None, NaN or pd.NA, however the table stored "all" / "full core"


def _int(v, what, lo=1):
    """A finite whole number >= lo, as int; anything else is an error (no silent casting of NaN or 2.5)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what}={v!r}: not a number") from None
    if not (math.isfinite(f) and f.is_integer() and f >= lo):
        raise ValueError(f"{what}={v!r}: need a whole number >= {lo}")
    return int(f)


def _rate(v, what):
    f = float(v)
    if not (math.isfinite(f) and 0 <= f <= 1):
        raise ValueError(f"{what}={v!r}: need a rate in 0..1")
    return f


def _plain(v):
    return v.item() if hasattr(v, "item") else v  # numpy scalar -> JSON-able Python value


def _axis(values):
    nums = sorted({v for v in values if v != "full"})
    return nums + (["full"] if "full" in values else [])


def _point(r, where):
    fov = "full" if _missing(r.fov_um) else float(r.fov_um)
    if fov != "full" and not (math.isfinite(fov) and fov > 0):
        raise ValueError(f"{where}: fov_um={r.fov_um!r}: need a positive size or null (full core)")
    p = {"patients": _int(r.patients, f"{where}: patients", 2),
         "regions": "all" if _missing(r.regions_per_patient) else _int(r.regions_per_patient, f"{where}: regions_per_patient"),
         "fov_um": int(fov) if fov != "full" and fov.is_integer() else fov,
         "miscall": _rate(r.eps, f"{where}: eps")}
    for k in ("power", "identification_rate", "power_lo", "power_hi", "fwer", "wrong_identification_rate"):
        p[k] = _rate(getattr(r, k), f"{where}: {k}")
    if not p["identification_rate"] <= p["power"]:
        raise ValueError(f"{where}: identification_rate {p['identification_rate']} above power {p['power']}")
    if not p["power_lo"] <= p["power"] <= p["power_hi"]:
        raise ValueError(f"{where}: power {p['power']} outside its interval [{p['power_lo']}, {p['power_hi']}]")
    return p


def build_grid(df: pd.DataFrame, cohort: dict, max_regions: int, *, cohort_key: str,
               hr: float | None = None, measured: bool = False, source_name: str = "atlas results") -> dict:
    missing = [c for c in NEED if c not in df.columns]
    if missing:
        raise ValueError(f"results lack columns {missing}")
    df = df[df.cohort == cohort_key]
    if df.empty:
        raise ValueError(f"no {cohort_key} rows in the results")
    hrs = sorted(df.hr_per_source_sd.unique())
    if hr is None and len(hrs) == 1:
        hr = hrs[0]
    if hr is None:
        raise ValueError(f"results hold several effects {hrs}; pass --hr")
    df = df[df.hr_per_source_sd == hr]
    if df.empty:
        raise ValueError(f"no {cohort_key} rows at HR {hr}")
    model = {}
    for k in [c for c in MODEL if c in df.columns]:
        values = df[k].drop_duplicates()
        if len(values) != 1:
            raise ValueError(f"rows mix {k} values {sorted(map(str, values))}; export one model at a time")
        model[k] = _plain(values.iloc[0])
    counts = df.power_n.drop_duplicates()
    if len(counts) != 1:
        raise ValueError(f"rows mix power_n {sorted(map(str, counts))}; every point needs the same study count")
    studies = _int(counts.iloc[0], "power_n")
    facts_cols = [*SOURCE, *(["region_um"] if "region_um" in df.columns else [])]
    facts = df[facts_cols].drop_duplicates()
    if len(facts) != 1:
        raise ValueError(f"rows disagree on the source cohort facts {facts_cols}")
    facts = {k: _int(facts.iloc[0][k], k, 0) for k in SOURCE} | (
        {} if "region_um" not in facts_cols or _missing(facts.iloc[0]["region_um"])
        else {"region_um": float(facts.iloc[0]["region_um"])})
    if "region_um" in facts and not (math.isfinite(facts["region_um"]) and facts["region_um"] > 0):
        raise ValueError(f"region_um={facts['region_um']}: need a positive size")
    lines = [k for k in REGION_LINES if k <= max_regions]
    points, all_regions, dropped = [], [], 0
    for i, r in enumerate(df.itertuples()):
        p = _point(r, f"row {i}")
        if p["regions"] == "all":
            all_regions.append(p)
        elif p["regions"] in lines:
            points.append(p)
        else:
            dropped += 1
    if not points:
        raise ValueError(f"no design points on the regions lines {lines}")
    keys = [tuple(p[k] for k in ("patients", "regions", "fov_um", "miscall")) for p in points + all_regions]
    if len(set(keys)) != len(keys):
        raise ValueError("results repeat a design point; keep one row per point")
    note = (f"Model-based Monte Carlo (#5): {len(points)} design points, {studies:,} planted-effect studies each, "
            f"from {source_name}.")
    if not measured:
        note += " Marked SYNTHETIC until the full atlas run on the real cohort."
    axes = {k: _axis([p[k] for p in points]) for k in ("patients", "regions", "fov_um", "miscall")}
    order = lambda p: tuple(axes[k].index(p[k]) for k in axes)  # noqa: E731
    meta = {"method": METHOD, "source": source_name, "split": SPLIT, "model": model, "region_lines": lines,
            "max_regions_per_patient": max_regions, "cohorts": {cohort_key: facts},
            "all_regions_points": all_regions, "points_off_region_lines": dropped}
    return {"schema": "preflight-grid/1", "source": "MEASURED" if measured else "SYNTHETIC", "note": note,
            "studies_per_point": studies, "effect_hr_per_sd": float(hr), "cohort": cohort,
            "axes": axes, "points": sorted(points, key=order), "meta": meta}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--cohort", default="charville", choices=sorted(NAMES))
    ap.add_argument("--hr", type=float)
    ap.add_argument("--measured", action="store_true", help="a full atlas run on the real cohort")
    ap.add_argument("--out", type=Path, default=ROOT / "demo" / "data" / "grid.json")
    a = ap.parse_args(argv)
    try:
        cohort, max_regions = cohort_block(a.cohort)
        grid = build_grid(read_results(a.results), cohort, max_regions, cohort_key=a.cohort, hr=a.hr,
                          measured=a.measured, source_name=a.results.name)
        text = json.dumps(grid, indent=1, allow_nan=False) + "\n"  # validated in full before anything is written
    except ValueError as e:
        sys.exit(f"export_grid: {e}")
    a.out.write_text(text)
    print(f"{a.out}: {len(grid['points'])} points, source {grid['source']}, HR {grid['effect_hr_per_sd']}")


if __name__ == "__main__":
    main()
