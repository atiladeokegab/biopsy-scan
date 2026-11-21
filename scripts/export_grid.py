"""Turn the atlas's design-point results (#5) into the demo's power grid: demo/data/grid.json (preflight-grid/1).

Run from the repo root (cohort counts need `make data`):
  uv run python -I scripts/export_grid.py RESULTS [--cohort charville] [--hr 2.0] [--measured] [--out PATH]

RESULTS is a .parquet, .csv or .json(l) table with one row per design point: the atlas's run_point
summary (cohort, patients, regions_per_patient (null = all), fov_um (null = full core), eps,
hr_per_source_sd, power with power_n/_lo/_hi, identification_rate, source_n, source_events,
source_regions, region_um). Other columns are ignored.

The chart's regions lines are 1, 2, 4 and 8, capped by the most regions any usable patient has;
"all regions" points and the per-cohort source facts go to `meta`, which the page does not read.
source stays SYNTHETIC unless --measured is passed, which is for a full atlas run on the real cohort.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # python -I drops the cwd

import pandas as pd  # noqa: E402

NAMES = {"charville": "Charville", "upmc": "UPMC"}
REGION_LINES = (1, 2, 4, 8)
SOURCE = ("source_n", "source_events", "source_regions", "region_um")
NEED = ["cohort", "patients", "regions_per_patient", "fov_um", "eps", "hr_per_source_sd",
        "power", "power_n", "power_lo", "power_hi", "identification_rate", *SOURCE]


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

    pts = load_patients(load_regions(cohort))
    pts = pts[pts.usable]
    n_pat, n_reg = len(pts), int(pts.n_regions.sum())
    block = {"name": NAMES[cohort], "patients": n_pat, "regions": max(1, round(n_reg / n_pat)),
             "regions_mean": round(n_reg / n_pat, 2), "fov_um": "full", "miscall": 0.0}
    return block, int(pts.n_regions.max())


def _missing(v):
    return v is None or pd.isna(v)  # None, NaN or pd.NA, however the table stored "all" / "full core"


def _num(v):
    return int(v) if float(v).is_integer() else float(v)


def _axis(values):
    nums = sorted({v for v in values if v != "full"})
    return nums + (["full"] if "full" in values else [])


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
    facts = df[list(SOURCE)].drop_duplicates()
    if len(facts) != 1:
        raise ValueError(f"rows disagree on the source cohort facts {SOURCE}")
    lines = [k for k in REGION_LINES if k <= max_regions]
    points, all_regions, dropped = [], [], 0
    for r in df.itertuples():
        p = {"patients": int(r.patients), "regions": None if _missing(r.regions_per_patient) else int(r.regions_per_patient),
             "fov_um": "full" if _missing(r.fov_um) else _num(r.fov_um), "miscall": float(r.eps),
             "power": float(r.power), "identification_rate": float(r.identification_rate),
             "power_lo": float(r.power_lo), "power_hi": float(r.power_hi)}
        if p["regions"] is None:
            all_regions.append({**p, "regions": "all"})
        elif p["regions"] in lines:
            points.append(p)
        else:
            dropped += 1
    if not points:
        raise ValueError(f"no design points on the regions lines {lines}")
    keys = [tuple(p[k] for k in ("patients", "regions", "fov_um", "miscall")) for p in points + all_regions]
    if len(set(keys)) != len(keys):
        raise ValueError("results repeat a design point; keep one row per point")
    studies = int(df.power_n.min())  # shortcut: one count for the page; points with more studies are not shown as such
    note = (f"Atlas (#5): {len(points)} design points, at least {studies:,} planted-effect studies each, "
            f"from {source_name}.")
    if not measured:
        note += " Marked SYNTHETIC until the full atlas run on the real cohort."
    axes = {k: _axis([p[k] for p in points]) for k in ("patients", "regions", "fov_um", "miscall")}
    order = lambda p: tuple(axes[k].index(p[k]) for k in axes)  # noqa: E731
    meta = {"source": source_name, "region_lines": lines, "max_regions_per_patient": max_regions,
            "cohorts": {cohort_key: {k: _num(v) for k, v in facts.iloc[0].items()}},
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
    cohort, max_regions = cohort_block(a.cohort)
    grid = build_grid(read_results(a.results), cohort, max_regions, cohort_key=a.cohort, hr=a.hr,
                      measured=a.measured, source_name=a.results.name)
    a.out.write_text(json.dumps(grid, indent=1) + "\n")
    print(f"{a.out}: {len(grid['points'])} points, source {grid['source']}, HR {grid['effect_hr_per_sd']}")


if __name__ == "__main__":
    main()
