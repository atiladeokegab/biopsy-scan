"""Turn the atlas's design-point results (#5) into the demo's power grid: demo/data/grid.json (preflight-grid/1).

Run from the repo root (cohort counts need `make data`):
  uv run python -I scripts/export_grid.py RESULTS [--cohort charville] [--hr 2.0] [--measured] [--out PATH]

RESULTS is a .parquet, .csv or .json(l) table with one row per design point, in the shape of
atlas.runner.run_point's summary: cohort, patients, regions_per_patient, fov_um (null = full field),
eps, hr_per_source_sd, power, power_n, identification_rate.
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
NEED = ["cohort", "patients", "regions_per_patient", "fov_um", "eps", "hr_per_source_sd",
        "power", "power_n", "identification_rate"]


def read_results(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    raise ValueError(f"{path}: expected .parquet, .csv, .json or .jsonl")


def cohort_block(cohort: str) -> dict:
    """Our cohort as core loads it: usable patients with a verified identity, and their regions."""
    from core.cohorts import load_patients, load_regions

    reg = load_regions(cohort)
    pts = load_patients(reg)
    usable = set(pts.patient_id[pts.usable])
    n_pat, n_reg = len(usable), int(reg.patient_id.isin(usable).sum())
    return {"name": NAMES[cohort], "patients": n_pat, "regions": max(1, round(n_reg / n_pat)),
            "regions_mean": round(n_reg / n_pat, 2), "fov_um": "full", "miscall": 0.0}


def _fov(v):
    return "full" if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _axis(values):
    nums = sorted({v for v in values if v != "full"})
    return [int(v) if float(v).is_integer() else v for v in nums] + (["full"] if "full" in values else [])


def build_grid(df: pd.DataFrame, cohort: dict, *, cohort_key: str, hr: float | None = None,
               measured: bool = False, source_name: str = "atlas results") -> dict:
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
    points = []
    for r in df.itertuples():
        fov = _fov(r.fov_um)
        points.append({"patients": int(r.patients), "regions": int(r.regions_per_patient),
                       "fov_um": int(fov) if fov != "full" and fov.is_integer() else fov,
                       "miscall": float(r.eps), "power": float(r.power),
                       "identification_rate": float(r.identification_rate)})
    keys = [tuple(p[k] for k in ("patients", "regions", "fov_um", "miscall")) for p in points]
    if len(set(keys)) != len(keys):
        raise ValueError("results repeat a design point; keep one row per point")
    studies = int(df.power_n.min())  # shortcut: one count for the page; points with more studies are not shown as such
    note = (f"Atlas (#5): {len(points)} design points, at least {studies:,} planted-effect studies each, "
            f"from {source_name}.")
    if not measured:
        note += " Marked SYNTHETIC until the full atlas run on the real cohort."
    axes = {k: _axis([p[k] for p in points]) for k in ("patients", "regions", "fov_um", "miscall")}
    return {"schema": "preflight-grid/1", "source": "MEASURED" if measured else "SYNTHETIC", "note": note,
            "studies_per_point": studies, "effect_hr_per_sd": float(hr), "cohort": cohort,
            "axes": axes, "points": sorted(points, key=lambda p: tuple(axes[k].index(p[k]) for k in axes))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--cohort", default="charville", choices=sorted(NAMES))
    ap.add_argument("--hr", type=float)
    ap.add_argument("--measured", action="store_true", help="a full atlas run on the real cohort")
    ap.add_argument("--out", type=Path, default=ROOT / "demo" / "data" / "grid.json")
    a = ap.parse_args(argv)
    grid = build_grid(read_results(a.results), cohort_block(a.cohort), cohort_key=a.cohort, hr=a.hr,
                      measured=a.measured, source_name=a.results.name)
    a.out.write_text(json.dumps(grid, indent=1) + "\n")
    print(f"{a.out}: {len(grid['points'])} points, source {grid['source']}, HR {grid['effect_hr_per_sd']}")


if __name__ == "__main__":
    main()
