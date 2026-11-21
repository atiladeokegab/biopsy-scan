"""scripts/export_grid.py: atlas run_point summaries -> preflight-grid/1 that demo/check_data.py accepts."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_grid", ROOT / "scripts" / "export_grid.py")
eg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg)

COHORT = {"name": "Charville", "patients": 159, "regions": 2, "regions_mean": 1.8, "fov_um": "full", "miscall": 0.0}
FACTS = {"source_n": 159, "source_events": 38, "source_regions": 286, "region_um": 720.0}


def summaries(hr=2.0, regions=(1, 2, None, 4)):
    """Invented rows in run_point's final summary shape, including columns the export ignores."""
    rows = []
    for n in (40, 160):
        for k in regions:
            for fov in (200.0, None):
                for eps in (0.0, 0.1):
                    power = min(1.0, n / 200 + (0.1 if fov is None else 0) - eps)
                    rows.append({"design_id": f"{n}{k}{fov}{eps}", "cohort": "charville", "patients": n,
                                 "regions_per_patient": k, "fov_um": fov, "eps": eps, "backend": "numpy",
                                 "alpha": 0.05, "hr_per_source_sd": hr, "plant_feature": 17, "plant_mean": 0.3,
                                 "plant_sd": 0.1, "shape": 1.5, "power": power, "power_count": int(power * 1000),
                                 "power_n": 1000, "power_lo": max(0.0, power - 0.03), "power_hi": min(1.0, power + 0.03),
                                 "identification_rate": power * 0.8, "fwer": 0.05,
                                 "wrong_identification_rate": power * 0.1, "omnibus_fwer": 0.04, **FACTS})
    return rows


def build(rows, **kw):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    return eg.build_grid(df, COHORT, kw.pop("max_regions", 2), cohort_key=kw.pop("cohort_key", "charville"), **kw)


def check(grid, tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "grid.json").write_text(json.dumps(grid, allow_nan=False))
    shutil.copy(ROOT / "demo" / "data" / "tissue.json", d / "tissue.json")
    return subprocess.run([sys.executable, "-I", str(ROOT / "demo" / "check_data.py"), str(d)],
                          capture_output=True, text=True)


def test_parquet_summaries_become_a_grid_check_data_accepts(tmp_path):
    path = tmp_path / "points.parquet"
    pd.DataFrame(summaries()).to_parquet(path)
    grid = eg.build_grid(eg.read_results(path), COHORT, 2, cohort_key="charville")
    assert grid["source"] == "SYNTHETIC" and "SYNTHETIC" in grid["note"]
    assert grid["axes"]["fov_um"] == [200, "full"] and grid["axes"]["regions"] == [1, 2]
    assert len(grid["points"]) == 16 and grid["studies_per_point"] == 1000 and grid["effect_hr_per_sd"] == 2.0
    assert all(p["power_lo"] <= p["power"] <= p["power_hi"] for p in grid["points"])
    assert all(p["fwer"] == 0.05 and "wrong_identification_rate" in p and "omnibus_fwer" not in p for p in grid["points"])
    assert grid["meta"]["model"] == {"alpha": 0.05, "plant_feature": 17, "plant_mean": 0.3, "plant_sd": 0.1, "shape": 1.5}
    r = check(grid, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "16 points, source SYNTHETIC" in r.stdout


@pytest.mark.parametrize("store", ["object", "float", "Int64", "csv", "jsonl"])
def test_all_regions_rows_route_to_meta_without_casting(store, tmp_path):
    df = pd.DataFrame(summaries())
    if store == "object":
        df["regions_per_patient"] = df.regions_per_patient.astype(object).where(df.regions_per_patient.notna(), None)
    elif store == "Int64":
        df["regions_per_patient"] = df.regions_per_patient.astype("Int64")
    elif store in ("csv", "jsonl"):
        path = tmp_path / f"points.{store}"
        df.to_csv(path, index=False) if store == "csv" else df.to_json(path, orient="records", lines=True)
        df = eg.read_results(path)
    meta = build(df)["meta"]
    assert len(meta["all_regions_points"]) == 8 and {p["regions"] for p in meta["all_regions_points"]} == {"all"}
    assert meta["region_lines"] == [1, 2] and meta["points_off_region_lines"] == 8  # regions 4 > 2 available


def test_source_facts_and_method_in_meta_even_when_measured():
    grid = build(summaries(), measured=True)
    assert grid["source"] == "MEASURED" and "SYNTHETIC" not in grid["note"]
    assert grid["meta"]["cohorts"] == {"charville": FACTS}
    assert grid["meta"]["split"] == "all"  # the grid reads both sides of the locked UPMC split
    assert grid["meta"]["method"].startswith("Model-based Monte Carlo") and "Monte Carlo" in grid["note"]
    no_size = build(pd.DataFrame(summaries()).drop(columns="region_um"))
    assert no_size["meta"]["cohorts"]["charville"] == {k: v for k, v in FACTS.items() if k != "region_um"}
    assert build(summaries(), max_regions=8)["axes"]["regions"] == [1, 2, 4]


def test_several_effects_need_an_explicit_hr():
    rows = summaries(2.0) + summaries(1.5)
    with pytest.raises(ValueError, match="--hr"):
        build(rows)
    assert build(rows, hr=1.5)["effect_hr_per_sd"] == 1.5


@pytest.mark.parametrize("column, value", [("alpha", 0.01), ("plant_feature", 3), ("shape", 2.0),
                                           ("plant_sd", 0.2)])
def test_mixed_model_settings_never_merge_into_one_curve(column, value):
    rows = summaries()
    rows[5][column] = value
    with pytest.raises(ValueError, match=f"rows mix {column}"):
        build(rows)


def test_mixed_study_counts_are_refused():
    rows = summaries()
    rows[3]["power_n"] = 500
    with pytest.raises(ValueError, match="rows mix power_n"):
        build(rows)


@pytest.mark.parametrize("column, value, message", [
    ("patients", np.nan, "patients"), ("patients", 40.5, "patients"), ("regions_per_patient", 1.5, "regions_per_patient"),
    ("fov_um", -5.0, "fov_um"), ("fov_um", np.inf, "fov_um"), ("eps", 1.2, "eps"), ("power", np.nan, "power"),
    ("identification_rate", 0.99, "identification_rate .* above power"), ("power_hi", 0.0, "outside its interval"),
    ("fwer", -0.1, "fwer"), ("wrong_identification_rate", np.nan, "wrong_identification_rate"),
    ("source_events", 3.5, "source_events"), ("region_um", -1.0, "region_um"),
])
def test_bad_values_are_refused(column, value, message):
    rows = summaries()
    for r in rows if column.startswith(("source", "region_um")) else rows[:1]:
        r[column] = value
    with pytest.raises(ValueError, match=message):
        build(rows)


def test_bad_tables_fail_clearly():
    with pytest.raises(ValueError, match="repeat a design point"):
        build(summaries() + summaries()[:1])
    with pytest.raises(ValueError, match="lack columns"):
        build(pd.DataFrame(summaries()).drop(columns="alpha"))
    with pytest.raises(ValueError, match="no upmc rows"):
        build(summaries(), cohort_key="upmc")
    with pytest.raises(ValueError, match="no design points on the regions lines"):
        build(summaries(regions=(None, 8)))
    rows = summaries()
    rows[0]["source_events"] = 37
    with pytest.raises(ValueError, match="disagree on the source cohort facts"):
        build(rows)


def test_cli_writes_strict_json_or_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(eg, "cohort_block", lambda cohort: (COHORT, 2))
    good, bad, out = tmp_path / "good.parquet", tmp_path / "bad.parquet", tmp_path / "grid.json"
    pd.DataFrame(summaries()).to_parquet(good)
    rows = summaries()
    rows[0]["power"] = np.nan
    pd.DataFrame(rows).to_parquet(bad)
    with pytest.raises(SystemExit) as e:
        eg.main([str(bad), "--out", str(out)])
    assert e.value.code and not out.exists()
    eg.main([str(good), "--out", str(out)])
    json.loads(out.read_text(), parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
