"""scripts/export_grid.py: atlas run_point summaries -> preflight-grid/1 that demo/check_data.py accepts."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

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
                                 "hr_per_source_sd": hr, "power": power, "power_count": int(power * 1000),
                                 "power_n": 1000, "power_lo": max(0.0, power - 0.03), "power_hi": min(1.0, power + 0.03),
                                 "identification_rate": power * 0.8, "fwer": 0.05,
                                 "wrong_identification_rate": power * 0.1, "omnibus_fwer": 0.04, **FACTS})
    return rows


def build(rows, **kw):
    return eg.build_grid(pd.DataFrame(rows), COHORT, kw.pop("max_regions", 2), cohort_key=kw.pop("cohort_key", "charville"), **kw)


def check(grid, tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "grid.json").write_text(json.dumps(grid))
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
    r = check(grid, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "16 points, source SYNTHETIC" in r.stdout


def test_all_regions_and_lines_beyond_the_cohort_go_to_meta():
    meta = build(summaries())["meta"]
    assert meta["region_lines"] == [1, 2] and meta["points_off_region_lines"] == 8  # regions 4 > 2 available
    assert len(meta["all_regions_points"]) == 8 and {p["regions"] for p in meta["all_regions_points"]} == {"all"}
    assert meta["cohorts"] == {"charville": FACTS}
    assert build(summaries(), max_regions=8)["axes"]["regions"] == [1, 2, 4]


def test_measured_only_when_asked():
    grid = build(summaries(), measured=True)
    assert grid["source"] == "MEASURED" and "SYNTHETIC" not in grid["note"]


def test_several_effects_need_an_explicit_hr():
    rows = summaries(2.0) + summaries(1.5)
    with pytest.raises(ValueError, match="--hr"):
        build(rows)
    assert build(rows, hr=1.5)["effect_hr_per_sd"] == 1.5


def test_bad_results_fail_clearly():
    with pytest.raises(ValueError, match="repeat a design point"):
        build(summaries() + summaries()[:1])
    with pytest.raises(ValueError, match="lack columns"):
        build(pd.DataFrame(summaries()).drop(columns="power_lo").to_dict("records"))
    with pytest.raises(ValueError, match="no upmc rows"):
        build(summaries(), cohort_key="upmc")
    with pytest.raises(ValueError, match="no design points on the regions lines"):
        build(summaries(regions=(None, 8)))
    rows = summaries()
    rows[0]["source_events"] = 37
    with pytest.raises(ValueError, match="disagree on the source cohort facts"):
        build(rows)


def test_nullable_int_regions_read_as_all():
    df = pd.DataFrame(summaries())
    df["regions_per_patient"] = df.regions_per_patient.astype("Int64")  # pd.NA, as a nullable parquet column reads
    assert len(eg.build_grid(df, COHORT, 2, cohort_key="charville")["meta"]["all_regions_points"]) == 8
