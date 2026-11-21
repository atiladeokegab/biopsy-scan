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


def summaries(hr=2.0):
    """Invented rows in run_point's summary shape, including columns the export ignores."""
    rows = []
    for n in (40, 160):
        for fov in (200.0, None):
            for eps in (0.0, 0.1):
                power = min(1.0, n / 200 + (0.1 if fov is None else 0) - eps)
                rows.append({"design_id": f"{n}{fov}{eps}", "cohort": "charville", "patients": n,
                             "regions_per_patient": 2, "fov_um": fov, "eps": eps, "backend": "numpy",
                             "alpha": 0.05, "hr_per_source_sd": hr, "power": power, "power_n": 1000,
                             "identification_rate": power * 0.8, "fwer": 0.05})
    return rows


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
    grid = eg.build_grid(eg.read_results(path), COHORT, cohort_key="charville")
    assert grid["source"] == "SYNTHETIC" and "SYNTHETIC" in grid["note"]
    assert grid["axes"]["fov_um"] == [200, "full"] and grid["axes"]["patients"] == [40, 160]
    assert len(grid["points"]) == 8 and grid["studies_per_point"] == 1000 and grid["effect_hr_per_sd"] == 2.0
    r = check(grid, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "8 points, source SYNTHETIC" in r.stdout


def test_measured_only_when_asked():
    grid = eg.build_grid(pd.DataFrame(summaries()), COHORT, cohort_key="charville", measured=True)
    assert grid["source"] == "MEASURED" and "SYNTHETIC" not in grid["note"]


def test_several_effects_need_an_explicit_hr():
    df = pd.DataFrame(summaries(2.0) + summaries(1.5))
    with pytest.raises(ValueError, match="--hr"):
        eg.build_grid(df, COHORT, cohort_key="charville")
    assert eg.build_grid(df, COHORT, cohort_key="charville", hr=1.5)["effect_hr_per_sd"] == 1.5


def test_bad_results_fail_clearly():
    with pytest.raises(ValueError, match="repeat a design point"):
        eg.build_grid(pd.DataFrame(summaries() + summaries()[:1]), COHORT, cohort_key="charville")
    with pytest.raises(ValueError, match="lack columns"):
        eg.build_grid(pd.DataFrame(summaries()).drop(columns="power_n"), COHORT, cohort_key="charville")
    with pytest.raises(ValueError, match="no upmc rows"):
        eg.build_grid(pd.DataFrame(summaries()), COHORT, cohort_key="upmc", hr=2.0)
