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
    # not marked a SYNTHETIC grid, but it still discloses its SYNTHETIC outcomes (check_data requires it, #110)
    assert grid["source"] == "MEASURED" and "Marked SYNTHETIC" not in grid["note"]
    assert "SYNTHETIC Weibull outcomes" in grid["note"]
    assert grid["meta"]["cohorts"] == {"charville": FACTS}
    assert grid["meta"]["cohort_split"] == "all"  # only the marker counts read both sides of the UPMC lock
    assert grid["meta"]["input_split"] == "unknown"  # these results carry no split column
    assert grid["meta"]["method"].startswith("Model-based Monte Carlo") and "Monte Carlo" in grid["note"]
    no_size = build(pd.DataFrame(summaries()).drop(columns="region_um"))
    assert no_size["meta"]["cohorts"]["charville"] == {k: v for k, v in FACTS.items() if k != "region_um"}
    assert build(summaries(), max_regions=8)["axes"]["regions"] == [1, 2, 4]


def test_input_split_comes_from_the_results_not_the_cohort_block():
    df = pd.DataFrame(summaries())
    assert build(df.assign(split="discovery"))["meta"]["input_split"] == "discovery"
    for bad in (df.assign(split=["discovery", "all"] * (len(df) // 2)), df.assign(split="held-out")):
        with pytest.raises(ValueError, match="split"):
            build(bad)


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


# The measured path (#110): Atlas's committed summaries must publish and pass check_data
COUNTED = ("power", "fwer", "identification_rate", "wrong_identification_rate")


def measured_rows():
    """summaries() with the binomial counts run_point stores: every rate is its count over its studies."""
    rows = summaries()
    for r in rows:
        for rate in COUNTED:
            r[f"{rate}_n"] = 1000
            r[f"{rate}_count"] = round(r[rate] * 1000)
            r[rate] = r[f"{rate}_count"] / 1000
        r["power_lo"], r["power_hi"] = max(0.0, r["power"] - 0.03), min(1.0, r["power"] + 0.03)
    return rows


def test_measured_rows_keep_their_counts_and_pass_check_data(tmp_path):
    grid = build(measured_rows(), measured=True)
    assert grid["source"] == "MEASURED" and check(grid, tmp_path).returncode == 0
    for p in grid["points"]:
        for rate in COUNTED:
            assert type(p[f"{rate}_count"]) is int and p[f"{rate}_count"] / 1000 == p[rate]
    for text in ("SYNTHETIC", "legacy R4", "not an R15", "ESTIMATE", "not observed clinical rates"):
        assert text in grid["note"]


@pytest.mark.parametrize("change, message", [
    ({"power_count": np.nan}, "power_count"), ({"power_count": 2.5}, "power_count"), ({"power_count": -1}, "power_count"),
    ({"power_count": 1001}, "power_count"), ({"power_count": 7}, "power_count"),
    ({"fwer_count": 7}, "fwer_count"), ({"fwer_n": 0}, "fwer_n"),
])
def test_measured_counts_must_be_whole_and_match_their_rates(change, message):
    rows = measured_rows()
    rows[0].update(change)
    with pytest.raises(ValueError, match=message):
        build(rows, measured=True)


def test_measured_export_needs_power_count_and_never_derives_it():
    with pytest.raises(ValueError, match="power_count"):
        build(pd.DataFrame(measured_rows()).drop(columns="power_count"), measured=True)
    with pytest.raises(ValueError, match="fwer_n"):
        build(pd.DataFrame(measured_rows()).drop(columns="fwer_n"), measured=True)


def test_default_synthetic_points_keep_their_fields():
    p = build(measured_rows())["points"][0]
    assert set(p) == {"patients", "regions", "fov_um", "miscall", "power", "identification_rate", "power_lo",
                      "power_hi", "fwer", "wrong_identification_rate"}


@pytest.mark.parametrize("cohort, n_points", [("charville", 48), ("upmc", 96)])
def test_real_atlas_summary_exports_measured_and_passes_check_data(cohort, n_points, tmp_path):
    base = ROOT / "atlas" / "results" / "production" / cohort
    if not (base / "summary.parquet").exists():
        pytest.skip(f"Atlas's {cohort} production export not in this checkout")
    committed = json.loads((base / "grid.json").read_text())
    df = eg.read_results(base / "summary.parquet")
    grid = eg.build_grid(df, committed["cohort"], committed["meta"]["max_regions_per_patient"], cohort_key=cohort,
                         measured=True)
    result = check(grid, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(grid["points"]) == n_points and "legacy R4" in grid["note"] and "not an R15" in grid["note"]
    for p in grid["points"] + grid["meta"]["all_regions_points"]:
        row = df[(df.patients == p["patients"]) & (df.eps == p["miscall"])
                 & (df.regions_per_patient.isna() if p["regions"] == "all" else df.regions_per_patient == p["regions"])
                 & (df.fov_um.isna() if p["fov_um"] == "full" else df.fov_um == p["fov_um"])]
        assert len(row) == 1
        for rate in COUNTED:
            assert (p[rate], p[f"{rate}_count"], p[f"{rate}_n"]) == (
                row[rate].iloc[0], row[f"{rate}_count"].iloc[0], row[f"{rate}_n"].iloc[0])


# Finite-bank provenance (#40, after Atlas's #45)
BANK_ROW = {"source_hash": "s" * 64, "grammar_hash": "g" * 64, "measurement_model": "finite-bank approximation",
            "bank_draws": 64, "bank_seed": 20261121}


def bank_rows():
    """summaries() as #45 emits them: one bank per imaging design, the same bank at every patient count."""
    rows = summaries()
    for r in rows:
        design = f"{r['regions_per_patient']}-{r['fov_um']}-{r['eps']}"
        r.update(BANK_ROW, bank_key=f"key-{design}", bank_hash=f"hash-{design}")
    return rows


def test_bank_summary_round_trips_with_its_label_a_hashes_and_one_bank_per_imaging_design(tmp_path):
    path = tmp_path / "points.parquet"
    pd.DataFrame(bank_rows()).to_parquet(path)
    grid = json.loads(json.dumps(eg.build_grid(eg.read_results(path), COHORT, 2, cohort_key="charville"),
                                 allow_nan=False))
    meta = grid["meta"]
    assert (meta["measurement_model"], meta["bank_draws"]) == ("finite-bank approximation", 64)
    assert (meta["source_hash"], meta["grammar_hash"]) == ("s" * 64, "g" * 64)
    for text in (meta["method"], grid["note"]):
        assert "finite-bank approximation" in text and "A=64" in text
    for p in grid["points"] + meta["all_regions_points"]:
        regions = None if p["regions"] == "all" else p["regions"]
        fov = None if p["fov_um"] == "full" else float(p["fov_um"])
        design = f"{regions}-{fov}-{p['miscall']}"
        assert (p["bank_key"], p["bank_hash"], p["bank_seed"]) == (f"key-{design}", f"hash-{design}", 20261121)
    assert len({p["bank_key"] for p in grid["points"]}) == 2 * 2 * 2  # distinct imaging banks all retained
    assert [p["power"] for p in grid["points"]] == [p["power"] for p in build(summaries())["points"]]
    assert check(grid, tmp_path).returncode == 0


@pytest.mark.parametrize("column, value", [("source_hash", "t" * 64), ("grammar_hash", "h" * 64),
                                           ("measurement_model", "direct"), ("bank_draws", 32)])
def test_mixed_source_grammar_method_or_a_is_refused(column, value):
    rows = bank_rows()
    rows[3][column] = value
    with pytest.raises(ValueError, match=f"rows mix {column}"):
        build(rows)


@pytest.mark.parametrize("change, message", [
    ({"bank_key": "key-other"}, "imaging design"), ({"bank_hash": "hash-other"}, "imaging design"),
    ({"bank_seed": 1}, "imaging design"), ({"bank_key": None}, "bank_key"), ({"bank_hash": np.nan}, "bank_hash"),
])
@pytest.mark.parametrize("row", [16, -1])  # P=160 on the 1-region line, and P=160 off the lines (4 regions)
def test_one_imaging_design_keeps_one_bank_at_every_patient_count(change, message, row):
    rows = bank_rows()
    assert rows[row]["patients"] == 160
    rows[row].update(change)  # P=40 has the same imaging design with the original bank
    with pytest.raises(ValueError, match=message):
        build(rows)


def test_two_imaging_designs_cannot_share_one_bank():
    rows = bank_rows()
    for r in rows:
        if r["eps"] == 0.1:
            r["bank_key"] = r["bank_key"].replace("0.1", "0.0")
            r["bank_hash"] = r["bank_hash"].replace("0.1", "0.0")
    with pytest.raises(ValueError, match="share bank"):
        build(rows)


@pytest.mark.parametrize("change, message", [({"bank_draws": 1}, "bank_draws"), ({"measurement_model": "guess"}, "measurement_model"),
                                             ({"source_hash": None}, "source_hash")])
def test_bank_curve_needs_a_known_model_a_of_at_least_two_and_its_hashes(change, message):
    rows = bank_rows()
    for r in rows:
        r.update(change)
    with pytest.raises(ValueError, match=message):
        build(rows)


def test_bank_curve_needs_every_provenance_column():
    for column in ("source_hash", "grammar_hash", "bank_draws", "bank_key"):
        with pytest.raises(ValueError, match=column):
            build(pd.DataFrame(bank_rows()).drop(columns=column))


def test_legacy_and_direct_rows_still_export_without_bank_fields():
    legacy = build(summaries())
    assert "measurement_model" not in legacy["meta"] and "finite-bank" not in legacy["note"] + legacy["meta"]["method"]
    assert not any("bank_key" in p for p in legacy["points"])
    direct = bank_rows()
    for r in direct:
        r.update(measurement_model="direct", bank_draws=0, bank_seed=None, bank_key=None, bank_hash=None)
    grid = build(direct)
    assert grid["meta"]["measurement_model"] == "direct" and "finite-bank" not in grid["note"]
    assert not any("bank_key" in p for p in grid["points"])
    rows = summaries()
    rows[0]["bank_key"] = "stray"
    with pytest.raises(ValueError, match="direct"):
        build(rows)


REAL = ROOT / "atlas" / "results" / "production" / "charville" / "summary.parquet"


@pytest.mark.skipif(not REAL.exists(), reason="Atlas's #5 production export not in this checkout")
def test_real_atlas_export_keeps_its_12_banks_and_label(tmp_path):
    df = eg.read_results(REAL)
    grid = eg.build_grid(df, COHORT, 2, cohort_key="charville")
    meta, everything = grid["meta"], grid["points"] + grid["meta"]["all_regions_points"]
    assert (meta["measurement_model"], meta["bank_draws"]) == ("finite-bank approximation", 64)
    assert (meta["source_hash"], meta["grammar_hash"]) == (df.source_hash.iloc[0], df.grammar_hash.iloc[0])
    assert {p["bank_key"] for p in everything} == set(df.bank_key)
    assert check(grid, tmp_path).returncode == 0


def test_distinct_bank_keys_may_share_a_content_hash():
    """#45 keys a bank by its imaging descriptor; two descriptors can image identical content (a FOV past the core)."""
    rows = bank_rows()
    for r in rows:
        r["bank_hash"] = r["bank_hash"].replace("200.0", "None")  # FOV 200 and full core: same content, own keys
    grid = build(rows)
    pts = grid["points"] + grid["meta"]["all_regions_points"]
    assert len({p["bank_key"] for p in pts}) == len({(p["regions"], p["fov_um"], p["miscall"]) for p in pts})
    assert len({p["bank_hash"] for p in pts}) < len({p["bank_key"] for p in pts})


def test_bank_seed_stays_an_exact_integer_through_parquet_and_json(tmp_path):
    seed = 2**53 + 1  # not representable as a float64
    rows = bank_rows()
    for r in rows:
        r["bank_seed"] = seed
    path = tmp_path / "points.parquet"
    pd.DataFrame(rows).to_parquet(path)
    grid = eg.build_grid(eg.read_results(path), COHORT, 2, cohort_key="charville")
    text = json.dumps(grid, allow_nan=False)
    assert all(type(p["bank_seed"]) is int and p["bank_seed"] == seed for p in grid["points"])
    assert str(seed) in text and all(p["bank_seed"] == seed for p in json.loads(text)["points"])
