"""UPMC patient grouping, the locked held-out split and its guard, on an invented fixture (offline)."""
import json
import zipfile

import pytest

from core import cohorts, heldout

HEAD = "region_id,status,primary_outcome,recurred,hpvstatus,survival_status,survival_day,acquisition_id_visualizer,sample_label_visualizer\n"
ROWS = [  # region, tuple, label
    ("UPMC_c001_v001_r001_reg001", "DOD,0,1,0,1,100", "HN1-1 A_1"),
    ("UPMC_c002_v001_r001_reg002", "DOD,0,1,0,1,100", "HN1-1  B_2"),     # same case, double space
    ("UPMC_c003_v001_r001_reg003", "DOD,0,1,0,1,100", "NA"),             # NA in a case's tuple: ambiguous
    ("UPMC_c004_v001_r001_reg004", "NED,1,0,1,0,200", "777 A_3"),
    ("UPMC_c001_v001_r001_reg005", "NED,1,0,0,0,300", "NA"),             # tuple patient
    ("UPMC_c002_v001_r001_reg006", "NED,1,0,0,0,300", "NA"),
    ("UPMC_c001_v001_r001_reg007", "NED,1,0,0,0,400", "NA"),             # spans both coverslip groups
    ("UPMC_c005_v001_r001_reg008", "NED,1,0,0,0,400", "NA"),
    ("UPMC_c006_v001_r001_reg009", "DOD,0,,0,1,500", "888 A_4"),          # case with two outcome tuples
    ("UPMC_c006_v001_r001_reg010", "DOD,0,,0,1,501", "888 B_5"),
]
# Twenty more single-region tuple patients, 6 with an event, so the split has strata to draw from.
ROWS += [(f"UPMC_c007_v001_r001_reg{11 + i:03d}", f"{'DOD' if i < 6 else 'NED'},0,0,0,{int(i < 6)},{1000 + i}", "NA")
         for i in range(20)]


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    (tmp_path / "upmc_labels.csv").write_text(HEAD + "".join(f"{r},{t},a,{lab}\n" for r, t, lab in ROWS))
    with zipfile.ZipFile(tmp_path / "upmc_raw_data.zip", "w") as z:
        for r, _, _ in ROWS:
            z.writestr(f"raw_data/{r}.cell_data.csv", "CELL_ID,X,Y\n1,0,0\n2,10,0\n")
            z.writestr(f"raw_data/{r}.cell_types.csv", "CELL_ID,CELL_TYPE\n1,Tumor\n2,Stroma\n")
    m = heldout.build(tmp_path / "upmc_labels.csv")
    (tmp_path / "lock.json").write_text(json.dumps(m))
    monkeypatch.setattr(cohorts, "LOCK_FILE", tmp_path / "lock.json")
    return tmp_path


def test_patient_rule_maps_cases_and_na_tuples_and_reports_conflicts(data_dir):
    import pandas as pd
    lab = pd.read_csv(data_dir / "upmc_labels.csv", dtype=str, keep_default_na=False)
    pid, conflicts = cohorts.upmc_patient_ids(lab)
    assert pid[:9].tolist() == ["HN1-1", "HN1-1", "", "777", "T:NED|1|0|0|0|300", "T:NED|1|0|0|0|300", "", "", ""]
    assert pid[9] == ""
    assert [(c["kind"], c["group"]) for c in conflicts] == [
        ("case with conflicting outcomes", "888"),
        ("NA region in a tuple holding a case number", "T:DOD|0|1|0|1|100"),
        ("NA tuple spanning both coverslip groups", "T:NED|1|0|0|0|400"),
    ]
    assert conflicts[2]["regions"] == ["UPMC_c001_v001_r001_reg007", "UPMC_c005_v001_r001_reg008"]


def test_split_is_stratified_seeded_whole_units_and_hashed(data_dir):
    m = heldout.build(data_dir / "upmc_labels.csv")
    c = m["counts"]
    assert c["patients"] == 23 and c["regions_mapped"] == 25 and c["regions_unmapped"] == 5
    # 26 tuples: 8 with an event (1 case + 1 two-tuple case + 6 singles), 18 without; 30% of each, rounded.
    assert c["strata"] == {"0": {"units": 17, "locked": 5}, "1": {"units": 9, "locked": 3}}
    assert m == heldout.build(data_dir / "upmc_labels.csv")                       # frozen seed
    assert m["locked_regions"] != heldout.build(data_dir / "upmc_labels.csv", seed=1)["locked_regions"]
    assert m["sha256"] == cohorts.manifest_hash(m)
    lab = cohorts.load_regions("upmc", data_dir, "tuple", split="locked")
    assert set(lab.region_id) == set(m["locked_regions"])
    every = set(cohorts.load_regions("upmc", data_dir, "tuple").patient_id)
    assert not every & set(lab.patient_id)                                       # no tuple on both sides


def test_discovery_loaders_never_return_locked_regions(data_dir):
    locked = cohorts.locked_regions()
    assert locked
    for ident in ("case", "tuple", "tuple+case", "patient"):
        assert not locked & set(cohorts.load_regions("upmc", data_dir, ident).region_id)
    assert not locked & set(cohorts.load_cells("upmc", data_dir=data_dir)["region_ids"])
    assert set(cohorts.load_cells("upmc", data_dir=data_dir, split="locked")["region_ids"]) == locked


def test_guard_raises_when_discovery_asks_for_a_locked_region(data_dir):
    r = sorted(cohorts.locked_regions())[0]
    with pytest.raises(cohorts.LockedPatientError):
        cohorts.load_cells("upmc", [r], data_dir)
    assert cohorts.load_cells("upmc", [r], data_dir, split="locked")["region_ids"] == [r]


def test_without_the_guard_the_locked_region_leaks(data_dir, monkeypatch):
    """The red case the guard exists for: with no lock the same request returns the locked cells."""
    r = sorted(cohorts.locked_regions())[0]
    monkeypatch.setattr(cohorts, "LOCK_FILE", None)
    assert cohorts.load_cells("upmc", [r], data_dir)["region_ids"] == [r]


def test_edited_manifest_is_refused(data_dir):
    m = json.loads((data_dir / "lock.json").read_text())
    m["locked_regions"] = m["locked_regions"][1:]
    (data_dir / "lock.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="sha256"):
        cohorts.load_regions("upmc", data_dir)


def test_committed_manifest_is_intact():
    m = json.loads((cohorts.ROOT / "core" / "upmc_locked.json").read_text())
    assert m["sha256"] == cohorts.manifest_hash(m) and m["seed"] == heldout.SEED
    c = m["counts"]
    assert len(m["locked_regions"]) == c["locked"]["regions"] and len(m["locked_patients"]) == c["locked"]["patients"]
    assert c["locked"]["regions"] + c["discovery"]["regions"] == c["regions"]
    for s in c["strata"].values():
        assert s["locked"] == int(s["units"] * heldout.FRACTION + 0.5)
