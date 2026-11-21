"""A-13.2@1: the held-out split is stratified by event, seeded, hashed and reproduced byte for byte (offline)."""
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


def test_split_is_stratified_seeded_whole_units_and_hashed(data_dir):
    m = heldout.build(data_dir / "upmc_labels.csv")
    c = m["counts"]
    assert c["patients"] == 23 and c["regions_mapped"] == 25 and c["regions_unmapped"] == 5
    # 26 tuples, 25 units (case 888 links tuples 500 and 501): 8 with an event (HN1-1, 888, 6 singles),
    # 17 without; 30% of each, rounded.
    assert c["units"] == 25
    assert c["strata"] == {"0": {"units": 17, "locked": 5}, "1": {"units": 8, "locked": 2}}
    assert m == heldout.build(data_dir / "upmc_labels.csv")                       # frozen seed
    assert m["locked_regions"] != heldout.build(data_dir / "upmc_labels.csv", seed=1)["locked_regions"]
    assert m["sha256"] == cohorts.manifest_hash(m)
    lab = cohorts.load_regions("upmc", data_dir, "tuple", split="locked")
    assert set(lab.region_id) == set(m["locked_regions"])
    every = set(cohorts.load_regions("upmc", data_dir, "tuple").patient_id)
    assert not every & set(lab.patient_id)                                       # no tuple on both sides


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


def test_rebuild_reproduces_the_manifest_byte_for_byte_and_never_overwrites(data_dir):
    out = data_dir / "manifest.json"
    assert heldout.main(data_dir / "upmc_labels.csv", out) == 0
    first = out.read_bytes()
    assert heldout.main(data_dir / "upmc_labels.csv", out) == 0 and out.read_bytes() == first
    out.write_text(out.read_text().replace('"seed": 20261121', '"seed": 1'))
    edited = out.read_bytes()
    assert heldout.main(data_dir / "upmc_labels.csv", out) == 1 and out.read_bytes() == edited


@pytest.mark.parametrize("seed", range(8))
def test_no_patient_is_on_both_sides_under_any_identity(data_dir, seed):
    # Case 888's regions sit in two clinical tuples (500, 501); sampling tuples alone split it (seed 0).
    m = heldout.build(data_dir / "upmc_labels.csv", seed=seed)
    (data_dir / "lock.json").write_text(json.dumps(m))
    case888 = {"UPMC_c006_v001_r001_reg009", "UPMC_c006_v001_r001_reg010"}
    assert len(case888 & set(m["locked_regions"])) in (0, 2)
    for identity in ("case", "tuple", "tuple+case", "patient"):
        disc, lock = (set(cohorts.load_regions("upmc", data_dir, identity, split=s).patient_id)
                      for s in ("discovery", "locked"))
        assert not disc & lock, (identity, disc & lock)
