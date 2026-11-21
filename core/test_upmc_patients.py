"""A-13.1@1: UPMC regions grouped by patient with the documented rule; counts come from the data (offline)."""
import pandas as pd

from core import cohorts, heldout
from core.test_holdout import ROWS, data_dir  # noqa: F401  (the shared invented fixture)


def test_patient_rule_maps_cases_and_na_tuples_and_reports_conflicts(data_dir):
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


def test_counts_are_reported_and_no_two_patients_share_a_region(data_dir):  # noqa: F811
    c = heldout.build(data_dir / "upmc_labels.csv")["counts"]
    # MEASURED on the fixture: 23 patients (2 case + 1 NA tuple + 20 singles), 7 with an event; 5 regions unmapped.
    assert (c["patients"], c["patient_events"], c["regions_unmapped"]) == (23, 7, 5)
    reg = cohorts.load_regions("upmc", data_dir, "patient", split="all")
    assert reg.region_id.is_unique and len(reg) == c["regions_mapped"] == len(ROWS) - 5
    assert reg.groupby("region_id").patient_id.nunique().max() == 1
