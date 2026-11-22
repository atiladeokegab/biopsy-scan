"""bench/hn_power.py: the discovery source never reads an outcome column and never holds a locked patient."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench import hn_power as h

REGIONS = ["UPMC_c001_v001_r001_reg001", "UPMC_c001_v001_r001_reg002", "UPMC_c004_v001_r001_reg003",
           "UPMC_c002_v001_r001_reg004", "UPMC_c005_v001_r001_reg005"]
CASES = ["HN1 a", "HN1 b", "HN2", "LOCK9", "NA"]


@pytest.fixture
def labels(tmp_path, monkeypatch):
    """Five regions: two of patient HN1, one of HN2, one locked (LOCK9), one unlabelled. Outcome cells are junk."""
    pd.DataFrame({"region_id": REGIONS, "status": "junk", "primary_outcome": "junk", "recurred": "junk",
                  "hpvstatus": "junk", "survival_status": "junk", "survival_day": "junk",
                  "acquisition_id_visualizer": "x", "sample_label_visualizer": CASES}).to_csv(tmp_path / "upmc_labels.csv",
                                                                                              index=False)
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"locked_regions": [REGIONS[3]], "locked_patients": ["LOCK9"]}))
    monkeypatch.setattr(h, "LOCK", lock)
    return tmp_path


def test_discovery_patients_read_no_outcome_column(labels, monkeypatch):
    seen = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: seen.append(k.get("usecols")) or real(*a, **k))
    reg = h.discovery_patients(labels)
    assert seen == [h.LABEL_COLUMNS] and not h.OUTCOME_COLUMNS & set(seen[0])
    assert list(reg.region_id) == REGIONS[:3] and list(reg.case) == ["HN1", "HN1", "HN2"]
    assert list(reg.group) == [0, 0, 1]


def test_a_locked_patient_on_the_discovery_side_is_refused(labels):
    reg = h.discovery_patients(labels)
    with pytest.raises(PermissionError):
        h.check_disjoint(pd.concat([reg, pd.DataFrame({"region_id": ["other"], "case": ["LOCK9"]})]),
                         json.loads(h.LOCK.read_text()))
    with pytest.raises(PermissionError):
        h.check_disjoint(pd.concat([reg, pd.DataFrame({"region_id": [REGIONS[3]], "case": ["HN3"]})]),
                         json.loads(h.LOCK.read_text()))


def test_a_lock_that_names_a_discovery_case_is_refused(labels):
    h.LOCK.write_text(json.dumps({"locked_regions": [REGIONS[3]], "locked_patients": ["LOCK9", "HN2"]}))
    with pytest.raises(PermissionError):
        h.discovery_patients(labels)


def test_placeholder_outcomes_reach_the_planned_event_fraction():
    from atlas.simulation import fit_model

    rng = np.random.default_rng(0)
    cells = {"time": np.ones(40), "event": np.zeros(40, bool), "strata": np.repeat([0, 1], 20)}
    m = fit_model(cells, rng.normal(size=(40, 3)), hr=2.0, target=h.TARGET_EVENT_FRACTION)
    assert m["expected_event_fraction"] == pytest.approx(h.TARGET_EVENT_FRACTION, abs=1e-9)


UPMC = Path(os.environ.get("DATA_DIR", "data")) / "upmc_labels.csv"


@pytest.mark.skipif(not UPMC.exists(), reason="UPMC labels not present")
def test_real_discovery_side_is_34_patients_disjoint_from_the_lock():
    reg = h.discovery_patients(UPMC.parent)
    lock = json.loads((h.ROOT / "core" / "upmc_locked.json").read_text())
    assert reg.case.nunique() == 34 and len(reg) == 125
    assert not set(reg.region_id) & set(lock["locked_regions"])
    assert not set(reg.case) & set(map(str, lock["locked_patients"]))
