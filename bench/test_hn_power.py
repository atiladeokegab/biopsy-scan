"""bench/hn_power.py: the discovery source never reads an outcome column and never holds a locked patient."""
import json
import os
import subprocess
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


def _cache(where, regions):
    np.savez(where / "upmc_cells.discovery.npz", region_ids=np.array(regions, dtype=str))


def test_a_cache_of_discovery_regions_passes(labels):
    _cache(labels, [REGIONS[0], REGIONS[1], REGIONS[2], REGIONS[4]])  # the unlabelled region is discovery, not locked
    h.check_cache(labels)


def test_a_cache_holding_a_locked_region_is_refused(labels):
    _cache(labels, REGIONS)
    with pytest.raises(PermissionError):
        h.check_cache(labels)


def test_a_cache_holding_a_locked_patients_region_is_refused(labels):
    """A region of a locked patient that the region list missed is still caught by its case number."""
    lock = json.loads(h.LOCK.read_text())
    h.LOCK.write_text(json.dumps({**lock, "locked_patients": ["LOCK9", "HN2"]}))
    _cache(labels, REGIONS[:3])
    with pytest.raises(PermissionError):
        h.check_cache(labels)


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
    if (UPMC.parent / "upmc_cells.discovery.npz").exists():
        h.check_cache(UPMC.parent)


# --- the published curve: receipts validated, failures kept, the answer re-derived (Prometheus #133 round 1)
RAW5 = h.ROOT / "bench" / "results" / "modal-hn-power-20261122T042657Z.json"
RAW300 = h.ROOT / "bench" / "results" / "modal-hn-power-20261122T053210Z.json"


def _receipts():
    return [json.loads(RAW5.read_text()), json.loads(RAW300.read_text())]


def test_the_committed_curve_rederives_from_its_receipts():
    got = h.from_raw(RAW5, RAW300)
    assert json.loads(h.RESULT.read_text())["answer"] == got["answer"]
    assert got["answer"]["missing_patients"] == [] and [p["patients"] for p in got["points"]] == list(h.PLANNED)
    assert got["metadata"]["measurement_model"] and "conditional" in got["metadata"]["mc_interval_scope"]


def test_a_failed_planned_point_makes_the_answer_partial():
    ds = _receipts()
    p200 = next(p for p in ds[0]["points"] if p["patients"] == 200)
    del p200["rows"]
    p200["error"] = "TimeoutError()"
    out = h.validate(h.summarize(*h.collect(ds)))
    assert out["answer"]["missing_patients"] == [200] and out["answer"]["statement"].startswith("PARTIAL")


def test_a_point_from_another_source_is_refused():
    ds = _receipts()
    next(p for p in ds[0]["points"] if p["patients"] == 400)["rows"][0]["source_hash"] = "0" * 64
    with pytest.raises(ValueError, match="source_hash"):
        h.collect(ds)


def test_a_point_whose_row_is_for_another_count_is_refused():
    ds = _receipts()
    next(p for p in ds[0]["points"] if p["patients"] == 400)["rows"][0]["patients"] = 401
    with pytest.raises(ValueError, match="one row"):
        h.collect(ds)


def test_conflicting_duplicates_are_refused():
    ds = _receipts()
    dup = json.loads(json.dumps(next(p for p in ds[0]["points"] if p["patients"] == 100)))
    dup["rows"][0]["power_count"] += 1
    ds[1]["points"].append(dup)
    with pytest.raises(ValueError, match="conflicting"):
        h.collect(ds)


@pytest.mark.parametrize("edit", [
    lambda s: s["answer"].__setitem__("n_80", 200.0),
    lambda s: s["points"][2].__setitem__("power_lo", 0.5),
    lambda s: s["points"][1].__setitem__("rejections", 5),
    lambda s: s["points"][0].__setitem__("power_hi", 0.3),  # an interval edit the answer alone would not show
])
def test_a_tampered_published_curve_is_refused(edit):
    s = json.loads(h.RESULT.read_text())
    h.validate(s)
    edit(s)
    with pytest.raises(ValueError):
        h.validate(s)


def test_the_remote_image_never_holds_the_locked_manifest(tmp_path):
    out = h.stage(h.ROOT, "HEAD", tmp_path)
    assert (out / "core" / "cohorts.py").exists() and not list(out.rglob("upmc_locked*"))


def test_the_payload_producer_refuses_tracked_changes(tmp_path):
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    (tmp_path / "f.py").write_text("a = 1\n")
    git("add", "f.py")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "t")
    assert len(h.producer(tmp_path)) == 40
    (tmp_path / "f.py").write_text("a = 2\n")
    with pytest.raises(RuntimeError, match="tracked changes"):
        h.producer(tmp_path)


# --- Zeus ruling on #133 round 2: every copy validated, counts bound to the budget, provenance required
def _point(ds, n, i=0):
    return next(p for p in ds[i]["points"] if p["patients"] == n)


def test_a_duplicate_with_other_metadata_is_refused_whatever_the_order():
    for first in (True, False):
        ds = _receipts()
        dup = json.loads(json.dumps(_point(ds, 100)))
        dup["metadata"]["run_id"] = "another run"
        clean = json.loads(json.dumps(_point(ds, 100)))
        ds[1]["points"] += [dup, clean] if first else [clean, dup]
        with pytest.raises(ValueError, match="conflicting"):
            h.collect(ds)


def test_counts_below_the_declared_budget_are_refused_unless_marked_partial():
    ds = _receipts()
    row = _point(ds, 200)["rows"][0]
    row["power_count"], row["power_n"] = 74, 100
    with pytest.raises(ValueError, match="budget"):
        h.collect(ds)
    _point(ds, 200)["status"] = "PARTIAL"
    pts = h.summarize(*h.collect(ds))["points"]
    assert next(p for p in pts if p["patients"] == 200)["label"] == "PARTIAL: 100 of 200 planted studies"


@pytest.mark.parametrize("field", ["source_hash", "bank_hash"])
def test_provenance_missing_everywhere_is_refused(field):
    ds = _receipts()
    for d in ds:
        for p in d["points"]:
            p["rows"][0].pop(field, None)
            p["metadata"].pop(field, None)
    with pytest.raises(ValueError, match="missing or null"):
        h.collect(ds)


def test_null_provenance_is_refused():
    ds = _receipts()
    for d in ds:
        for p in d["points"]:
            p["rows"][0]["grammar_hash"] = None
            p["metadata"]["grammar_hash"] = None
    with pytest.raises(ValueError, match="missing or null"):
        h.collect(ds)
