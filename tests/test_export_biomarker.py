"""scripts/export_biomarker.py: KM steps by hand, and the committed panel data's shape (offline)."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_biomarker", ROOT / "scripts" / "export_biomarker.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)


def test_km_matches_a_hand_count():
    # day 2: 1 death of 5 at risk -> 0.8; day 3 censored; day 5: 1 death of 3 at risk (5, 5, 7) -> 0.5333.
    t, e = np.array([2, 3, 5, 5, 7.0]), np.array([1, 0, 1, 0, 0], bool)
    assert eb.km(t, e) == [[0.0, 1.0], [2.0, 0.8], [5.0, 0.5333]]


def test_committed_panel_data_is_consistent():
    b = json.loads((ROOT / "demo" / "data" / "biomarker.json").read_text())
    assert b["exploratory"] and b["already_seen"] and b["km"]["p_value"] is None
    assert b["label"] == b["km"]["label"] == eb.LABEL == "exploratory, cutoff chosen on the same cohort, no p-value"
    assert "cutoff chosen on the same cohort, no p-value" in b["note"]
    assert sum(a["n"] for a in b["km"]["arms"]) == b["cohort"]["patients_with_value"]
    for r in b["regions"]:
        c = r["cells"]
        assert len({len(v) for v in c.values()}) == 1 and len(c["x"]) == r["counts"]["cells"]
        assert sum(t == 0 for t in c["type"]) == r["counts"]["tumour"]
        assert all(m <= (t == 1) for m, t in zip(c["near_tumour"], c["type"]))  # only macrophages are flagged


def test_near_tumour_uses_the_features_inclusive_100um_rule():
    # core.truth keeps a pair when d <= 100 (inclusive), so a macrophage at exactly 100 µm counts.
    xy = np.array([[0, 0], [100, 0], [60, 80], [100.0001, 0], [0, 50]], np.float64)
    tum = np.array([1, 0, 0, 0, 0], bool)
    mac = np.array([0, 1, 1, 1, 0], bool)
    assert eb.near_tumour(xy, tum, mac).tolist() == [False, True, True, False, False]
    assert not eb.near_tumour(xy, np.zeros(5, bool), mac).any()

