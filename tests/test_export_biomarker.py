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


def _tiny_cells():
    rng = np.random.default_rng(0)
    n = 6
    return {"patient_names": np.array(["a", "b"]), "region_names": np.array(["r0", "r1"]),
            "time": np.array([1.0, 2.0]), "event": np.array([1, 0]), "strata": np.array([0, 0], np.int32),
            "cell_type": np.array([0, 1, 2, 0, 1, 2], np.int32), "x": rng.random(n), "y": rng.random(n),
            "region_id": np.array([0, 0, 0, 1, 1, 1]), "channels": rng.random((n, 3)).astype(np.float32),
            "channel_stats": {"mean": [0.0], "std": [1.0]},
            "type_names": np.array(["Tumor 5", "Macrophage", "Other"]), "channel_names": np.array(["PD1", "A", "B"])}


def _saved(cells, **override):
    from audit.charville_v2 import _source_fingerprint
    from core.grammar import grammar_hash, grammar_v2
    saved = {"source_fingerprint": _source_fingerprint(cells, cells["channel_stats"]),
             "grammar_v2_hash": grammar_hash(grammar_v2(list(cells["type_names"]))),
             "n_features_tested": 1, "maxT": {"top_hits": [{"feature": "nope", "grammar_index": -1}]}}
    return {**saved, **override}


def test_build_refuses_a_source_unlike_the_one_the_statistic_saw(tmp_path, monkeypatch):
    import pytest
    cells = _tiny_cells()
    monkeypatch.setattr(eb.cohorts, "load", lambda *a, **k: cells)
    for override, match in (({"source_fingerprint": "0" * 64}, "source fingerprint"),
                            ({"grammar_v2_hash": "0" * 64}, "grammar")):
        monkeypatch.setattr(eb, "RESULT", tmp_path / "v2.json")
        eb.RESULT.write_text(json.dumps(_saved(cells, **override)))
        with pytest.raises(ValueError, match=match):
            eb.build(tmp_path)


def test_source_fingerprint_sees_which_patient_owns_each_region():
    # Swapping region ownership changes truth_v2's per-patient values, so it must change the fingerprint.
    from audit.charville_v2 import _source_fingerprint
    a, b = _tiny_cells(), _tiny_cells()
    a["patient_of_region"], b["patient_of_region"] = np.array([0, 1], np.int32), np.array([1, 0], np.int32)
    assert _source_fingerprint(a, a["channel_stats"]) != _source_fingerprint(b, b["channel_stats"])
