"""Offline validation and schema checks for audit/charville_v2.py and results/charville-v2.json (Req: A-12.3)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit.charville_v2 import run_charville_v2
from audit.test_audit import _fake_stats_api
from core.grammar import EXPRESSION_CHANNELS, EXPRESSION_TRANSFORM, grammar, grammar_hash, grammar_v2


def test_run_charville_v2_offline_contract(tmp_path):
    rng = np.random.default_rng(1)
    type_names = ["B cell", "Stroma", "Tumor 1"]
    n_p, n_reg, n_per = 6, 6, 15
    cells = {
        "cell_type": rng.integers(0, 3, n_reg * n_per, dtype=np.int8),
        "x": rng.random(n_reg * n_per, dtype=np.float32) * 120.0,
        "y": rng.random(n_reg * n_per, dtype=np.float32) * 120.0,
        "channels": rng.normal(0.0, 1.0, (n_reg * n_per, len(EXPRESSION_CHANNELS))).astype(np.float32),
        "region_id": np.repeat(np.arange(n_reg, dtype=np.int32), n_per),
        "patient_of_region": np.arange(n_p, dtype=np.int32),
        "time": np.linspace(10.0, 60.0, n_p),
        "event": np.array([1, 0, 1, 0, 1, 0], dtype=bool),
        "strata": np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        "type_names": type_names,
        "region_names": [f"Charville_c00{1 + i // 3}_v001_r001_reg00{i}" for i in range(n_reg)],
        "patient_names": [f"P{i}" for i in range(n_p)],
    }

    out_file = tmp_path / "charville-v2.json"
    fake_stats = _fake_stats_api()
    rec = run_charville_v2(
        out_path=out_file,
        n_resamples=29,
        load_fn=lambda cohort, include_channels=True: cells,
        perm_fn=fake_stats.permutation_test,
        score_fn=fake_stats.score,
    )
    saved = json.loads(out_file.read_text())
    assert saved["exploratory"] is True and saved["already_seen"] is True and saved["clinical_claims"] is False
    assert saved["grammar_v1_hash"] == grammar_hash(grammar(type_names))
    assert saved["grammar_v2_hash"] == grammar_hash(grammar_v2(type_names))
    assert saved["m"] == 10 and saved["transform"] == EXPRESSION_TRANSFORM
    assert saved["n_features_total"] == 1 * 3 * 3 * (2 + 10) == 108
    assert saved["labels"]["maxT"] == "MEASURED"
    assert saved["labels"]["gpu_timing"] == "UNMEASURED"
    assert saved["maxT"]["top_hits"][0]["family"] in {"v1", "v2"}
    assert set(saved["labels"].values()) <= {"MEASURED", "UNMEASURED"}


def test_committed_charville_v2_results_artifact_valid():
    artifact = Path(__file__).resolve().parents[1] / "results" / "charville-v2.json"
    saved = json.loads(artifact.read_text())
    assert saved["cohort"] == "charville"
    assert saved["n_patients"] == 159 and saved["n_events"] == 38 and saved["n_regions"] == 286
    assert saved["exploratory"] is True and saved["already_seen"] is True and saved["clinical_claims"] is False
    assert saved["grammar_v1_hash"] == "772f6a845266470456af96f92d0573cc1bec72432ad74d789439baff066c2156"
    assert saved["grammar_v2_hash"] != saved["grammar_v1_hash"]
    assert saved["m"] == 10 and saved["n_features_total"] == 3780
    assert saved["transform"] == EXPRESSION_TRANSFORM
    assert saved["maxT"]["n_resamples"] == 9999
    assert set(saved["labels"].values()) <= {"MEASURED", "UNMEASURED"}
    assert saved["runtime"]["status"] == "MEASURED"
    assert saved["gpu_timing"]["status"] == "UNMEASURED"
