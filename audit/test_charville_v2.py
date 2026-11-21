"""Offline validation and schema checks for audit/charville_v2.py and results/charville-v2.json (Req: A-12.3)."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from audit.charville_v2 import _code_revision, run_charville_v2
from audit.test_audit import _fake_stats_api
from core.grammar import EXPRESSION_CHANNELS, EXPRESSION_TRANSFORM, grammar, grammar_hash, grammar_v2

ROOT = Path(__file__).resolve().parents[1]
IN_GIT = (ROOT / ".git").exists()  # a file in a worktree, a directory in a clone, absent in GitHub's Download ZIP
NO_GIT = "no .git (e.g. GitHub's Download ZIP): code_revision is labelled 'unknown', checked in the offline contract"


def test_run_charville_v2_offline_contract(tmp_path):
    rng = np.random.default_rng(1)
    type_names = ["B cell", "Stroma", "Tumor 1"]
    n_p, n_reg, n_per = 6, 6, 15
    cells = {
        "cell_type": rng.integers(0, 3, n_reg * n_per, dtype=np.int8),
        "x": rng.random(n_reg * n_per, dtype=np.float32) * 120.0,
        "y": rng.random(n_reg * n_per, dtype=np.float32) * 120.0,
        "channels": rng.normal(0.0, 1.0, (n_reg * n_per, len(EXPRESSION_CHANNELS))).astype(np.float32),
        "channel_stats": {
            "channels": list(EXPRESSION_CHANNELS),
            "transform": EXPRESSION_TRANSFORM,
            "mean": [0.0] * len(EXPRESSION_CHANNELS),
            "std": [1.0] * len(EXPRESSION_CHANNELS),
        },
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
        seed=7,
        load_fn=lambda cohort, include_channels=True: cells,
        perm_fn=fake_stats.permutation_test,
        score_fn=fake_stats.score,
    )
    saved = json.loads(out_file.read_text())
    assert saved["exploratory"] is True and saved["already_seen"] is True and saved["clinical_claims"] is False
    assert saved["seed"] == 7
    assert len(saved["source_fingerprint"]) == 64
    if IN_GIT:
        assert saved["code_revision"] != "unknown"
    else:  # no checkout to read: provenance must say so, never guess
        assert saved["code_revision"] == "unknown"
    assert saved["normalization_stats"]["channels"] == list(EXPRESSION_CHANNELS)
    if saved["maxT"]["p"] > 0.05 and saved["omnibus"]["p"] > 0.05:
        assert saved["prepared_outcome"]["key"] == "exploratory_joint_search_nonsignificant"
    assert saved["grammar_v1_hash"] == grammar_hash(grammar(type_names))
    assert saved["grammar_v2_hash"] == grammar_hash(grammar_v2(type_names))
    assert saved["m"] == 10 and saved["transform"] == EXPRESSION_TRANSFORM
    assert saved["n_features_total"] == 1 * 3 * 3 * (2 + 10) == 108
    assert saved["labels"]["maxT"] == "MEASURED"
    assert saved["labels"]["gpu_timing"] == "UNMEASURED"
    assert saved["maxT"]["top_hits"][0]["family"] in {"v1", "v2"}
    assert set(saved["labels"].values()) <= {"MEASURED", "UNMEASURED"}


@pytest.mark.skipif(not IN_GIT, reason=NO_GIT)
def test_code_revision_is_the_checkout_head():
    assert re.fullmatch(r"[0-9a-f]{40}", _code_revision())


def test_offline_contract_passes_in_a_copy_without_git(tmp_path):
    """A judge's Download ZIP: the same tests in a copy with no .git pass, with the revision labelled unknown."""
    copy = tmp_path / "zip"
    for d in ("audit", "core", "stats"):
        shutil.copytree(ROOT / d, copy / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "pyproject.toml", copy)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)  # git must not find a repository above the copy either
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
         "audit/test_charville_v2.py::test_run_charville_v2_offline_contract",
         "audit/test_charville_v2.py::test_code_revision_is_the_checkout_head"],
        cwd=copy, env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    assert "1 passed, 1 skipped" in res.stdout and "Download ZIP" in res.stdout, res.stdout[-2000:]


def test_committed_charville_v2_results_artifact_valid():
    artifact = Path(__file__).resolve().parents[1] / "results" / "charville-v2.json"
    saved = json.loads(artifact.read_text())
    assert saved["cohort"] == "charville"
    assert saved["seed"] == 0
    assert len(saved["source_fingerprint"]) == 64
    assert isinstance(saved["code_revision"], str) and len(saved["code_revision"]) >= 7
    assert len(saved["normalization_stats"]["mean"]) == 10 and len(saved["normalization_stats"]["std"]) == 10
    assert saved["prepared_outcome"]["key"] != "atlas_design_choices"
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
