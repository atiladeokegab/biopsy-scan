import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit import external_validation as validation
from audit.external_basel import EVENT, GEOMETRY, SIDES_UM, align_outcomes, prepare_cells, private_path
from core.truth import truth_region
from scripts.check_external_validation_split import split_patients


def test_import_selects_one_fitting_image_and_preserves_cluster_roles():
    frame = pd.DataFrame([
        ("Basel", "P1", "too-short", 1, 900, 400, 10, 10, 26),
        ("Basel", "P1", "tie-B", 1, 500, 500, 10, 10, 26),
        ("Basel", "P1", "tie-A", 1, 500, 500, 10, 10, 14),
        ("Basel", "P1", "tie-A", 2, 500, 500, 20, 10, 5),
        ("Basel", "P1", "tie-A", 3, 500, 500, 30, 10, 27),
        ("Basel", "P1", "smaller", 1, 490, 490, 10, 10, 5),
        ("Basel", "P2", "not-fitting", 1, 300, 300, 10, 10, 3),
    ], columns=GEOMETRY)
    cells = prepare_cells(frame)
    assert cells["patient_names"] == ["P1"]
    assert cells["region_names"] == ["tie-A"]
    assert cells["cell_type"].tolist() == [0, 1, 2]


def test_endpoint_event_coding_and_exact_join():
    frame = pd.DataFrame({"patient_id": ["P2", "P1"], "patient_status": ["death", EVENT],
                          "patient_OS_months": [12.5, 24.0]})
    time, event = align_outcomes(frame, ["P1", "P2"])
    assert time.tolist() == [24.0, 12.5]
    assert event.tolist() == [True, False]
    with pytest.raises(ValueError, match="exactly"):
        align_outcomes(frame, ["P1", "P3"])


def test_sparse_feature_matches_truth_and_nested_rng_for_both_arms(monkeypatch):
    monkeypatch.setattr(validation, "DRAWS", 3)
    xy = np.array([[10., 10.], [20., 10.], [60., 10.], [80., 10.], [200., 200.]])
    types = np.array([0, 1, 2, 1, 0])
    cells = {"region_id": np.repeat(np.arange(3), 5),
             "patient_of_region": np.arange(3), "type_names": ["Tumor", "T cell", "Other"],
             "x": np.tile(xy[:, 0], 3), "y": np.tile(xy[:, 1], 3), "cell_type": np.tile(types, 3),
             "image_size_um": np.full((3, 2), SIDES_UM[1])}
    orders, uniforms = validation.acquisition_schedule(3, 2)
    dense, crops = validation.acquisition_features(cells, [0, 1, 2], uniforms)
    expected, focal = truth_region(xy, types, [0], 3)
    target = expected[0, 1, 1, 0] / focal[0]
    np.testing.assert_allclose(dense, target)
    np.testing.assert_allclose(crops[1], target)
    for draw in range(3):
        rng = validation._rng(2, draw)
        np.testing.assert_array_equal(orders[draw], rng.permutation(3))
        for patient, uniform in enumerate(rng.random((3, 2))):
            crop = (*tuple(uniform * (SIDES_UM[1] - SIDES_UM[0])), SIDES_UM[0])
            visible = validation.crop_mask(xy, crop)
            expected, focal = truth_region(xy[visible], types[visible], [0], 3)
            if focal[0]:
                assert crops[0, draw, patient] == expected[0, 1, 1, 0] / focal[0]
            else:
                assert np.isnan(crops[0, draw, patient])


def test_opening_receipt_rejected_before_outcomes_are_read(tmp_path, monkeypatch):
    geometry, manifest, protocol, pilot, receipt = (tmp_path / name for name in
                                                   ("geometry", "manifest", "protocol", "pilot", "receipt"))
    for path in (geometry, manifest, protocol):
        path.write_text("{}")
    protocol_hash = validation._sha256(protocol)
    monkeypatch.setattr(validation, "PROTOCOL_SHA256", protocol_hash)
    provenance = {"protocol_sha256": protocol_hash, "split_sha256": validation._sha256(manifest),
                  "geometry_sha256": validation._sha256(geometry), "source_doi": validation.SOURCE_DOI,
                  "source_md5": validation.SOURCE_MD5}
    pilot.write_text(json.dumps({"partition": "pilot", "provenance": provenance}))
    receipt.write_text(json.dumps({"protocol_sha256": protocol_hash, "split_sha256": "wrong"}))
    with pytest.raises(ValueError, match="receipt does not match"):
        validation.run(geometry, manifest, protocol, Path("absent-endpoints.csv"), receipt, pilot)


def test_private_path_distinguishes_a_checkout_from_an_empty_sandbox_marker(tmp_path):
    (tmp_path / ".git").mkdir()
    file = tmp_path / "coded.parquet"
    assert private_path(file) == file
    (tmp_path / ".git" / "HEAD").touch()
    with pytest.raises(ValueError, match="outside Git"):
        private_path(file)


def test_sparse_pilot_stops_before_feature_extraction(tmp_path, monkeypatch):
    geometry, manifest, protocol, outcomes = (tmp_path / name for name in
                                              ("geometry.parquet", "split.json", "protocol.md", "pilot.csv"))
    ids = [f"P{i}" for i in range(201)]
    pilot, reserved = split_patients(ids)
    manifest.write_text(json.dumps({"eligible": ids, "pilot": pilot, "reserved": reserved}))
    protocol.write_text("synthetic protocol")
    monkeypatch.setattr(validation, "PROTOCOL_SHA256", validation._sha256(protocol))
    pd.DataFrame([("Basel", patient, patient, 1, 500, 500, 100, 100, 14) for patient in ids],
                 columns=GEOMETRY).to_parquet(geometry)
    pd.DataFrame({"patient_id": pilot, "patient_status": [EVENT] * 4 + ["alive"] * (len(pilot) - 4),
                  "patient_OS_months": 12}).to_csv(outcomes, index=False)

    def unexpected_fit(*args):
        pytest.fail("an infeasible pilot should stop before loading any features")

    monkeypatch.setattr(validation, "acquisition_features", unexpected_fit)
    result = validation.run(geometry, manifest, protocol, outcomes)
    assert result["event_eligible_pair_ceiling"] == 0
    assert result["decision"] == "inconclusive"
    assert result["preferred"] is None


def test_comparison_keeps_same_scale_and_direction_in_patient_block_refits(monkeypatch):
    n = 67
    orders, _ = validation.acquisition_schedule(n, 2)
    dense = np.full(n, 11.0)
    crops = np.empty((2, validation.DRAWS, n))
    crops[0] = 11.25
    crops[1] = 12.0

    def fit_with_known_coefficient(x, time, event, strata):
        return float(x.mean()), 0.05

    monkeypatch.setattr(validation, "_cox", fit_with_known_coefficient)
    result = validation.compare(dense, crops, orders, np.arange(1, n + 1), np.ones(n, dtype=bool), 2, (10., 1.))
    assert result["valid_pairs"] == 200
    assert result["dense"]["log_hr"] == 1.0
    assert result["mean_error_a_minus_b"] == -0.75
    assert result["leave_block_out"] == [-0.75] * 10
    assert result["patient_ci95"] == [-0.75, -0.75]
    assert validation._supports(result, "A") == "supports pilot recommendation"
