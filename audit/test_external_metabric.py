import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit import external_metabric as metabric
from audit import external_validation as validation


def test_geometry_preserves_selected_order_and_exact_phenotype_roles():
    selection = pd.DataFrame([
        ("MB-0020", 8, 500, 600), ("MB-0001", 4, 470, 520),
    ], columns=[metabric.PATIENT, metabric.IMAGE, "width", "height"])
    cells = pd.DataFrame([
        ("MB-0001", 4, 1, 12., 4., 1, 1, 0, 0, "Basal"),
        ("MB-0020", 8, 2, 19., 4., 0, 1, 0, 0, " CD8^{+} T cells "),
        ("MB-0020", 8, 3, 20., 4., 0, 1, 0, 0, "CD4^{+} T cells & APCs"),
        ("MB-0020", 8, 1, 10., 4., 1, 1, 0, 0, "Basal"),
    ], columns=[metabric.PATIENT, metabric.IMAGE, metabric.CELL, *metabric.XY, *metabric.FLAGS, metabric.PHENOTYPE])
    result = metabric._prepare_cells(cells, selection)
    np.testing.assert_array_equal(result["region_id"], [0, 0, 0, 1])
    np.testing.assert_array_equal(result["cell_type"], [0, 1, 2, 0])
    np.testing.assert_array_equal(result["image_size_um"], [[500, 600], [470, 520]])
    cells.loc[0, metabric.XY[0]] = 470.
    with pytest.raises(ValueError, match="outside"):
        metabric._prepare_cells(cells, selection)


def test_endpoint_alignment_preserves_breslow_strata_and_exact_membership():
    frame = pd.DataFrame([("MB-2", 3.0, 1, "neg"), ("MB-1", 5.0, 0, "pos")],
                         columns=[metabric.PATIENT, "yearsToStatus", "DeathBreast", "ERStatus"])
    time, event, strata = metabric.align_endpoints(frame, ["MB-1", "MB-2"])
    np.testing.assert_array_equal(time, [5, 3])
    np.testing.assert_array_equal(event, [False, True])
    np.testing.assert_array_equal(strata, [1, 0])
    with pytest.raises(ValueError, match="exactly"):
        metabric.align_endpoints(frame, ["MB-1", "MB-3"])
    frame.loc[0, "ERStatus"] = "unknown"
    with pytest.raises(ValueError, match="stratum"):
        metabric.align_endpoints(frame, ["MB-1", "MB-2"])


def test_block_draws_leave_capacity_and_keep_the_arm_event_floor():
    orders, _ = validation.acquisition_schedule(100, 1)
    result = metabric._pilot_gate(orders, np.ones(100, dtype=bool))
    assert result == {"event_eligible_pairs": 200, "block_event_eligible_pairs": [200] * 10,
                      "dense_floor_pass": True, "passed": True}
    empty = metabric._pilot_gate(orders, np.zeros(100, dtype=bool))
    assert empty["event_eligible_pairs"] == 0 and not empty["passed"]


def test_reserved_table_stays_unread_without_matching_checkpoint(tmp_path, monkeypatch):
    invalid_receipt = tmp_path / "receipt.json"
    invalid_receipt.write_text(json.dumps({"protocol_sha256": "x", "split_sha256": "y", "pilot_sha256": "z"}))
    manifest, protocol, pilot = (tmp_path / name for name in ("split.json", "protocol.md", "pilot.json"))
    for source in (manifest, protocol, pilot):
        source.write_text("{}")

    def unexpected_read(*args, **kwargs):
        pytest.fail("coded data read before the exact opening receipt")

    monkeypatch.setattr(metabric.pd, "read_csv", unexpected_read)
    absent = Path("absent.csv")
    with pytest.raises(ValueError, match="receipt does not match"):
        metabric.run(absent, absent, absent, manifest, protocol, absent, invalid_receipt, pilot)


def test_v2_refits_common_coefficient_with_aligned_er_strata(monkeypatch):
    n = 100
    orders, _ = validation.acquisition_schedule(n, 2)
    strata = np.arange(n, dtype=int) % 2
    times = np.arange(1, n + 1)
    dense = np.ones(n)
    crops = np.empty((2, validation.DRAWS, n))
    crops[0] = 1.25
    crops[1] = 2.0
    fitted_sizes = set()

    def fit(x, time, event, groups):
        np.testing.assert_array_equal(groups, (time - 1) % 2)
        fitted_sizes.add(len(x))
        return float(np.mean(x)), 0.05

    monkeypatch.setattr(validation, "_cox", fit)
    result = validation.compare(dense, crops, orders, times, np.ones(n, dtype=bool), 2, (0., 1.),
                                patients=metabric.DESIGN, strata=strata)
    assert result["valid_pairs"] == 200 and fitted_sizes == {45, 90, 100}
    assert result["leave_block_out"] == [-0.75] * 10
    assert result["arms"]["A"]["area_mm2"] == result["arms"]["B"]["area_mm2"] == 8.1
