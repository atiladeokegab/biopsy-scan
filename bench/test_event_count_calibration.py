"""Calibration conditions on events without admitting evaluation maxima to its fitted cutoffs."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bench.event_count_calibration import (
    analyze,
    fit_thresholds,
    paired_difference,
    rate,
    reject,
)


def calibration():
    return [{"stream": "calibration", "i": i, "events": 20, "maxT": float(i + 1)} for i in range(20)] + [
        {"stream": "calibration", "i": i + 20, "events": 60, "maxT": float(i + 100)} for i in range(20)]


def test_cutoffs_condition_on_event_bin_instead_of_pooling_every_study():
    table = fit_thresholds(calibration())
    # Each bin has 20 calibration draws: ceil(21 * .95) = 20, its largest maximum.
    assert table["bins"]["low"]["threshold"] == pytest.approx(20.0)
    assert table["bins"]["high"]["threshold"] == pytest.approx(119.0)
    assert table["bins"]["low"]["n"] == table["bins"]["high"]["n"] == 20


def test_rank_boundary_ties_and_sparse_or_unseen_bins_never_reject():
    table = fit_thresholds(calibration())
    row = {"stream": "null", "i": 0, "events": 38, "maxT": 20.0}
    assert not reject(table, row) and reject(table, row | {"maxT": 20.1})
    assert not reject(table, row | {"events": 39, "maxT": 118.9})
    assert not reject(table, row | {"events": 159, "maxT": 119.0})
    assert reject(table, row | {"events": 0, "maxT": 21.0})
    sparse = fit_thresholds(calibration()[:18])
    assert sparse["bins"]["low"]["rank"] == 19
    assert sparse["bins"]["low"]["threshold"] is None
    assert sparse["bins"]["high"]["n"] == 0 and sparse["bins"]["high"]["threshold"] is None
    assert not reject(sparse, row | {"maxT": 1000000.0})
    assert not reject(sparse, row | {"events": 159, "maxT": 1000000.0})
    exact = fit_thresholds(calibration(), "exact")
    assert not reject(exact, row | {"events": 21, "maxT": 1000000.0})
    assert reject(exact, row | {"events": 20, "maxT": 20.1})
    json.dumps(sparse, allow_nan=False)


@pytest.mark.parametrize("change", [{"events": -1}, {"events": 160}, {"events": 2.5}, {"events": True},
                                    {"maxT": float("nan")}, {"maxT": float("inf")}, {"maxT": -1},
                                    {"stream": "null"}, {"stream": "alternative"}, {"i": -1}, {"i": True}])
def test_invalid_or_evaluation_rows_are_refused_during_calibration(change):
    rows = calibration()
    rows[0] |= change
    with pytest.raises(ValueError):
        fit_thresholds(rows)


def test_duplicate_ids_refused_and_row_order_cannot_change_thresholds():
    rows = calibration()
    with pytest.raises(ValueError):
        fit_thresholds(rows + [rows[0]])
    assert fit_thresholds(rows) == fit_thresholds(rows[::-1])


def evaluation():
    return [{"stream": s, "i": i, "events": e, "maxT": v, "lab_p": p}
            for s in ("null", "alternative") for i, (e, v, p) in enumerate([(20, 21.0, 0.01), (60, 119.0, 0.5)])]


def test_evaluation_changes_cannot_refit_thresholds_and_differences_are_paired():
    records = calibration() + evaluation()
    report = analyze(records)
    altered = copy.deepcopy(records)
    for r in altered:
        if r["stream"] != "calibration":
            r["maxT"], r["lab_p"] = 1000000.0, 0.0
    other = analyze(altered[::-1])
    for mode in ("pooled", "two_bins", "exact"):
        assert report["methods"][mode]["calibration"] == other["methods"][mode]["calibration"]
    paired = report["methods"]["two_bins"]["paired_vs_pooled"]["alternative"]
    assert (paired["wins"], paired["losses"], paired["difference"]) == (1, 1, 0.0)
    assert report["methods"]["lab"]["power"]["k"] == 1
    assert paired_difference([True, False], [True, False])["paired_bootstrap95"] == [0.0, 0.0]


def test_missing_ids_or_lab_pairs_do_not_produce_a_normal_report():
    records = calibration() + evaluation()
    with pytest.raises(ValueError, match="missing"):
        analyze([r for r in records if (r["stream"], r["i"]) != ("null", 0)])
    records[-1].pop("lab_p")
    with pytest.raises(ValueError, match="lab p"):
        analyze(records)


def test_wilson_matches_an_independent_library_and_handles_endpoints():
    from statsmodels.stats.proportion import proportion_confint

    for k, n in ((0, 20), (20, 20), (46, 1000), (153, 1000)):
        np.testing.assert_allclose(rate([True] * k + [False] * (n - k))["wilson95"],
                                   proportion_confint(k, n, method="wilson"), atol=1e-15)


def test_committed_result_rebuilds_from_its_records_without_real_data():
    path = Path(__file__).with_name("results") / "event-count-calibration.json"
    saved = json.loads(path.read_text())
    rebuilt = analyze(saved["records"])
    assert all(saved[k] == v for k, v in rebuilt.items())
    assert saved["labels"] == {"outcomes": "SYNTHETIC", "power": "ESTIMATE", "fwer": "ESTIMATE", "counts_and_compute": "MEASURED"}
    assert saved["execution"]["modal_runs"] == 0
    assert [(saved["methods"][m]["fwer"]["k"], saved["methods"][m]["power"]["k"]) for m in ("pooled", "lab")] == [(46, 153), (50, 203)]
