"""Calibration conditions on events without admitting evaluation maxima to its fitted cutoffs."""

import pytest

from bench.event_count_calibration import fit_thresholds


def calibration():
    return [{"stream": "calibration", "i": i, "events": 20, "maxT": float(i + 1)} for i in range(20)] + [
        {"stream": "calibration", "i": i + 20, "events": 60, "maxT": float(i + 100)} for i in range(20)]


def test_cutoffs_condition_on_event_bin_instead_of_pooling_every_study():
    table = fit_thresholds(calibration())
    # Each bin has 20 calibration draws: ceil(21 * .95) = 20, its largest maximum.
    assert table["bins"]["low"]["threshold"] == pytest.approx(20.0)
    assert table["bins"]["high"]["threshold"] == pytest.approx(119.0)
    assert table["bins"]["low"]["n"] == table["bins"]["high"]["n"] == 20
