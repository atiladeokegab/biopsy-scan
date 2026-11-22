import json
from pathlib import Path

import pytest

from bench.plot_h100_design import measured_points

RESULTS = Path(__file__).parent / "results"
ORIGINAL = RESULTS / "modal-h100-design-scaling-20261122T134234Z.json"
ISOLATED = RESULTS / "modal-h100-design-scaling-20261122T140815Z.json"


def test_cross_launch_receipts_accept_once():
    plan, points = measured_points([ORIGINAL, ISOLATED])
    assert sum(plan.values()) == 3000
    assert {size: len(runs) for size, runs in points.items()} == {1: 1, 2: 1, 8: 1}
    point, = points[8]
    assert point["studies_per_hour_compute_span"] == pytest.approx(10_800_000 / point["compute_span_s"])


def test_copy_of_valid_receipt_cannot_count_as_a_repeat(tmp_path):
    alias = tmp_path / "copy.json"
    alias.write_bytes(ISOLATED.read_bytes())
    with pytest.raises(ValueError, match="duplicate worker task IDs"):
        measured_points([ISOLATED, alias])


@pytest.mark.parametrize(("field", "value"), [
    ("end_to_end_wall_s", -1),
    ("end_to_end_wall_s", float("inf")),
    ("compute_span_s", 0),
    ("studies_per_hour_compute_span", 11),
    ("wall_s_per_evaluated_study", 12),
])
def test_rejects_invalid_reported_timing(tmp_path, field, value):
    data = json.loads(ISOLATED.read_text())
    data["repeats"][0]["8"][field] = value
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="invalid compute windows or wall time|recorded timing disagrees"):
        measured_points([path])


def test_rejects_windows_without_common_overlap(tmp_path):
    data = json.loads(ISOLATED.read_text())
    workers = data["repeats"][0]["8"]["workers"]
    start = max(w["compute_window"][1] for w in workers)
    workers[0]["compute_window"] = [start, start + 1]
    path = tmp_path / "disjoint.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="invalid compute windows"):
        measured_points([path])
