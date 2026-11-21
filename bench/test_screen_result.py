"""A-15.4: the screen timing JSON reports equal work and labels every figure."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bench.screen import main


def test_equal_work_json_labels_every_figure(tmp_path):
    out = tmp_path / "screen.json"
    first = main(["--patients", "30", "--candidates", "12", "--B", "3", "--studies", "4", "--out", str(out)])
    assert json.loads(out.read_text()) == first
    fp = first["workload"]["fingerprints"]
    assert fp["B"] == 3 and all(len(fp[k]) == 64 for k in ("patient_rows", "X", "C"))
    assert first["data"] == "SYNTHETIC" and first["runs"]["cpu"]["label"] == "MEASURED"
    gpu = first["runs"]["gpu"]
    assert gpu["label"] in ("MEASURED", "UNMEASURED")
    assert (first["equal_work_speedup"]["label"] == "MEASURED") == (gpu["label"] == "MEASURED")
    assert "value" in first["equal_work_speedup"] or gpu["label"] == "UNMEASURED"
    assert first["runs"]["cpu"]["hardware"]["affinity"] >= 1 and "blas" in first["runs"]["cpu"]["hardware"]
    batched = first["batched"]
    assert batched["studies"] == 4 and batched["runs"]["cpu"]["label"] == "MEASURED"
    assert (batched["equal_work_speedup"]["label"] == "MEASURED") == (batched["runs"]["gpu"]["label"] == "MEASURED")


def _double(shift):
    """A screen result test double; shift != 0 makes the 'GPU' disagree on z and the null."""
    def timed(args, B, backend):
        d = 0.0 if backend == "numpy" else shift
        return 1.0, {"U": np.array([1.0, 2.0]), "V": np.array([1.0, 1.0]), "z": np.array([1.0, 2.0]) + d,
                     "null_maxT": np.array([1.0]) + d, "adjusted_p": np.array([1.0, 0.5]), "p": 0.5,
                     "maxT": 2.0 + d, "metadata": {"statistic": "test double", "clinical": {"shape": [8, 1]},
                                                   "null": {"name": "test double"}}}
    return timed


@pytest.mark.parametrize("shift", [0.0, 8.0])
def test_speedup_is_withheld_unless_every_timed_output_matches(tmp_path, monkeypatch, shift):
    import bench.screen as bench

    monkeypatch.setattr(bench, "gpu_backend", lambda: "cupy")
    monkeypatch.setattr(bench, "_time", _double(shift))
    fake = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(
        getDeviceProperties=lambda _n: {"name": b"TEST-DOUBLE-NOT-A-GPU"})))
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setattr(bench, "_time_batch", lambda X, T, E, C, s, backend: (1.0, {
        "U": np.zeros((2, 2)), "V": np.ones((2, 2)), "z": np.zeros((2, 2)), "maxT": np.zeros(2)}))
    result = bench.main(["--patients", "8", "--candidates", "2", "--B", "1", "--studies", "2",
                         "--out", str(tmp_path / "r.json")])
    speed = result["equal_work_speedup"]
    if shift:
        assert result["runs"]["gpu"]["matches_cpu"] is False
        assert speed["label"] == "WITHHELD" and "value" not in speed
        assert set(result["runs"]["gpu"]["mismatched"]) == {"z", "null_maxT"}
    else:
        assert speed["label"] == "MEASURED" and speed["value"] == 1.0
    assert result["batched"]["equal_work_speedup"]["label"] == "MEASURED"


def test_a_missing_output_fails_parity():
    from bench.screen import mismatched

    assert mismatched({"z": np.ones(2)}, {}, ("z",)) == ["z"]
    assert mismatched({"z": np.ones(2)}, {"z": np.ones(3)}, ("z",)) == ["z"]
