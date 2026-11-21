"""A-14.2: the baseline's results JSON names its machine, inputs, outputs and MEASURED wall time."""
import json

import numpy as np
import pytest

from bench import impl_a
from bench.test_impl_a import _fixture


def _report(tmp_path, seed=0):
    time, event, strata, C, X = _fixture(seed)
    ids = [f"p{i}" for i in range(len(time))]
    rep = impl_a.report([impl_a.measure("fixture", ids, time, event, strata, C, ["hpv"], X,
                                        [f"f{j}" for j in range(X.shape[1])], cores=2)], cores=2)
    path = tmp_path / "impl-a.json"
    path.write_text(json.dumps(rep))
    return json.loads(path.read_text())


def test_results_json_names_machine_inputs_outputs_and_measured_time(tmp_path):
    rep = _report(tmp_path)
    assert rep["cpu_model"] and rep["cores_used"] == 2
    assert set(rep["versions"]) >= {"python", "numpy", "scipy", "statsmodels"}
    assert "workflow" in rep["comparison"]
    (c,) = rep["cohorts"]
    assert set(c["fingerprints"]) == {"patient_rows", "X", "C"}
    assert all(len(v) == 64 for v in c["fingerprints"].values())
    assert [t["workers"] for t in c["timings"]] == [1, 2]
    assert all(t["label"] == "MEASURED" and t["wall_s"] > 0 for t in c["timings"])
    assert len(c["outputs"]) == 7 and set(c["outputs"][0]) >= {"feature", "coef", "wald_p"}


def test_compare_only_with_matching_fingerprints(tmp_path):
    a, b = _report(tmp_path, 0)["cohorts"][0], _report(tmp_path, 0)["cohorts"][0]
    assert impl_a.compare(a, b)["label"] == "workflow comparison"
    with pytest.raises(ValueError, match="not comparable"):
        impl_a.compare(a, _report(tmp_path, 1)["cohorts"][0])


def test_fingerprint_changes_with_any_input():
    time, event, strata, C, X = _fixture()
    ids = [str(i) for i in range(len(time))]
    base = impl_a.fingerprints(ids, time, event, strata, C, X)
    X2 = X.copy()
    X2[0, 0] += 1e-12
    assert impl_a.fingerprints(ids, time, event, strata, C, X2)["X"] != base["X"]
    assert impl_a.fingerprints(ids[::-1], time, event, strata, C, X)["patient_rows"] != base["patient_rows"]
    assert np.isnan(X).any() and impl_a.fingerprint(X) == base["X"]  # NaN is hashed stably
