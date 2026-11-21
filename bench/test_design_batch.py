"""bench.design_batch: Wilson intervals, disjoint seed streams, summary rules and the parity gate."""

import numpy as np
import pytest

from bench import design_batch as db


def test_wilson_matches_the_closed_form_and_edges():
    lo, hi = db.wilson(5, 100)
    assert lo == pytest.approx(0.021543, abs=1e-6) and hi == pytest.approx(0.111750, abs=1e-6)
    assert db.wilson(0, 20)[0] == 0.0 and db.wilson(20, 20)[1] == 1.0
    with pytest.raises(ValueError):
        db.wilson(3, 2)


def test_streams_are_disjoint_and_reproducible():
    seen = {s for stream in db.STREAMS for i in range(200) for s in db.seeds(stream, i)}
    assert len(seen) == 3 * 3 * 200
    assert db.seeds("null", 7) == db.seeds("null", 7)


def _arm(name, z_by_stream=None, p_by_stream=None):
    recs = []
    for stream in db.STREAMS:
        for i, v in enumerate((z_by_stream or p_by_stream).get(stream, [])):
            r = {"stream": stream, "i": i, "x_sha": f"{stream}{i}"}
            r.update({"z": np.asarray(v, float)} if z_by_stream else {"p": v})
            recs.append(r)
    return {"arm": name, "records": recs}


def test_batched_summary_uses_one_calibration_threshold_and_counts_identification():
    cal = [[0.0, i / 10] for i in range(1, 40)]  # maxT 0.1 .. 3.9; the 95% rank rule picks 3.8
    out = db.summarize(_arm("gpu", {"calibration": cal, "null": [[0, 1.0], [0, 3.85]],
                                    "alternative": [[5.0, 0], [0, 5.0], [1, 1]]}), plant=0)
    assert out["threshold"] == pytest.approx(3.8)
    assert out["fwer"]["k"] == 1 and out["power"]["k"] == 2 and out["identification"]["k"] == 1


def test_lab_summary_calls_on_p_and_parity_flags_any_x_or_z_mismatch():
    lab = db.summarize(_arm("lab", p_by_stream={"null": [0.01, 0.5], "alternative": [0.04, 0.05, 0.2]}), plant=0)
    assert lab["fwer"]["k"] == 1 and lab["power"]["k"] == 2
    z = {"null": [[1.0, 2.0]]}
    a, b = _arm("numba", z), _arm("gpu", {"null": [[1.0, 2.0 + 1e-6]]})
    assert db.parity({"numba": a, "gpu": _arm("gpu", z)})["all"]
    assert not db.parity({"numba": a, "gpu": b})["all"]
    b["records"][0]["z"] = np.array([1.0, 2.0])
    b["records"][0]["x_sha"] = "other"
    assert not db.parity({"numba": a, "gpu": b})["all"]
