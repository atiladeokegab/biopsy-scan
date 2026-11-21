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


def test_hybrid_parity_split_and_headline():
    z = {"calibration": [[1.0, 2.0]], "null": [[0.5, 3.0]]}
    arms = {"numba": _arm("numba", z), "gpu": _arm("gpu", z), "hybrid": _arm("hybrid", z)}
    par = db.parity(arms)
    assert par["all"] and par["z_hybrid_vs_numba"]["compared"] == 2
    arms["hybrid"]["records"][1]["z"] = np.array([0.5, 3.1])
    assert not db.parity(arms)["all"]
    prior = {"gpu": {"studies": [{"stream": "null", "i": 0, "x_sha": "null0", "maxT": 3.0}]}}
    assert db.prior_parity(_arm("hybrid", z)["records"], prior)["all"]
    assert not db.prior_parity(arms["hybrid"]["records"], prior)["all"]  # maxT 3.1 against 3.0
    s = {"hybrid": {"split": db.split("hybrid", {"studies_wall_s": 9.0, "scoring_wall_s": 1.0}),
                    "stages": {"wall_per_study_s": 0.1}},
         "numba": {"split": db.split("numba", {"reimage_worker_wall_s": 3.0, "scoring_worker_wall_s": 1.0}),
                   "stages": {"wall_per_study_s": 0.2}}}
    assert s["hybrid"]["split"]["reimage_share"] == 0.9
    assert db.headline(s).startswith("Re-imaging dominates every arm")


def test_legacy_stage_names_are_renamed_and_the_end_to_end_adds_every_recorded_stage():
    old = {"prep_s": 10.0, "adjacency_s": 5.0, "reimage_cpu_s": 80.0, "scoring_cpu_s": 2.0, "studies_wall_s": 12.0,
           "evaluated_studies": 4}
    st = db.complete(old, load_s=1.0, warm_s=2.0, warm_score_s=None)
    assert "reimage_cpu_s" not in st and st["reimage_worker_wall_s"] == 80.0 and st["scoring_worker_wall_s"] == 2.0
    assert st["compute_wall_s"] == 12.0 and st["wall_per_study_s"] == 3.0 and st["end_to_end_in_container_s"] == 30.0
    assert db.split("numba", st)["reimage_share"] == 80 / 82
