"""Bench harness: a tiny offline run; every number carries its hardware, pending rungs carry none."""
import json

from bench import chart, ladder


def test_tiny_run_writes_json_and_chart(tmp_path):
    path = ladder.main(["--out", str(tmp_path), "--trials", "2", "--cells", "300"])
    res = json.loads(path.read_text())
    assert res["hardware"]["cpu"] and res["hardware"]["logical_cpus"]
    ref = res["rungs"]["numpy_reference"]
    assert ref["trials"] == 2 and ref["neighbour_reads"] > 0 and ref["amortised_s_per_trial"] > 0
    assert all(isinstance(v, (dict, str)) for v in res["rungs"].values())
    assert all(v.startswith("pending") for v in res["rungs"].values() if isinstance(v, str))
    s = chart.svg(res)
    assert s.startswith("<svg") and res["hardware"]["cpu"] in s and "numpy_reference" in s


def test_ratios_are_against_the_named_reference_not_the_slowest_bar():
    hw = {"cpu": "test cpu", "logical_cpus": 1, "gpu": None}
    unit = {"trial": "t", "excludes": "e"}
    res = {"hardware": hw, "data": "synthetic", "unit": unit, "baseline": "numpy_reference",
           "rungs": {"numpy_reference": {"trials_per_s": 10.0}, "numba_all_core": {"trials_per_s": 5.0}}}
    s = chart.svg(res)
    assert "(1x vs numpy_reference)" in s and "(0.5x vs numpy_reference)" in s and "(2x" not in s
