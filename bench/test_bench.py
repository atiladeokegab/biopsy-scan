"""Bench harness: a tiny offline run; every number carries its hardware, pending rungs carry none."""
import json

from bench import chart, ladder


def test_tiny_run_writes_json_and_chart(tmp_path):
    path = ladder.main(["--out", str(tmp_path), "--studies", "2", "--cells", "300"])
    res = json.loads(path.read_text())
    assert res["hardware"]["cpu"] and res["hardware"]["logical_cpus"]
    ref = res["rungs"]["numpy_reference"]
    assert ref["studies"] == 2 and ref["neighbour_reads"] > 0 and ref["amortised_s_per_study"] > 0
    assert all(isinstance(v, (dict, str)) for v in res["rungs"].values())
    assert all(v.startswith("pending") for v in res["rungs"].values() if isinstance(v, str))
    s = chart.svg(res)
    assert s.startswith("<svg") and res["hardware"]["cpu"] in s and "numpy_reference" in s
