"""A-15.4: the screen timing JSON reports equal work and labels every figure."""

import json

from bench.screen import main


def test_equal_work_json_labels_every_figure(tmp_path):
    out = tmp_path / "screen.json"
    r14 = tmp_path / "r14.json"
    first = main(["--patients", "30", "--candidates", "12", "--B", "3", "--out", str(out)])
    assert json.loads(out.read_text()) == first
    fp = first["workload"]["fingerprints"]
    assert fp["B"] == 3 and all(len(fp[k]) == 64 for k in ("patient_rows", "X", "C"))
    assert first["data"] == "SYNTHETIC" and first["runs"]["cpu"]["label"] == "MEASURED"
    gpu = first["runs"]["gpu"]
    assert gpu["label"] in ("MEASURED", "UNMEASURED")
    assert (first["equal_work_speedup"]["label"] == "MEASURED") == (gpu["label"] == "MEASURED")
    assert "value" in first["equal_work_speedup"] or gpu["label"] == "UNMEASURED"
    assert first["workflow_comparison"]["label"] == "UNMEASURED"

    r14.write_text(json.dumps({"fingerprints": {**fp, "X": "0" * 64}, "seconds": 9.0}))
    mismatched = main(["--patients", "30", "--candidates", "12", "--B", "3", "--out", str(out), "--r14", str(r14)])
    assert mismatched["workflow_comparison"]["label"] == "UNMEASURED"

    r14.write_text(json.dumps({"fingerprints": fp, "seconds": 9.0}))
    matched = main(["--patients", "30", "--candidates", "12", "--B", "3", "--out", str(out), "--r14", str(r14)])
    assert matched["workflow_comparison"]["label"] == "MEASURED"
    assert "workflow comparison" in matched["workflow_comparison"]["kind"]
