"""Bench harness: a tiny offline run; every number carries its hardware, pending rungs carry none."""
import json

import numpy as np

from bench import chart, fleet, ladder


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


def test_gate_rejects_empty_truncated_and_misshapen_outputs():
    S, m = np.ones((2, 3, 3, 2)), np.array([1, 2])
    ref = [(S, m), (S, m)]
    assert ladder.same_output(ref, [(S.copy(), m.copy()), (S.copy(), m.copy())])
    assert not ladder.same_output(ref, [])
    assert not ladder.same_output(ref, ref[:1])
    assert not ladder.same_output(ref, [(S[:1], m), (S, m)])
    assert not ladder.same_output(ref, [(S, m + 1), (S, m)])


def test_cuda_simulator_never_yields_a_gpu_rung(monkeypatch):
    monkeypatch.setenv("NUMBA_ENABLE_CUDASIM", "1")
    gpu = ladder.rungs()["gpu"]
    assert isinstance(gpu, str) and "simulator" in gpu


def test_ladder_devices_flag_records_request_and_never_fakes_gpu(tmp_path):
    path = ladder.main(["--out", str(tmp_path), "--trials", "2", "--cells", "200", "--devices", "2"])
    res = json.loads(path.read_text())
    assert res["devices_requested"] == 2
    gpu = res["rungs"]["gpu"]
    assert isinstance(gpu, dict) or gpu.startswith("pending")


def test_fleet_reports_1_2_4_8_measured_or_unmeasured_without_guessing(tmp_path, monkeypatch):
    path = fleet.main(["--out", str(tmp_path), "--trials", "2", "--cells", "120"])
    res = json.loads(path.read_text())
    rows = res["fleet"]
    assert [r["devices"] for r in rows] == [1, 2, 4, 8]
    for r in rows:
        assert r["status"] in {"MEASURED", "UNMEASURED"}
        if r["status"] == "UNMEASURED":
            assert r["studies_per_s"] is None and r["reason"]
        else:
            assert isinstance(r["studies_per_s"], float) and r["studies_per_s"] > 0

    monkeypatch.setenv("NUMBA_ENABLE_CUDASIM", "1")
    sim_res = fleet.run_fleet(studies=2, cells=120)
    assert all(r["status"] == "UNMEASURED" and "simulator" in r["reason"].lower() for r in sim_res["fleet"])


def test_fleet_requires_actual_executed_shard_count_and_device_names(monkeypatch):
    """Even with 8 mocked GPUs, 2 trials must mark 4 and 8 GPUs UNMEASURED, and missing names reject MEASURED."""
    monkeypatch.delenv("NUMBA_ENABLE_CUDASIM", raising=False)
    monkeypatch.setattr(fleet, "cuda_simulator", lambda: False)
    monkeypatch.setattr(fleet, "visible_gpus", lambda: 8)
    calls = []

    def fake_warm_sharded(design, warmup_seeds, warm_seeds, *, devices=None, fake_shards=False):
        del warmup_seeds, fake_shards
        calls.append((len(warm_seeds), devices))
        ref = fleet.reimage_run({**design, "backend": "numpy"}, warm_seeds)
        executed = min(len(warm_seeds), devices)
        return ref, {
            "devices_executed": executed,
            "device_names": [f"MockGPU-{i}" for i in range(executed)],
            "wall_s": 0.25,
            "cold_warmup_s": 0.5,
        }

    monkeypatch.setattr(fleet, "measure_warm_sharded", fake_warm_sharded)
    res = fleet.run_fleet(studies=2, cells=60, patients=4, regions_per_patient=1, k=1)
    by_dev = {r["devices"]: r for r in res["fleet"]}
    assert by_dev[1]["status"] == "MEASURED" and by_dev[1]["gpus"] == ["MockGPU-0"]
    assert by_dev[2]["status"] == "MEASURED" and by_dev[2]["gpus"] == ["MockGPU-0", "MockGPU-1"]
    assert by_dev[4]["status"] == "UNMEASURED" and by_dev[4]["studies_per_s"] is None
    assert by_dev[8]["status"] == "UNMEASURED" and by_dev[8]["studies_per_s"] is None

    # Empty device name must never yield MEASURED.
    def fake_unnamed(design, warmup_seeds, warm_seeds, *, devices=None, fake_shards=False):
        del warmup_seeds, fake_shards
        ref = fleet.reimage_run({**design, "backend": "numpy"}, warm_seeds)
        return ref, {
            "devices_executed": devices,
            "device_names": ["" for _ in range(devices)],
            "wall_s": 0.25,
            "cold_warmup_s": 0.5,
        }

    monkeypatch.setattr(fleet, "measure_warm_sharded", fake_unnamed)
    res2 = fleet.run_fleet(studies=2, cells=60, patients=4, regions_per_patient=1, k=1, device_counts=[1])
    assert res2["fleet"][0]["status"] == "UNMEASURED" and "name" in res2["fleet"][0]["reason"].lower()
