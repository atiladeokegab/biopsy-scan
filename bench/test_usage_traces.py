"""demo/usage/traces.json replays its MEASURED source sample for sample: nothing added, smoothed or dropped."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_replay_matches_its_measured_source():
    replay = json.loads((REPO / "demo" / "usage" / "traces.json").read_text())
    source = json.loads((REPO / replay["source"]).read_text())
    assert source["status"] == "MEASURED" and replay["label"] == "MEASURED" and replay["sample_interval_s"] == 0.1
    assert set(replay["arms"]) == set(source["arms"]) == {"current", "fused", "gpu"}
    for k, r in replay["arms"].items():
        a = source["arms"][k]
        lo, hi = a["compute_window_s"]
        assert r["wall_s"] == round(hi - lo, 3) and r["studies"] == a["studies"] == len(r["done_s"]) == 8
        assert r["summary"] == source["summary"][k]
        raw = {round(t - lo, 2): round(sum(c) / 100, 2) for t, c in a["cpu"] if lo - 0.5 <= t <= hi + 0.5}
        assert [t for t, _ in r["cpu"]] == list(raw) and all(raw[t] == v for t, v in r["cpu"])
        assert len(r["gpu"]) == sum(lo - 0.5 <= t <= hi + 0.5 for t, *_ in a["gpu"])
        assert r["done_s"] == sorted(r["done_s"]) and all(0 < t <= r["wall_s"] + 0.01 for t in r["done_s"])
    assert source["summary"]["current"]["mean_gpu_util_pct"] == 0  # squidpy and lifelines never touch the GPU
    assert source["cost"]["usd"] <= source["cost"]["cap_usd"]
