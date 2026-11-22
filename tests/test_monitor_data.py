"""monitor/data.json is a fresh build of the measured traces, and the page hardcodes none of their numbers."""
import importlib.util
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("monitor_build", ROOT / "monitor" / "build.py")
mb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mb)
DATA = json.loads((ROOT / "monitor" / "data.json").read_text())


def test_data_json_is_a_fresh_build():
    assert (ROOT / "monitor" / "data.json").read_text() == json.dumps(mb.build(), separators=(",", ":")) + "\n"


def test_lanes_replay_measured_samples_on_the_race_clock():
    race = json.loads((ROOT / "demo" / "app" / "data.json").read_text())["race"]
    assert DATA["race"] == {"seconds": race["seconds"], "race_s_per_real_s": race["race_s_per_real_s"]}
    for lane in DATA["lanes"].values():
        assert lane["cpu"] and lane["gpu_util"]
        assert all(a[0] <= b[0] for a, b in zip(lane["cpu"], lane["cpu"][1:]))
    busy = lambda lane: [c for t, c in lane["cpu"] if 0 <= t <= lane["wall_s"]]  # noqa: E731
    cur = busy(DATA["lanes"]["current"])
    assert 7 < sum(cur) / len(cur) < 8  # about 7.5 cores, as measured
    if DATA["ours_source"] == mb.TRACES:  # CPU-fused fallback until the H100 usage trace lands
        ours = busy(DATA["lanes"]["ours"])
        assert 6.5 < sum(ours) / len(ours) < 8


def test_newest_h100_usage_file_wins(tmp_path, monkeypatch):
    res = tmp_path / "bench" / "results"
    res.mkdir(parents=True)
    real = mb.newest_h100(ROOT / "bench" / "results")
    assert real, "the H100 usage trace is committed"
    for stamp in ("20261122T140000Z", "20261122T160000Z"):
        shutil.copy(ROOT / real, res / f"gpu-fused-usage-h100-{stamp}.json")
    for rel in (mb.TRACES, mb.RACE):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    monkeypatch.setattr(mb, "REPO", tmp_path)
    assert mb.build(res)["ours_source"] == "bench/results/gpu-fused-usage-h100-20261122T160000Z.json"


def test_ours_is_the_timed_h100_launch():
    ours = DATA["lanes"]["ours"]
    assert DATA["ours_source"].startswith("bench/results/gpu-fused-usage-h100-")
    assert ours["gpu"].startswith("H100") and ours["studies"] == 3000 and 6 < ours["wall_s"] < 6.3
    util = sorted(u for t, u in ours["gpu_util"] if 0 <= t <= ours["wall_s"])
    assert util[len(util) // 2] == 100  # the kernel holds the H100 busy


def test_page_shows_no_ratio_and_hardcodes_no_trace_numbers():
    ui = {p.name: p.read_text() for p in (ROOT / "monitor").glob("*.[hj][ts]*") if p.suffix in (".html", ".js")}
    nums = {f"{lane['wall_s']}" for lane in DATA["lanes"].values()} | {str(lane["studies"]) for lane in DATA["lanes"].values() if lane["studies"] >= 10}
    for name, text in ui.items():
        assert not re.search(r"(?i)\bratio\b|faster|\d\s*[x×]\b", text.replace("aspect-ratio", "")), name
        assert not [n for n in nums if re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])", text)], name
        assert not re.search(r"(?i)\blive\b", text), name  # it replays measured samples
