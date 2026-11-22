"""The L4 frame-cadence writer refuses MEASURED for a broken scene (offline: no Modal call, no browser)."""
import importlib.util
from pathlib import Path

import pytest

JOB = Path(__file__).resolve().parents[1] / "gpu" / "fps_l4.py"


@pytest.fixture
def fps(monkeypatch, tmp_path):
    pytest.importorskip("modal")
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.modal.toml
    for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE"):
        monkeypatch.delenv(key, raising=False)
    spec = importlib.util.spec_from_file_location("fps_l4", JOB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def take(seconds=15.5, fps=60):
    """rAF timestamps (ms) at a steady cadence over the given span."""
    return [i * 1000 / fps for i in range(int(seconds * fps) + 1)]


def scene(**over):
    out = {"renderer": "ANGLE (NVIDIA, Vulkan (NVIDIA NVIDIA L4))", "last_step": 3, "errors": [], "end_controls": ["reimage-replay"]}
    return {**out, **over}


def test_a_whole_working_take_is_measured(fps):
    res = fps.result(scene(), take(), wall=60.0, stamp="20261122T000000Z")
    assert res["status"] == "MEASURED" and res["median_fps"] == 60.0
    assert "observed frame cadence on an L4, headless, vsync-capped" in res["label"]
    assert "throughput" not in res["label"]


@pytest.mark.parametrize("out,ts,why", [
    (scene(last_step=-1), take(), "last step"),                       # the take never started
    (scene(errors=["TypeError: p is not defined"]), take(), "error"),  # a page error
    (scene(errors=["console: 3D view off: no WebGL"]), take(), "error"),  # WebGL failed, fallback shown
    (scene(), take(seconds=4.0), "duration"),                         # sample cut short
    (scene(), take(seconds=15.5, fps=2), "frames"),                   # the scene stalled
    (scene(end_controls=["reimage-replay", "reimage-pause"]), take(seconds=15.5, fps=60), "take did not end"),  # Pause still showing
    (scene(end_controls=[]), take(seconds=15.5, fps=60), "take did not end"),                                    # no controls at all
])
def test_a_broken_scene_is_failed_not_measured(fps, out, ts, why):
    res = fps.result(out, ts, wall=60.0, stamp="20261122T000000Z")
    assert res["status"] == "FAILED"
    assert why in res["reason"]
    assert "median_fps" not in res  # no cadence figure from a broken scene
