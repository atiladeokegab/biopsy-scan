"""The Modal GPU runner imports offline: no credentials, no network, no Modal calls."""
import importlib.util
import socket
from pathlib import Path

import numpy as np
import pytest

from reimage.fleet import shard_bounds

APP = Path(__file__).resolve().parents[1] / "gpu" / "modal_app.py"


@pytest.fixture
def modal_app(monkeypatch, tmp_path):
    pytest.importorskip("modal")
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.modal.toml
    for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE"):
        monkeypatch.delenv(key, raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("importing gpu/modal_app.py tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    spec = importlib.util.spec_from_file_location("modal_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_functions_and_gpu_parameters_exist(modal_app):
    assert modal_app.GPUS == ("A100", "H100", "L4") and modal_app.DEFAULT_GPU == "A100"
    for name in ("gpu_tests", "time_kernel", "fleet_shard", "screen_hook", "ab_screen"):
        assert getattr(modal_app, name)._spec_.gpus == "A100", name
    assert modal_app.fetch_data._spec_.gpus is None  # the data download is CPU only
    assert "/data" in modal_app.time_kernel._spec_.volumes
    assert modal_app.ab_screen._spec_.cpu >= 16  # the quiet A/B box
    assert callable(modal_app.run_tests) and callable(modal_app.run_ladder) and callable(modal_app.run_ab)
    assert modal_app._check_gpu("H100") == "H100"
    with pytest.raises(SystemExit):
        modal_app._check_gpu("T4")


@pytest.mark.parametrize("studies,n", [(16, 4), (7, 3), (5, 1)])
def test_fleet_shards_match_reimage_fleet_contract(studies, n):
    split = [a.tolist() for a in np.array_split(list(range(studies)), n)]
    assert split == [list(range(s, e)) for _, s, e in shard_bounds(studies, n)]


def test_mount_uploads_tracked_files_only(modal_app, tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    (repo / "kept.py").write_text("x = 1\n")
    git("add", "kept.py")
    for private in (".env", "key.pem", ".pytest_cache/v/cache/nodeids"):  # untracked: never uploaded
        (repo / private).parent.mkdir(parents=True, exist_ok=True)
        (repo / private).write_text("private\n")
    ignore = modal_app.tracked_only(repo)
    assert not ignore(Path("kept.py"))
    assert all(ignore(Path(p)) for p in (".env", "key.pem", ".pytest_cache/v/cache/nodeids"))
    # and the image really uses it: this repo's own source goes up, an untracked .env would not
    assert callable(modal_app.SOURCE["ignore"])
    assert not modal_app.SOURCE["ignore"](Path("gpu/modal_app.py"))
    assert modal_app.SOURCE["ignore"](Path(".env")) and modal_app.SOURCE["ignore"](Path(".modal.toml"))


def test_screen_result_named_gpu_survives_device_metadata(modal_app, monkeypatch):
    class FakeScreen:
        @staticmethod
        def screen(X, t, ev, n_perm, backend):
            return {"maxT": 3.0, "p": 0.2, "metadata": {"backend": backend}}

    device = {"model": "FAKE-GPU", "driver": "0"}
    monkeypatch.setattr(modal_app, "_pinned", lambda name: FakeScreen)
    monkeypatch.setattr(modal_app, "gpu_info", lambda: device)
    out = modal_app.screen_hook.local()
    assert out["backends"]["gpu"]["backend"] == "gpu" and out["backends"]["gpu"]["maxT"] == 3.0
    assert out["gpu"] == device


def test_parity_false_fails_closed_with_no_timing(modal_app, monkeypatch, tmp_path):
    import json

    (tmp_path / "bench" / "results").mkdir(parents=True)
    monkeypatch.setattr(modal_app, "REPO", tmp_path)
    monkeypatch.setattr(modal_app, "_LAUNCH", {"commit": "abc", "dirty": False, "launched_utc": "t"})
    payload = {"kind": "design_point", "slots": {"c_ours_1_gpu": {"seconds": 27.7},
                                                 "d_ours_n_gpu": {"runs": {"2": {"compute_s": 14.3, "speedup_vs_N1": 1.9,
                                                                                 "bit_identical_to_N1": False}}}},
               "e_gpu_parity": {"exact_all_containers": True}}
    with pytest.raises(SystemExit):
        modal_app._save("A100", payload)
    (written,) = (tmp_path / "bench" / "results").iterdir()
    report = json.loads(written.read_text())
    assert report["status"] == "FAILED"
    assert report["parity_failed"] == ["slots.d_ours_n_gpu.runs.2.bit_identical_to_N1"]
    text = written.read_text()
    assert all(k not in text for k in ("seconds", "compute_s", "speedup", "27.7", "14.3"))


def test_parity_true_is_measured(modal_app, monkeypatch, tmp_path):
    import json

    (tmp_path / "bench" / "results").mkdir(parents=True)
    monkeypatch.setattr(modal_app, "REPO", tmp_path)
    monkeypatch.setattr(modal_app, "_LAUNCH", {"commit": "abc", "dirty": False, "launched_utc": "t"})
    path = modal_app._save("A100", {"kind": "ladder", "kernel": {"parity_exact": True, "speedup_median": 3.3}})
    report = json.loads(path.read_text())
    assert report["status"] == "MEASURED" and report["kernel"]["speedup_median"] == 3.3
