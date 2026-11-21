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
    for name in ("gpu_tests", "time_kernel", "fleet_shard", "screen_hook"):
        assert getattr(modal_app, name)._spec_.gpus == "A100", name
    assert modal_app.fetch_data._spec_.gpus is None  # the data download is CPU only
    assert "/data" in modal_app.time_kernel._spec_.volumes
    assert callable(modal_app.run_tests) and callable(modal_app.run_ladder)
    assert modal_app._check_gpu("H100") == "H100"
    with pytest.raises(SystemExit):
        modal_app._check_gpu("T4")


@pytest.mark.parametrize("studies,n", [(16, 4), (7, 3), (5, 1)])
def test_fleet_shards_match_reimage_fleet_contract(studies, n):
    split = [a.tolist() for a in np.array_split(list(range(studies)), n)]
    assert split == [list(range(s, e)) for _, s, e in shard_bounds(studies, n)]
