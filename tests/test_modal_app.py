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
    assert modal_app.GPUS == ("A100-40GB", "A100-80GB", "H100", "L4") and modal_app.DEFAULT_GPU == "A100-40GB"
    for name in ("gpu_tests", "time_kernel", "fleet_shard", "screen_hook", "ab_screen"):
        assert getattr(modal_app, name)._spec_.gpus == "A100-40GB", name  # exact type: no 40/80 GB mix
    assert modal_app.fetch_data._spec_.gpus is None  # the data download is CPU only
    assert "/data" in modal_app.time_kernel._spec_.volumes
    assert modal_app.ab_screen._spec_.cpu >= 16  # the quiet A/B box
    assert callable(modal_app.run_tests) and callable(modal_app.run_ladder) and callable(modal_app.run_ab)
    assert callable(modal_app.run_gpu_vs)
    assert modal_app._check_gpu("H100") == "H100"
    for bad in ("T4", "A100"):  # plain A100 may be either memory size
        with pytest.raises(SystemExit):
            modal_app._check_gpu(bad)


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
        modal_app._save("A100-40GB", payload)
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
    path = modal_app._save("A100-40GB", {"kind": "ladder", "kernel": {"parity_exact": True, "speedup_median": 3.3}})
    report = json.loads(path.read_text())
    assert report["status"] == "MEASURED" and report["kernel"]["speedup_median"] == 3.3


def test_simulator_subprocess_tests_are_not_counted_as_device_coverage(modal_app):
    sim = modal_app.simulator_tests(APP.parents[1])
    assert {"test_gpu_v2_matches_reference_and_fleet_is_bit_identical_under_cuda_simulator",
            "test_gpu_deterministic_float64_order_and_rejects_cpu_snapshot",
            "test_two_fake_shards_on_cuda_simulator_are_bit_identical"} <= sim
    assert "test_invalid_device_requests_fail_honestly" not in sim  # names CUDASIM only in a message


def test_gpu_tests_refuses_the_cuda_simulator(modal_app, monkeypatch):
    pytest.importorskip("numba")
    monkeypatch.setenv("NUMBA_ENABLE_CUDASIM", "1")
    with pytest.raises(RuntimeError, match="real CUDA device"):
        modal_app.gpu_tests.local()


def _part(task, seeds, start, end):
    return {"compute_window": [start, end], "seeds": seeds, "gpu": {"modal_task_id": task}}


def test_scaling_uses_measured_overlap_not_max_shard(modal_app):
    serial = modal_app.overlap([_part("a", [0, 1], 0.0, 10.0), _part("b", [2, 3], 10.5, 20.0)], [0, 1, 2, 3])
    assert serial["compute_span_s"] == 20.0 and not serial["all_overlapped"]  # max(shard) would claim 10 s
    together = modal_app.overlap([_part("a", [0, 1], 0.0, 10.0), _part("b", [2, 3], 1.0, 10.0)], [0, 1, 2, 3])
    assert together["all_overlapped"] and together["seed_coverage_exact"] and together["distinct_task_ids"]
    gaps = modal_app.overlap([_part("a", [0, 1], 0.0, 1.0), _part("a", [1, 3], 0.0, 1.0)], [0, 1, 2, 3])
    assert not gaps["seed_coverage_exact"] and not gaps["distinct_task_ids"]
    assert modal_app.parity_failures({"runs": {"2": gaps}}) == ["runs.2.seed_coverage_exact", "runs.2.distinct_task_ids"]


def test_screen_comparison_needs_every_output_not_just_z(modal_app):
    b = {"U": [1.0], "V": [2.0], "z": [0.5, np.nan], "n": [10], "maxT": 3.0, "p": 0.2, "adjusted_p": [0.4],
         "null_maxT": [1.0, 2.0]}
    assert modal_app.compare_screens(b, dict(b))["all"]
    c = {**b, "null_maxT": [1.0, 2.5], "p": 0.3}  # same z, different resamples
    out = modal_app.compare_screens(b, c)
    assert out["z"] and not out["null_maxT"] and not out["p"] and not out["all"]


def _results(modal_app, monkeypatch, tmp_path):
    out = tmp_path / "bench" / "results"
    out.mkdir(parents=True)
    monkeypatch.setattr(modal_app, "REPO", tmp_path)
    monkeypatch.setattr(modal_app, "_LAUNCH", {"commit": "abc", "dirty": False, "launched_utc": "t"})
    return out


@pytest.mark.parametrize("failed,unexpected", [(1, []), (0, ["test_gpu_runs_on_device"])])
def test_failed_or_unexpectedly_skipped_gpu_tests_never_write_measured(modal_app, monkeypatch, tmp_path,
                                                                       failed, unexpected):
    import json

    out = _results(modal_app, monkeypatch, tmp_path)
    result = {"exit_code": 1 if failed else 0, "passed": 38, "failed": failed, "errors": 0, "skipped": 1,
              "unexpected_skips": unexpected, "in_function_s": 40.0,
              "gpu": {"model": "NVIDIA A100-SXM4-40GB", "driver": "610"}}
    monkeypatch.setattr(modal_app, "_tests", lambda gpu: result)
    with pytest.raises(SystemExit):
        modal_app.run_tests("A100-40GB")
    (written,) = out.iterdir()
    report = json.loads(written.read_text())
    assert report["status"] == "FAILED" and report["parity_failed"]
    assert "MEASURED" not in written.read_text()


class _FakeClock:
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    perf_counter = time


def test_gpu_vs_startup_is_per_gpu_and_hardware_is_kept_per_block(modal_app, monkeypatch, tmp_path):
    """Two GPUs run at once; waiting on the slower one must not count as the faster one's start-up."""
    import json

    _results(modal_app, monkeypatch, tmp_path)
    clock = _FakeClock()
    monkeypatch.setattr(modal_app, "time", clock)
    monkeypatch.setattr(modal_app, "_spend", lambda *a: None)
    monkeypatch.setattr(modal_app, "fetch_data", type("F", (), {"remote": staticmethod(lambda: [])}))
    in_function = {"A100-40GB": 10.0, "H100": 2.0}  # both enter 5 s after launch
    block = {"n": 0}

    class Call:
        def __init__(self, gpu, seeds, launched):
            self.gpu, self.seeds, self.launched = gpu, seeds, launched

        def get(self):
            done = self.launched + 5 + in_function[self.gpu]
            clock.now = max(clock.now, done)  # waiting on this call
            return {"seeds": self.seeds, "features": np.array([[float(s)] for s in self.seeds]),
                    "compute_s": in_function[self.gpu] - 1, "in_function_s": in_function[self.gpu],
                    "entered_epoch": self.launched + 5,
                    "gpu": {"model": f"{self.gpu} card", "modal_task_id": f"t{block['n']}"}}

    class Fn:
        def __init__(self, gpu):
            self.gpu = gpu

        def spawn(self, seeds):
            block["n"] += 1
            return Call(self.gpu, seeds, clock.now)

    monkeypatch.setattr(modal_app, "fleet_shard", type("S", (), {"with_options": staticmethod(lambda gpu: Fn(gpu))}))
    modal_app.run_gpu_vs(studies=8, blocks=2, gpus="A100-40GB,H100")
    (written,) = (tmp_path / "bench" / "results").iterdir()
    report = json.loads(written.read_text())
    assert report["status"] == "MEASURED"
    for g in in_function:
        row = report["rows"][g]
        assert row["startup_s"] == [5.0, 5.0], g  # not 13 s for whichever GPU is collected second
        assert len({b["modal_task_id"] for b in row["gpus_per_block"]}) == 2, g  # one per block, none overwritten
        assert row["same_gpu_model"] is True, g


@pytest.mark.parametrize("containers,studies,fleet", [(4, 16, "0"), (4, 16, "1,32"), (4, 16, "-2"), (4, 16, "a"),
                                                      (32, 16, ""), (0, 16, "")])
def test_bad_fleet_sizes_stop_before_any_launch(modal_app, monkeypatch, containers, studies, fleet):
    def launched():
        raise AssertionError("a Modal call ran before the fleet sizes were validated")

    monkeypatch.setattr(modal_app, "fetch_data", type("F", (), {"remote": staticmethod(launched)}))
    with pytest.raises(SystemExit):
        modal_app.run_ladder("A100-40GB", containers=containers, studies=studies, fleet=fleet)


def test_good_fleet_sizes_are_sorted_and_include_one(modal_app):
    assert modal_app._fleet_sizes(4, 16, "8, 2,16") == [1, 2, 4, 8, 16]


@pytest.mark.parametrize("studies,blocks", [(1000, 0), (2, 4), (1000, -1)])
def test_bad_gpu_vs_blocks_stop_before_any_launch(modal_app, monkeypatch, studies, blocks):
    def launched(*a):
        raise AssertionError("a Modal call or spend entry ran before --blocks was validated")

    monkeypatch.setattr(modal_app, "fetch_data", type("F", (), {"remote": staticmethod(launched)}))
    monkeypatch.setattr(modal_app, "_spend", launched)
    with pytest.raises(SystemExit):
        modal_app.run_gpu_vs(studies=studies, blocks=blocks, gpus="A100-40GB,H100")
