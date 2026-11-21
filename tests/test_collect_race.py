"""The speed race's collector: bench files feed rows by id, and the committed race matches its source file."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collect_race", ROOT / "demo" / "collect_race.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def test_race_list_feeds_rows_by_id():
    d = {"status": "MEASURED", "studies": 1000,
         "race": [{"id": "gpu-1", "seconds": 0.8, "startup_s": 90.0, "label": "MEASURED", "hardware": "Modal", "gpu": "1x H100"}]}
    [(rid, f)] = cr.rows_from(Path("modal-gpu.json"), d)
    assert rid == "gpu-1" and f["seconds"] == 0.8 and f["startup_s"] == 90.0 and f["gpu"] == "1x H100"
    assert f["source"] == "bench/results/modal-gpu.json"


def test_impl_a_feeds_only_the_workflow_footnote():
    d = {"cpu_model": "X", "cohorts": [{"cohort": "charville", "candidates": 3780, "fingerprints": {"X": "ab"}, "timings": [
        {"workers": 1, "wall_s": 300.0, "label": "MEASURED"}, {"workers": 8, "wall_s": 64.0, "label": "MEASURED"}]}]}
    [(rid, f)] = cr.rows_from(Path("impl-a.json"), d)
    assert rid == "workflow" and f["seconds"] == 64.0 and f["cores"] == 8 and f["note"] == "300.0 s on 1 core"
    assert f["fingerprints"] == {"X": "ab"}


def test_other_bench_files_feed_nothing():
    assert list(cr.rows_from(Path("ladder.json"), {"rungs": []})) == []


def test_committed_race_matches_bench_results():
    """Atlas #79 (1): every number shown is the one in the file it cites."""
    b = json.loads((ROOT / "demo" / "data" / "baseline.json").read_text())
    race = {r["id"]: r for r in b["rows"]} | {"workflow": b["workflow"]}
    for path in sorted((ROOT / "bench" / "results").glob("*.json")):
        d = json.loads(path.read_text())
        for rid, fields in (cr.rows_from(path, d) if isinstance(d, dict) else []):
            if rid == "interim":
                assert b["interim"] == fields, "interim: rerun python3 -I demo/collect_race.py"
                continue
            for k, v in fields.items():
                assert race[rid][k] == v, f"{rid}.{k}: rerun python3 -I demo/collect_race.py"
    for r in [*b["rows"], b["workflow"]]:
        if r and r["seconds"] is not None:
            assert (ROOT / r["source"]).is_file(), f"{r['source']}: a timed row cites a committed file"


spec = importlib.util.spec_from_file_location("check_data", ROOT / "demo" / "check_data.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)


def _check(b):
    """The checker's complaint, or None when the file passes."""
    try:
        cd.check_baseline(b, "baseline.json")
    except SystemExit as e:
        return str(e)


def _row(**o):
    return {"id": "a", "method": "m", "role": "screen", "label": "MEASURED", "seconds": 4.0, "hardware": "h",
            "cores": None, "gpu": "1x H100", "source": "s", **o}


def test_checker_validates_each_field_on_its_own():
    """Atlas #79 (4): a gpu string never excuses a bad cores value; seconds must be positive and finite."""
    base = {"schema": "preflight-baseline/1", "task": "t"}
    assert _check(base | {"rows": [_row()]}) is None
    for bad in ({"cores": "<b>8</b>"}, {"cores": True}, {"cores": 0}, {"seconds": float("nan")}, {"seconds": True},
                {"seconds": -1}, {"startup_s": 0}, {"label": "PENDING"}, {"source": ""}, {"label": "GUESS"}):
        assert _check(base | {"rows": [_row(**bad)]}), bad
    assert _check(base | {"rows": [_row()], "workflow": _row(seconds=None)}), "a timed footnote needs seconds"


LADDER = {"kind": "ladder", "status": "MEASURED",
          "kernel": {"unit": "one study: sampling, crop, recount, features", "gpu": {"model": "NVIDIA H100"},
                     "rows": {"numpy_reference": {"median_s": 2.88046, "samples_s": [1] * 5},
                              "gpu": {"median_s": 0.87341, "samples_s": [1] * 5}}},
          "fleet": {"studies": 16, "design": "bench/real_cpu.py DESIGN", "note": "compute only", "runs": {
              "4": {"gpus": ["A100-40", "A100-80"], "containers_seen": 4, "studies_per_container": [4] * 4,
                    "compute_s_max": 3.7977, "startup_and_transfer_s": 5.0945},
              "1": {"gpus": ["A100-40"], "containers_seen": 1, "studies_per_container": [16],
                    "compute_s_max": 11.4448, "startup_and_transfer_s": 90.293}}}}


def test_ladder_feeds_interim_compute_only_with_startup_apart():
    """The lead's interim race: kernel 0.873 vs 2.880 s/study, fleet 3.80 vs 11.44 s compute, from the file."""
    [(rid, groups)] = cr.rows_from(Path("modal-a100-x.json"), LADDER)
    assert rid == "interim"
    k, f = ({r["id"]: r for r in g["rows"]} for g in groups)
    assert (k["kernel-numpy"]["seconds"], k["kernel-gpu"]["seconds"], k["kernel-gpu"]["gpu"]) == (2.88, 0.873, "NVIDIA H100")
    assert [r["id"] for r in groups[1]["rows"]] == ["fleet-1", "fleet-4"]
    assert (f["fleet-1"]["seconds"], f["fleet-1"]["startup_s"]) == (11.44, 90.3), "start-up is apart, never in seconds"
    assert (f["fleet-4"]["seconds"], f["fleet-4"]["gpu"]) == (3.8, "4x A100-40 / A100-80")
    assert all(r["source"] == "bench/results/modal-a100-x.json" for g in groups for r in g["rows"])
    assert not any("speedup" in key or "ratio" in key for g in groups for r in g["rows"] for key in r)
    b = json.loads((ROOT / "demo" / "data" / "baseline.json").read_text())
    assert _check(b | {"interim": groups}) is None
    assert _check(b | {"interim": [groups[0] | {"rows": [k["kernel-gpu"] | {"label": "PENDING"}]}]}), "PENDING with a number"
    assert list(cr.rows_from(Path("x.json"), LADDER | {"status": "FAILED"})) == []


DP = {"kind": "design_point", "status": "MEASURED", "studies": 1000, "slots": {
    "a_lab_all_cores": {"label": "MEASURED", "seconds": 1043.2, "workers": 48},
    "b_ours_cpu_all_cores": {"label": "MEASURED", "seconds": 673.1, "workers": 48},
    "c_ours_1_gpu": {"label": "MEASURED", "seconds": 865.4, "gpu": ["NVIDIA H100"]},
    "d_ours_n_gpu": {"label": "MEASURED", "runs": {"16": {"compute_s": 54.1, "startup_and_transfer_s": 6.2,
                                                          "containers_seen": 16, "gpus": ["NVIDIA H100"]}}}}}


def test_design_point_fills_the_four_rows_only_when_measured_at_1000_studies():
    """Atlas #79 re-check (b): a FAILED run, or any workload but the declared design point, fills nothing."""
    rows = dict(cr.rows_from(Path("dp.json"), DP))
    assert sorted(rows) == ["gpu-1", "gpu-16", "lab-cpu", "preflight-cpu"]
    assert rows["gpu-1"]["gpu"] == "1x NVIDIA H100" and rows["lab-cpu"]["cores"] == 48
    assert (rows["gpu-16"]["seconds"], rows["gpu-16"]["startup_s"], rows["gpu-16"]["gpu"]) == (54.1, 6.2, "16x NVIDIA H100")
    for bad in ({"status": "FAILED"}, {"status": None}, {"studies": 32}, {"studies": "1000"}, {"studies": None}):
        assert list(cr.rows_from(Path("dp.json"), DP | bad)) == [], bad
    race = {"race": [{"id": "gpu-1", "seconds": 1.0, "label": "MEASURED", "hardware": "h", "gpu": "g"}]}
    assert list(cr.rows_from(Path("r.json"), race)) == [], "a race list must declare MEASURED and 1000 studies"
    assert list(cr.rows_from(Path("r.json"), race | {"status": "MEASURED", "studies": 32})) == []


def test_interim_titles_say_exactly_what_ran():
    """Atlas #79 re-check (a): the kernel step is feature generation only, never 're-imaged and scored'."""
    [(_, groups)] = cr.rows_from(Path("m.json"), LADDER)
    assert groups[0]["title"] == "One study, feature generation only (sampling, crop, recount, features)"
    b = json.loads((ROOT / "demo" / "data" / "baseline.json").read_text())
    assert not any("scor" in g["title"] or "re-imag" in g["title"] for g in b.get("interim", []))


def test_claims_are_traced_to_their_files():
    """Atlas #79 re-check (c): every claims number equals what its file gives, and the checker refuses gaps."""
    committed = json.loads((ROOT / "demo" / "data" / "claims.json").read_text())
    assert committed == cr.claims(write=False), "rerun python3 -I demo/collect_race.py"
    def bad(c):
        try:
            cd.check_claims(c, "claims.json")
        except SystemExit as e:
            return str(e)
    assert bad(committed) is None
    for k, f, v in (("power", "source", "bench/results/nope.json"), ("power", "sim", float("nan")), ("maxT", "p", None),
                    ("false_alarms", "label", None), ("power", "hi", 0.1), ("maxT", "tests", "3780")):
        assert bad(committed | {k: committed[k] | {f: v}}), (k, f, v)
