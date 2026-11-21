"""The speed race's collector: bench files feed rows by id, and the committed race matches its source file."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collect_race", ROOT / "demo" / "collect_race.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def test_a_generic_race_list_feeds_nothing():
    """Atlas #79 round 3 (2): no producer writes one, and it bypassed the GPU parity check; only design_point fills rows."""
    d = {"status": "MEASURED", "studies": 1000,
         "race": [{"id": "gpu-1", "seconds": 0.8, "startup_s": 90.0, "label": "MEASURED", "hardware": "Modal", "gpu": "1x H100"}]}
    assert list(cr.rows_from(Path("modal-gpu.json"), d)) == []


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
    base = {"schema": "preflight-baseline/1", "task": "t", "scope": {"comparison": "WORKFLOW c", "calibration": "EXPLORATORY c"}}
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


# The shape #75's gpu/modal_app.py run_design_point writes (origin/modal-gpu 25fcaca, lines 704-760), values invented.
DP = {"schema_version": 1, "runner": "modal", "status": "MEASURED", "kind": "design_point", "studies": 1000,
      "unit": "one design point: 1000 Charville studies, bench/real_cpu.py DESIGN. ...",
      "slots": {
          "a_lab_all_cores": {"label": "MEASURED", "seconds": 1043.2, "measured_studies": 1000, "measured_seconds": 1043.2,
                              "basis": "measured", "workers": 48, "host": "h", "positives": 61},
          "b_ours_cpu_all_cores": {"label": "MEASURED", "seconds": 673.1, "workers": 48, "host": "h",
                                   "plus_threshold_and_calls_s": 0.41},
          "c_ours_1_gpu": {"label": "MEASURED", "seconds": 865.4, "gpu": ["NVIDIA H100"], "plus_threshold_and_calls_s": 0.38},
          "d_ours_n_gpu": {"label": "MEASURED", "runs": {"1": {"compute_span_s": 865.4}, "16": {
              "containers_requested": 16, "containers_seen": 16, "gpus": ["NVIDIA H100"], "compute_s_max_shard": 57.0,
              "modal_wall_s": 70.0, "compute_span_s": 54.1, "common_overlap_s": 50.2, "startup_and_transfer_s": 6.2,
              "preparation_s_slowest": 3.0, "parity_exact_all": True, "bit_identical_to_N1": True,
              "speedup_vs_N1": 16.0, "efficiency": 1.0}}}},
      "scope": {"comparison": "WORKFLOW comparison: different statistic (conditional permutation maxT vs R-15 Cox score) "
                              "and different calibration; not equal work",
                "calibration": "EXPLORATORY: one paired shuffle per study (same X, outcomes shuffled within strata), "
                               "not an independent null study or unconditional calibration",
                "agreement": "same-call counts compare two methods; they do not establish validity"},
      "e_gpu_parity": {"exact_all_containers": True, "suite": "s"},
      "timing": "seconds = in-container compute (GPU slots: the measured compute span across containers); ..."}


def test_design_point_fills_the_four_rows_only_when_measured_at_1000_studies():
    """Atlas #79 re-check (b): a FAILED run, or any workload but the declared design point, fills nothing."""
    rows = dict(cr.rows_from(Path("dp.json"), DP))
    assert sorted(rows) == ["gpu-1", "gpu-16", "lab-cpu", "preflight-cpu", "scope"]
    assert rows["gpu-1"]["gpu"] == "1x NVIDIA H100" and rows["lab-cpu"]["cores"] == 48
    assert (rows["gpu-16"]["seconds"], rows["gpu-16"]["startup_s"], rows["gpu-16"]["gpu"]) == (54.1, 6.2, "16x NVIDIA H100")
    # Atlas #79 round 3 (1)(2): the measured compute span, the calls timed apart, the scope kept
    assert rows["preflight-cpu"]["calls_s"] == 0.41 and rows["gpu-1"]["calls_s"] == 0.38 and "calls_s" not in rows["lab-cpu"]
    assert rows["scope"] == {k: DP["scope"][k] for k in ("comparison", "calibration")}
    for bad in ({"status": "FAILED"}, {"status": None}, {"studies": 32}, {"studies": "1000"}, {"studies": None}):
        assert list(cr.rows_from(Path("dp.json"), DP | bad)) == [], bad
    race = {"race": [{"id": "gpu-1", "seconds": 1.0, "label": "MEASURED", "hardware": "h", "gpu": "g"}]}
    assert list(cr.rows_from(Path("r.json"), race)) == [], "a race list must declare MEASURED and 1000 studies"
    assert list(cr.rows_from(Path("r.json"), race | {"status": "MEASURED", "studies": 32})) == []


def test_interim_titles_say_exactly_what_ran():
    """Atlas #79 re-check (a): the kernel step is feature generation only, never 're-imaged and scored'."""
    [(_, groups)] = cr.rows_from(Path("m.json"), LADDER)
    assert groups[0]["title"] == "One study: re-imaging and feature extraction only (scoring not timed)"
    assert groups[1]["same_scope"] is False and "longest-shard compute" in groups[1]["title"]
    b = json.loads((ROOT / "demo" / "data" / "baseline.json").read_text())
    assert not any("scored" in g["title"] for g in b.get("interim", []))


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


def test_gpu_rows_need_exact_parity_and_failed_one_study_lists_stay_pending():
    """Atlas #79 re-check (4): Root's repro, a FAILED one-study report, never becomes a MEASURED 1,000 lane;
    GPU rows also need the file's exact-parity check."""
    failed = {"status": "FAILED", "studies": 1, "race": [{"id": "lab-cpu", "seconds": 2.0, "label": "MEASURED",
                                                          "hardware": "h", "cores": 8}]}
    assert list(cr.rows_from(Path("f.json"), failed)) == []
    rows = dict(cr.rows_from(Path("dp.json"), DP | {"e_gpu_parity": {"exact_all_containers": False}}))
    assert sorted(rows) == ["lab-cpu", "preflight-cpu", "scope"], "no GPU lane without exact parity"
    assert sorted(dict(cr.rows_from(Path("dp.json"), {k: v for k, v in DP.items() if k != "e_gpu_parity"}))) == ["lab-cpu", "preflight-cpu", "scope"]
    no_scope = {k: v for k, v in DP.items() if k != "scope"}
    assert list(cr.rows_from(Path("dp.json"), no_scope)) == [], "without its WORKFLOW/EXPLORATORY scope, every lane stays PENDING"


def test_timed_design_point_rows_need_their_scope_on_the_page():
    """Atlas #79 round 3 (2): the checker refuses timed race rows without the WORKFLOW/EXPLORATORY scope."""
    b = json.loads((ROOT / "demo" / "data" / "baseline.json").read_text())
    rows = dict(cr.rows_from(Path("dp.json"), DP))
    scope = rows.pop("scope")
    timed = b | {"rows": [r | rows.get(r["id"], {}) for r in b["rows"]]}
    assert _check(timed) and "scope" in _check(timed)
    assert _check(timed | {"scope": scope}) is None
    assert _check(timed | {"scope": scope, "rows": [r | {"calls_s": -1} for r in timed["rows"]]})


def test_power_and_false_alarms_say_which_model_they_are():
    """Atlas #79 re-check (1): 27% is the v1 planted-HR model's power, not PD1's; 28/700 is pooled over designs."""
    c = cr.claims(write=False)
    assert "HR 2" in c["power"]["model"] and "not the PD1 pattern's power" in c["power"]["model"]
    # Atlas #79 round 3 (3): the simulation calls by calibrated maxT; Bonferroni is the formula's
    m = c["power"]["model"]
    assert "per source SD" in m and "simulation: calibrated maxT" in m and "formula: Bonferroni" in m
    assert m.index("Bonferroni") > m.index("formula")
    assert c["false_alarms"]["designs"] == 7


BAD_CLAIMS = [("power", "patients", 0), ("power", "patients", 1.5), ("power", "n", 0), ("power", "n", 2.5),
              ("power", "sim", 1.2), ("power", "lo", 0.9), ("power", "formula", -0.1), ("power", "formula", 1.1),
              ("false_alarms", "hits", -1), ("false_alarms", "hits", 1.5), ("false_alarms", "hits", 701),
              ("false_alarms", "n", 0), ("false_alarms", "designs", 0), ("false_alarms", "target", 1.5),
              ("false_alarms", "target", -0.1), ("false_alarms", "target", None), ("maxT", "p", 0), ("maxT", "p", 1.1),
              ("maxT", "tests", 0), ("maxT", "tests", 2.5), ("maxT", "z", None), ("maxT", "p", True)]


def test_python_and_browser_claims_validators_agree():
    """Atlas #79 round 3 (4): check_data and claims.js accept and refuse exactly the same claims files."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        import pytest
        pytest.skip("node not installed")
    good = json.loads((ROOT / "demo" / "data" / "claims.json").read_text())
    cases = [good] + [good | {k: good[k] | {f: v}} for k, f, v in BAD_CLAIMS]
    js = ("import { claimTexts } from './demo/claims.js'; import { readFileSync } from 'node:fs';"
          "const cs = JSON.parse(readFileSync(0, 'utf8'));"
          "console.log(JSON.stringify(cs.map((c) => { try { claimTexts(c); return true } catch { return false } })));")
    out = subprocess.run([node, "--input-type=module", "-e", js], input=json.dumps(cases), capture_output=True, text=True,
                         cwd=ROOT, check=True).stdout
    def py_ok(c):
        try:
            cd.check_claims(c, "claims.json")
            return True
        except SystemExit:
            return False
    assert json.loads(out) == [py_ok(c) for c in cases]
    assert json.loads(out) == [True] + [False] * len(BAD_CLAIMS)
