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
    race = {r["id"]: r for r in b["rows"]} | {"workflow": b["workflow"], "scope": b.get("scope")}
    fed = {}  # as main(): a later file's fields for the same id win
    for path in sorted((ROOT / "bench" / "results").glob("*.json")):
        d = json.loads(path.read_text())
        for rid, fields in (cr.rows_from(path, d) if isinstance(d, dict) else []):
            fed[rid] = fields if rid in ("interim", "scope") else fed.get(rid, {}) | fields
    for rid, fields in fed.items():
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


def _run(n, span, gpus=("NVIDIA A100-SXM4-40GB",), **o):
    return {"containers_requested": n, "containers_seen": n, "gpus": list(gpus), "studies_per_container": [1000 // n] * n,
            "compute_s_max": span - 1, "compute_span_s": span, "startup_and_transfer_s": 30.0 / n, "all_overlapped": True,
            "seed_coverage_exact": True, "bit_identical_to_1": True, **o}


# The shape of Atlas's cleared run (bench/results/modal-a100-20261121T183709Z.json on #75), values from that file.
CLEARED = "modal-a100-20261121T183709Z.json"
LADDER = {"kind": "ladder", "status": "MEASURED",
          "kernel": {"unit": "one study: sampling, crop, recount, features", "gpu": {"model": "NVIDIA A100-SXM4-40GB"},
                     "parity_exact": True, "speedup_median": 3.1863154273577705,
                     "rows": {"numpy_reference": {"median_s": 2.235254685000001, "samples_s": [1] * 5},
                              "gpu": {"median_s": 0.7015170769999912, "samples_s": [1] * 5}}},
          "fleet": {"studies": 1000, "design": "bench/real_cpu.py DESIGN on Charville", "note": "compute only", "runs": {
              "1": _run(1, 843.7230155467987, startup_and_transfer_s=34.70365832602988),
              "4": _run(4, 192.30906319618225),
              "8": _run(8, 128.19890642166138, startup_and_transfer_s=6.983653898236668),
              "16": _run(16, 202.8938708305359, gpus=("NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB"))}}}


def test_ladder_feeds_interim_compute_only_with_startup_apart():
    """The lead (19:00Z): 1 A100 against 8 on the 1,000 studies, the compute span, start-up apart; no 16-GPU bar."""
    [(rid, groups)] = cr.rows_from(Path(CLEARED), LADDER)
    assert rid == "interim"
    k, f = ({r["id"]: r for r in g["rows"]} for g in groups)
    assert (k["kernel-numpy"]["seconds"], k["kernel-gpu"]["seconds"]) == (2.2353, 0.7015)
    assert [r["id"] for r in groups[1]["rows"]] == ["fleet-1", "fleet-8"], "no 16-GPU bar, no mixed 40/80 GB"
    assert (f["fleet-1"]["seconds"], f["fleet-1"]["startup_s"]) == (843.72, 34.7), "start-up is apart, never in seconds"
    assert (f["fleet-8"]["seconds"], f["fleet-8"]["gpu"]) == (128.2, "8x NVIDIA A100-SXM4-40GB")
    assert f["fleet-8"]["method"] == "8 A100s, 125 studies each" and f["fleet-8"]["note"] == "compute span"
    assert all(r["source"] == f"bench/results/{CLEARED}" for g in groups for r in g["rows"])
    assert not any("speedup" in key or "ratio" in key for g in groups for r in g["rows"] for key in r)
    b = json.loads((ROOT / "demo" / "data" / "baseline.json").read_text())
    assert _check(b | {"interim": groups}) is None
    assert _check(b | {"interim": [groups[0] | {"rows": [k["kernel-gpu"] | {"label": "PENDING"}]}]}), "PENDING with a number"
    assert _check(b | {"interim": [groups[1] | {"scope_line": ""}]}), "a group's scope line, if any, is text"
    assert list(cr.rows_from(Path(CLEARED), LADDER | {"status": "FAILED"})) == []


def test_only_the_cleared_fleet_run_feeds_the_page():
    """Atlas cleared one file; any other ladder file, or one whose 1-vs-8 pair is not clean, feeds nothing."""
    assert list(cr.rows_from(Path("modal-a100-20261121T171213Z.json"), LADDER)) == []
    runs = LADDER["fleet"]["runs"]
    for bad in ({"bit_identical_to_1": False}, {"all_overlapped": False}, {"seed_coverage_exact": False},
                {"gpus": ["NVIDIA A100-SXM4-80GB"]}, {"containers_seen": 7}):
        broken = LADDER | {"fleet": LADDER["fleet"] | {"runs": runs | {"8": runs["8"] | bad}}}
        assert list(cr.rows_from(Path(CLEARED), broken)) == [], bad
    no8 = LADDER | {"fleet": LADDER["fleet"] | {"runs": {"1": runs["1"]}}}
    assert list(cr.rows_from(Path(CLEARED), no8)) == []
    assert list(cr.rows_from(Path(CLEARED), LADDER | {"kernel": LADDER["kernel"] | {"parity_exact": False}})) == []


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
    assert sorted(rows) == ["gpu-1", "lab-cpu", "preflight-cpu", "scope"], "no 16-GPU lane (lead, 2026-11-21)"
    assert rows["gpu-1"]["gpu"] == "1x NVIDIA H100" and rows["lab-cpu"]["cores"] == 48
    # Atlas #79 round 3 (1)(2): the measured compute span, the calls timed apart, the scope kept
    assert rows["preflight-cpu"]["calls_s"] == 0.41 and rows["gpu-1"]["calls_s"] == 0.38 and "calls_s" not in rows["lab-cpu"]
    assert rows["scope"] == {k: DP["scope"][k] for k in ("comparison", "calibration")}
    for bad in ({"status": "FAILED"}, {"status": None}, {"studies": 32}, {"studies": "1000"}, {"studies": None}):
        assert list(cr.rows_from(Path("dp.json"), DP | bad)) == [], bad
    race = {"race": [{"id": "gpu-1", "seconds": 1.0, "label": "MEASURED", "hardware": "h", "gpu": "g"}]}
    assert list(cr.rows_from(Path("r.json"), race)) == [], "a race list must declare MEASURED and 1000 studies"
    assert list(cr.rows_from(Path("r.json"), race | {"status": "MEASURED", "studies": 32})) == []


def test_interim_titles_say_exactly_what_ran():
    """The lead (19:00Z): Atlas's cleared scope lines, word for word, each beside its bars; numbers from the file."""
    [(_, groups)] = cr.rows_from(Path(CLEARED), LADDER)
    assert groups[0]["scope_line"] == ("MEASURED, 3.19× faster: feature step only, NumPy vs GPU, "
                                  "same container, median of 5, exact parity")
    assert groups[1]["scope_line"] == ("MEASURED, observed 6.6× compute-span reduction 843.7→128.2 s for 1,000 Charville "
                                  "re-imaging/feature-generation studies, 8 A100-SXM4-40GB vs 1, one sweep; "
                                  "outputs bit-identical; inference outside timing")
    b = json.loads((ROOT / "demo" / "data" / "baseline.json").read_text())
    shown = " ".join(g["title"] + " " + g.get("scope_line", "") for g in b.get("interim", []))
    assert "scored" not in shown and "% of ideal" not in shown and "efficiency" not in shown
    assert [g.get("scope_line") for g in b["interim"]] == [g["scope_line"] for g in groups], "rerun python3 -I demo/collect_race.py"


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
    unscoped = {k: v for k, v in b.items() if k != "scope"}  # the committed race carries its own scope
    timed = unscoped | {"rows": [r | rows.get(r["id"], {}) for r in b["rows"]]}
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


def _four(**over):
    stages = {"compute_wall_s": 372.0, "end_to_end_in_container_s": 592.0, "evaluated_studies": 3000}
    arm = lambda a, w: {"container_class": "Modal A100 + 8 cores + 64 GiB", "workers": w, "split": {"reimage_share": 0.98},
                        "host": {"gpu": "NVIDIA A100-SXM4-40GB, 40960 MiB, 610"},
                        "stages": {**stages, "evaluated_studies": 2000 if a == "lab" else 3000}}
    d = {"status": "MEASURED", "design": {"studies_per_stream": 1000, "streams": {"calibration": 101, "null": 202, "alternative": 303}}, "parity": {"all": True},
         "labels": {"compute": "MEASURED", "outcomes": "SYNTHETIC (atlas/simulation.py at 79a7792)", "rates": "ESTIMATE (x)"},
         "arms": {a: arm(a, 1 if a == "gpu" else 8) for a in ("lab", "numba", "gpu", "hybrid")}}
    return d | over


def test_the_four_arm_file_fills_exactly_the_four_declared_lanes():
    rows = dict(cr.rows_from(Path("modal-design.json"), _four()))
    assert sorted(rows) == ["gpu-1", "hybrid", "lab-cpu", "preflight-cpu", "scope"]
    h = rows["hybrid"]
    assert (h["seconds"], h["startup_s"], h["cores"], h["gpu"]) == (372.0, 220.0, 8, "1x NVIDIA A100-SXM4-40GB")
    assert rows["gpu-1"]["cores"] is None and rows["lab-cpu"]["gpu"] is None and rows["lab-cpu"]["seconds"] == 372.0
    three = _four()
    del three["arms"]["hybrid"]
    for bad in (three, _four(status="FAILED"), _four(parity={"all": False}), _four(design={"studies_per_stream": 20, "streams": {}})):
        assert list(cr.rows_from(Path("modal-design.json"), bad)) == []


def test_interim_seconds_display_as_the_raw_file_rounds():
    """Prometheus (21 Nov): the NumPy median 2.2353 s shows as 2.24 s, never 2.23 s from rounding twice.

    race.js `secs` shows two decimals under 10 s and one above; the stored value must give the same text
    as the file's raw value would.
    """
    show = lambda s: f"{s:.2f}" if s < 10 else f"{s:.1f}"
    [(_, groups)] = cr.rows_from(Path(CLEARED), LADDER)
    rows = {r["id"]: r["seconds"] for g in groups for r in g["rows"]}
    raw = {"kernel-numpy": LADDER["kernel"]["rows"]["numpy_reference"]["median_s"],
           "kernel-gpu": LADDER["kernel"]["rows"]["gpu"]["median_s"],
           "fleet-1": LADDER["fleet"]["runs"]["1"]["compute_span_s"], "fleet-8": LADDER["fleet"]["runs"]["8"]["compute_span_s"]}
    assert {i: show(rows[i]) for i in raw} == {i: show(v) for i, v in raw.items()}
    assert show(rows["kernel-numpy"]) == "2.24"


def _fused_parity():
    """bench/design_batch.prior_parity's shape: X for every arm; maxT for the batched arms (not the lab's permutation)."""
    maxT = {"maxT_max_rel_diff": 1.6e-15, "maxT_within_tol": True, "maxT_compared": 3000}
    return {"all": True, "against_prior": {"fused": {"lab": {"x_bitwise": True, "x_compared": 2000}, "all": True,
                                                     **{a: {"x_bitwise": True, "x_compared": 3000, **maxT}
                                                        for a in ("numba", "gpu", "hybrid")}}}}


def test_the_five_arm_file_fills_only_the_fused_lane_and_only_with_full_x_parity():
    """Lead (21 Nov): a fifth lane, fused-cpu, from the --compare-with file, only with full X parity."""
    five = _four()
    for a in ("lab", "numba", "gpu", "hybrid"):
        five["arms"][a]["source"] = "bench/results/modal-design-four.json (source_commit x)"
    five["arms"]["fused"] = {**five["arms"]["numba"], "source": "this run"}
    five["parity"] = _fused_parity()
    rows = dict(cr.rows_from(Path("modal-design.json"), five))
    assert sorted(rows) == ["fused-cpu", "scope"]
    f = rows["fused-cpu"]
    assert (f["seconds"], f["cores"], f["gpu"]) == (372.0, 8, None)
    assert f["note"].endswith("of summed worker wall time; same features, same top score; run on a different host "
                              "of the same container class, about 2 h after the lab, Numba and A100 lanes and 70 min after "
                              "the hybrid; one sweep here. 3 fresh-container reruns at #120 0b27ea1 "
                              "(bench/results/repeats-20261121T2209Z.json, #126): median 0.0424 s/study, range "
                              "0.0422-0.0425, 3,000 studies each; descriptive")
    assert "fused lane: same features, same top score" in rows["scope"]["comparison"]
    five["parity"]["against_prior"]["fused"]["gpu"]["x_bitwise"] = False
    assert list(cr.rows_from(Path("modal-design.json"), five)) == []
    five["parity"]["against_prior"]["fused"]["gpu"] = {"x_bitwise": True, "x_compared": 0}
    assert list(cr.rows_from(Path("modal-design.json"), five)) == []


def test_a_four_arm_file_refreshes_every_lane_even_when_arms_are_copied_from_an_earlier_file():
    """Prometheus #120 round 2 (1): 194458Z copies lab, numba and gpu from 185537Z; it must still refresh them."""
    four = _four()
    for a in ("lab", "numba", "gpu"):
        four["arms"][a]["source"] = "bench/results/modal-design-earlier.json (source_commit x)"
    assert sorted(dict(cr.rows_from(Path("modal-design.json"), four))) == ["gpu-1", "hybrid", "lab-cpu",
                                                                           "preflight-cpu", "scope"]


def test_the_fused_lane_needs_complete_maxT_parity_on_every_batched_arm():
    """Prometheus #120 round 2 (2): partial or false maxT parity refuses the lane; X parity alone is not enough."""
    five = _four()
    five["arms"]["fused"] = {**five["arms"]["numba"], "source": "this run"}
    five["parity"] = _fused_parity()
    assert "fused-cpu" in dict(cr.rows_from(Path("modal-design.json"), five))
    bad = [("numba", "maxT_within_tol", False), ("gpu", "maxT_max_rel_diff", 2e-9), ("hybrid", "maxT_max_rel_diff", None),
           ("gpu", "maxT_within_tol", None), ("numba", "maxT_max_rel_diff", float("nan"))]
    for arm, key, value in bad:
        five["parity"] = _fused_parity()
        e = five["parity"]["against_prior"]["fused"][arm]
        e.pop(key) if value is None else e.update({key: value})
        assert list(cr.rows_from(Path("modal-design.json"), five)) == [], (arm, key, value)
    five["parity"] = _fused_parity()
    five["parity"]["against_prior"]["fused"]["all"] = False
    assert list(cr.rows_from(Path("modal-design.json"), five)) == []
    five["parity"] = _fused_parity()
    del five["parity"]["against_prior"]["fused"]["hybrid"]
    assert list(cr.rows_from(Path("modal-design.json"), five)) == []


def test_the_fused_lane_needs_full_coverage_not_one_compared_study():
    """Lead ruling on #120 (escalation): x_compared must be every shared study (2,000 lab, 3,000 batched), per arm."""
    five = _four()
    five["arms"]["fused"] = {**five["arms"]["numba"], "source": "this run"}
    five["parity"] = _fused_parity()
    assert "fused-cpu" in dict(cr.rows_from(Path("modal-design.json"), five))
    for arm in ("lab", "numba", "gpu", "hybrid"):
        five["parity"] = _fused_parity()
        five["parity"]["against_prior"]["fused"][arm]["x_compared"] = 1
        assert list(cr.rows_from(Path("modal-design.json"), five)) == [], arm
    five["parity"] = _fused_parity()
    five["parity"]["against_prior"]["fused"]["lab"]["x_compared"] = 3000  # more than the lab ran: not its coverage
    assert list(cr.rows_from(Path("modal-design.json"), five)) == []


def test_the_fused_lane_binds_to_the_planned_studies_even_when_every_observed_count_is_one():
    """Lead ruling on #120 (last round): the plan comes from the design (2,000 lab, 3,000 batched), never from counts."""
    five = _four()
    five["arms"]["fused"] = {**five["arms"]["numba"], "source": "this run"}
    for a in five["arms"]:
        five["arms"][a]["stages"] = {**five["arms"][a]["stages"], "evaluated_studies": 1}
    five["parity"] = _fused_parity()
    for e in five["parity"]["against_prior"]["fused"].values():
        if isinstance(e, dict):
            e.update(x_compared=1, **({"maxT_compared": 1} if "maxT_within_tol" in e else {}))
    assert list(cr.rows_from(Path("modal-design.json"), five)) == []


def test_maxT_coverage_is_inferred_from_X_only_for_the_legacy_064bd6e_file(capsys):
    """Lead ruling on #120: a file without maxT_compared is refused unless it is the legacy 064bd6e result."""
    five = _four()
    five["arms"]["fused"] = {**five["arms"]["numba"], "source": "this run"}
    five["parity"] = _fused_parity()
    for a in ("numba", "gpu", "hybrid"):
        del five["parity"]["against_prior"]["fused"][a]["maxT_compared"]
    for commit in (None, "1234567890abcdef", "064bd6e"):  # absent, another commit, a bare prefix
        five["source_commit"] = commit
        assert list(cr.rows_from(Path("modal-design.json"), five)) == [], commit
    five["source_commit"] = "064bd6e95f721c8be7747d4cfef5eb5bd4bbd79d"
    capsys.readouterr()
    assert "fused-cpu" in dict(cr.rows_from(Path("modal-design.json"), five))
    assert "maxT coverage inferred from X coverage (legacy file, prior_parity at 064bd6e)" in capsys.readouterr().err
