import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from bench import repeats as rp

RESULTS = Path(__file__).parent / "results"
DESIGN = {"seed": 0, "root_seed": 20261121, "streams": {"null": 202, "alternative": 303}, "studies_per_stream": 2}
LAB_PLAN = {"calibration": 0, "null": 2, "alternative": 2}
FUSED_PLAN = {"calibration": 2, "null": 2, "alternative": 2}
PRIOR_N = 3  # the prior runs more studies per stream than the reruns, like #113's 1,000 against 100


def studies(plan, lab):
    return [{"stream": s, "i": i, "x_sha": f"x-{s}-{i}", "maxT": 2.0 + i, **({"p": 0.5} if lab else {})}
            for s, n in plan.items() for i in range(n)]


def run(arm, wall=4.0, task="ta-1", workers=8, cls="Modal A100 + 8 cores + 64 GiB", gpu="A100", cpu=8.0, mib=65536):
    """A single-arm runner result shaped like #120's bench/modal_design.py output."""
    plan = LAB_PLAN if arm == "lab" else FUSED_PLAN
    ss = studies(plan, arm == "lab")
    return {"status": "FAILED", "parity": dict(rp.ONE_ARM), "clean_tree": True, "errors": {},
            "same_container_class": True, "source_commit": rp.HARNESS["commit"], "design": copy.deepcopy(DESIGN),
            "arms": {arm: {"workers": workers, "warm_s": 1.0, "container_class": cls,
                           "host": {"modal_task_id": task}, "summary": {"studies": plan}, "studies": ss,
                           "stages": {"studies_wall_s": wall, "compute_wall_s": wall, "evaluated_studies": len(ss),
                                      "wall_per_study_s": wall / len(ss), "end_to_end_in_container_s": wall + 10}}},
            "modal": {"gpu": gpu, "cpu": cpu, "memory_mib": mib, "wall_s": 99.0}}


def prior():
    lab, fused = ({k: PRIOR_N if v else 0 for k, v in plan.items()} for plan in (LAB_PLAN, FUSED_PLAN))
    return {"status": "MEASURED", "source_commit": "p", "design": dict(DESIGN, studies_per_stream=PRIOR_N), "arms": {
        "lab": {"summary": {"studies": lab}, "studies": studies(lab, True)},
        "numba": {"summary": {"studies": fused}, "studies": studies(fused, False)}}}


def pinned(monkeypatch, path):
    """Stand a fixture prior in for #113's file by pinning its digest."""
    monkeypatch.setattr(rp, "PRIOR_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def test_a_clean_single_arm_run_is_accepted_and_checked_against_the_prior(tmp_path):
    got = rp.extract(write(tmp_path, "a.json", run("lab")), "lab", prior())
    assert got["parity_vs_prior"]["all"] is True and got["parity_vs_prior"]["statistic_compared"] == 4
    assert got["runner_status"] == "FAILED" and got["runner_status_note"] == "FAILED: single-arm, no in-run parity"
    assert got["wall_per_study_s"] == pytest.approx(1.0) and len(got["file_sha256"]) == 64
    for arm, key, val, at in [("lab", "p", 0.4, 0), ("numba", "x_sha", "y", PRIOR_N)]:  # numba[PRIOR_N] is null 0
        pr = prior()
        pr["arms"][arm]["studies"][at][key] = val
        assert rp.extract(write(tmp_path, "a.json", run("lab")), "lab", pr)["parity_vs_prior"]["all"] is False


@pytest.mark.parametrize("spoil", [{"clean_tree": False}, {"errors": {"lab": "RuntimeError()"}},
                                   {"same_container_class": False}, {"parity": {"all": False, "reason": "an arm failed"}},
                                   {"status": "ERROR"}])
def test_a_repeat_with_a_dirty_tree_errors_or_a_real_parity_failure_is_rejected(tmp_path, spoil):
    with pytest.raises(ValueError, match="need MEASURED or a clean single-arm run"):
        rp.extract(write(tmp_path, "a.json", {**run("lab"), **spoil}), "lab")


def _truncate(r, a):
    r["arms"][a]["studies"] = r["arms"][a]["studies"][:1]


def _duplicate(r, a):
    r["arms"][a]["studies"] = [r["arms"][a]["studies"][0]] * len(r["arms"][a]["studies"])


def _unplanned(r, a):
    r["arms"][a]["studies"][0]["i"] = 99


def _nan_maxT(r, a):
    r["arms"][a]["studies"][0]["maxT"] = math.nan


def _nan_p(r, a):
    r["arms"][a]["studies"][0]["p"] = math.nan


def _count_mismatch(r, a):
    st = r["arms"][a]["stages"]
    st["evaluated_studies"] += 1
    st["wall_per_study_s"] = st["studies_wall_s"] / st["evaluated_studies"]


@pytest.mark.parametrize("arm", ["lab", "fused"])
@pytest.mark.parametrize("spoil, match", [(_truncate, "cover the plan"), (_duplicate, "duplicate"),
                                          (_unplanned, "cover the plan"), (_nan_maxT, "non-finite"),
                                          (_count_mismatch, "study records for")])
def test_coverage_duplicates_and_nonfinite_statistics_are_rejected_before_parity(tmp_path, arm, spoil, match):
    r = run(arm)
    spoil(r, arm)
    with pytest.raises(ValueError, match=match):
        rp.extract(write(tmp_path, "a.json", r), arm, prior())


def test_a_nan_lab_p_is_rejected(tmp_path):
    r = run("lab")
    _nan_p(r, "lab")
    with pytest.raises(ValueError, match="non-finite"):
        rp.extract(write(tmp_path, "a.json", r), "lab", prior())


@pytest.mark.parametrize("spoil", [_duplicate, _nan_maxT])
def test_the_prior_is_validated_before_it_is_keyed(tmp_path, monkeypatch, spoil):
    pr = prior()
    spoil(pr, "numba")
    with pytest.raises(ValueError, match="prior numba"):
        rp.load_prior(pinned(monkeypatch, write(tmp_path, "prior.json", pr)))


@pytest.mark.parametrize("arm", ["lab", "fused"])
def test_a_consistently_truncated_run_fails_against_the_declared_plan(tmp_path, arm):
    # Atlas round 2: one record, with summary, count and denominator all rewritten to match, still declares 2 per stream
    r = run(arm)
    a = r["arms"][arm]
    a["studies"] = a["studies"][:1]
    a["summary"]["studies"] = {st: int(st == a["studies"][0]["stream"]) for st in FUSED_PLAN}
    a["stages"].update(evaluated_studies=1, wall_per_study_s=a["stages"]["studies_wall_s"])
    with pytest.raises(ValueError, match="not the declared plan"):
        rp.extract(write(tmp_path, "a.json", r), arm, prior())


def test_only_the_pinned_prior_is_accepted(tmp_path):
    real = RESULTS / "modal-design-20261121T194458Z.json"
    assert rp.load_prior(real)["_sha256"] == rp.PRIOR_SHA256
    forged = json.loads(real.read_text())
    forged["source_commit"] = "unapproved-prior"
    with pytest.raises(ValueError, match="not the pinned"):
        rp.load_prior(write(tmp_path, real.name, forged))
    with pytest.raises(ValueError, match="not the pinned"):
        rp.load_prior(write(tmp_path, "fixture.json", prior()))


def test_a_different_design_or_seed_is_rejected(tmp_path):
    r = run("fused")
    r["design"]["root_seed"] += 1
    with pytest.raises(ValueError, match="design or seeds"):
        rp.extract(write(tmp_path, "a.json", r), "fused", prior())


@pytest.mark.parametrize("key, value, match", [("studies_wall_s", -4.0, "positive and finite"),
                                               ("compute_wall_s", math.inf, "positive and finite"),
                                               ("wall_per_study_s", 9.0, "studies_wall_s / evaluated"),
                                               ("evaluated_studies", 4.0, "positive integer"),
                                               ("evaluated_studies", 0, "positive integer")])
def test_negative_nonfinite_or_inconsistent_timings_are_rejected(tmp_path, key, value, match):
    r = run("fused")
    r["arms"]["fused"]["stages"][key] = value
    with pytest.raises(ValueError, match=match):
        rp.extract(write(tmp_path, "a.json", r), "fused", prior())


def _three(tmp_path, arm, tasks=("t1", "t2", "t3")):
    return [rp.extract(write(tmp_path, f"{arm}-{i}.json", run(arm, wall=4.0 + i, task=t)), arm, prior())
            for i, t in enumerate(tasks)]


def test_the_same_run_counted_twice_is_rejected_even_renamed(tmp_path):
    ok = {"lab": _three(tmp_path, "lab", ("a", "b", "c")), "fused": _three(tmp_path, "fused", ("d", "e", "f"))}
    rp.check_repeats(ok)
    same = {"lab": [ok["lab"][0]] * 3, "fused": ok["fused"]}
    with pytest.raises(ValueError, match="counted twice"):
        rp.check_repeats(same)
    renamed = write(tmp_path, "renamed.json", json.loads((tmp_path / "lab-0.json").read_text()))
    copy_ = rp.extract(renamed, "lab", prior())
    with pytest.raises(ValueError, match="counted twice"):
        rp.check_repeats({"lab": ok["lab"][:2] + [copy_], "fused": ok["fused"]})
    rerun = run("lab", wall=4.0, task="a")  # different bytes, same Modal task
    rerun["modal"]["wall_s"] = 98.0
    with pytest.raises(ValueError, match=r"counted twice \(run_id\)"):
        rp.check_repeats({"lab": ok["lab"][:2] + [rp.extract(write(tmp_path, "r.json", rerun), "lab", prior())],
                          "fused": ok["fused"]})


def test_a_uniform_but_wrong_class_is_rejected(tmp_path):
    l4 = dict(workers=4, cls="Modal L4 + 4 cores + 32 GiB", gpu="L4", cpu=4.0, mib=32768)
    reps = {arm: [rp.extract(write(tmp_path, f"{arm}{i}.json", run(arm, wall=4.0 + i, task=f"{arm}{i}", **l4)), arm,
                             prior()) for i in range(3)] for arm in ("lab", "fused")}
    with pytest.raises(ValueError, match="not the declared scope"):
        rp.check_repeats(reps)


def test_mixed_workers_are_rejected(tmp_path):
    reps = {"lab": _three(tmp_path, "lab", ("a", "b", "c")), "fused": _three(tmp_path, "fused", ("d", "e", "f"))}
    reps["fused"][0] = rp.extract(write(tmp_path, "w.json", run("fused", task="g", workers=4)), "fused", prior())
    with pytest.raises(ValueError, match="not the declared scope"):
        rp.check_repeats(reps)


def _cli(tmp_path, monkeypatch, pr):
    for i, t in enumerate(("a", "b", "c")):
        write(tmp_path, f"lab{i}.json", run("lab", wall=400.0 + i, task=t))
        write(tmp_path, f"fused{i}.json", run("fused", wall=24.0 + i, task="f" + t))
    spend = write(tmp_path, "spend.json", {"running_total_usd": 1.0, "completed_usd": {"value": 1.0},
                                           "aborted_usd": {"value": 0.0}})
    out = tmp_path / "out.json"
    args = ["--lab", *[str(tmp_path / f"lab{i}.json") for i in range(3)],
            "--fused", *[str(tmp_path / f"fused{i}.json") for i in range(3)],
            "--prior", str(pinned(monkeypatch, write(tmp_path, "prior.json", pr))), "--spend", str(spend),
            "--output", str(out)]
    return rp.main(args), out


def test_the_cli_fails_closed_on_an_external_mismatch(tmp_path, monkeypatch, capsys):
    code, out = _cli(tmp_path, monkeypatch, prior())
    assert code == 0 and json.loads(out.read_text())["status"] == rp.STATUS
    pr = prior()
    pr["arms"]["numba"]["studies"][3]["x_sha"] = "changed"
    bad = tmp_path / "bad"
    bad.mkdir()
    code, out = _cli(bad, monkeypatch, pr)
    assert code == 1 and not out.exists()  # no file, so no MEASURED headline
    assert "FAILED: a repeat failed the external parity check" in capsys.readouterr().err


def test_summary_is_median_and_range_and_the_ratio_range_is_the_extreme_pairs():
    reps = {"lab": [{"wall_per_study_s": v} for v in (0.9, 0.8, 1.0)],
            "fused": [{"wall_per_study_s": v} for v in (0.05, 0.04, 0.045)]}
    s = rp.summarize(reps)
    assert s["lab"] == {"n": 3, "median": 0.9, "min": 0.8, "max": 1.0}
    assert s["fused"]["median"] == pytest.approx(0.045)
    assert s["ratio"]["median"] == pytest.approx(0.9 / 0.045)
    assert s["ratio"]["min"] == pytest.approx(0.8 / 0.05) and s["ratio"]["max"] == pytest.approx(1.0 / 0.04)


def _committed():
    found = sorted(RESULTS.glob("repeats-*.json"))
    if not found:
        pytest.skip("no committed repeats result yet")
    return json.loads(found[-1].read_text())


def test_the_committed_result_is_reproduced_from_raw_runs_and_the_pinned_prior(tmp_path):
    r = _committed()
    prior_path = RESULTS / r["parity"]["prior"]["file"]
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == rp.PRIOR_SHA256 == r["parity"]["prior"]["sha256"]
    raw = RESULTS / "repeats-raw"
    files = {arm: [str(raw / x["file"]) for x in v] for arm, v in r["repeats"].items()}
    out = tmp_path / "again.json"
    assert rp.main(["--lab", *files["lab"], "--fused", *files["fused"], "--fused-previous", *files[rp.PREVIOUS],
                    "--prior", str(prior_path), "--spend", r["spend"]["log"], "--output", str(out)]) == 0
    assert json.loads(out.read_text()) == r  # parity recomputed from the prior, not trusted from saved flags
    assert all(len(v) == 3 for v in r["repeats"].values())
    assert "not an uncertainty interval" in r["ratio_note"] and "200" in r["scope"] and "3,000" in r["scope"]


def test_every_raw_repeat_is_in_the_spend_ledger_and_every_cost_is_an_estimate():
    r = _committed()
    ledger = json.loads((Path(__file__).parent.parent / r["spend"]["log"]).read_text())
    ran = {Path(e["actual"]["result"]).name for e in ledger["entries"] if "result" in e["actual"]}
    for v in r["repeats"].values():  # a lost ledger line would under-report the spend
        for x in v:
            assert x["file"].split("-", 3)[3] in ran, x["file"]
    done, aborted = ledger["completed_usd"], ledger["aborted_usd"]
    assert done["label"].startswith("ESTIMATE") and aborted["label"].startswith("ESTIMATE")
    assert "MEASURED" not in done["label"] + aborted["label"] and r["labels"]["spend"].startswith("ESTIMATE")
    assert ledger["running_total_usd"] == pytest.approx(done["value"] + aborted["value"])
    assert ledger["running_total_usd"] == pytest.approx(sum(e["actual"]["est_usd"] for e in ledger["entries"]))
    assert r["spend"] == {"log": r["spend"]["log"], **{k: ledger[k] for k in
                          ("running_total_usd", "completed_usd", "aborted_usd")}}
