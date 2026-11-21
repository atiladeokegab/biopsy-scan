import json
from pathlib import Path

import pytest

from bench import repeats as rp


def raw(arm, wall, n, status="MEASURED", workers=8):
    return {"status": status, "source_commit": "c", "arms": {arm: {
        "workers": workers, "host": {"gpu": "A100", "cpu_count": 24}, "warm_s": 1.0,
        "stages": {"studies_wall_s": wall, "compute_wall_s": wall, "evaluated_studies": n,
                   "wall_per_study_s": wall / n, "end_to_end_in_container_s": wall + 10}}},
        "modal": {"started_utc": "s", "finished_utc": "f", "wall_s": 99.0}}


def test_extract_keeps_one_arms_wall_and_refuses_an_unmeasured_run(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(raw("lab", 200.0, 200)))
    r = rp.extract(p, "lab")
    assert r["wall_per_study_s"] == pytest.approx(1.0) and r["evaluated_studies"] == 200
    assert len(r["file_sha256"]) == 64
    p.write_text(json.dumps(raw("lab", 200.0, 200, status="FAILED")))
    with pytest.raises(ValueError, match="MEASURED"):
        rp.extract(p, "lab")


def test_a_clean_single_arm_run_is_accepted_and_checked_against_the_prior(tmp_path):
    one = raw("lab", 200.0, 200, status="FAILED")
    one.update(parity={"all": False, "this_run": "one arm"}, clean_tree=True, errors={}, same_container_class=True)
    one["arms"]["lab"]["studies"] = [{"stream": "null", "i": 0, "x_sha": "x", "maxT": 3.0, "p": 0.5}]
    prior = {"arms": {"lab": {"studies": [{"stream": "null", "i": 0, "x_sha": "x", "maxT": 3.0, "p": 0.5}]},
                      "numba": {"studies": [{"stream": "null", "i": 0, "x_sha": "x", "maxT": 2.0}]}}}
    p = tmp_path / "a.json"
    p.write_text(json.dumps(one))
    assert rp.extract(p, "lab", prior)["parity_vs_prior"]["all"] is True
    prior["arms"]["lab"]["studies"][0]["p"] = 0.4
    assert rp.extract(p, "lab", prior)["parity_vs_prior"]["all"] is False
    prior["arms"]["lab"]["studies"][0]["p"] = 0.5
    prior["arms"]["numba"]["studies"][0]["x_sha"] = "y"
    assert rp.extract(p, "lab", prior)["parity_vs_prior"]["all"] is False
    got = rp.extract(p, "lab", prior)
    assert got["runner_status"] == "FAILED" and got["runner_status_note"] == "FAILED: single-arm, no in-run parity"


@pytest.mark.parametrize("spoil", [{"clean_tree": False}, {"errors": {"lab": "RuntimeError()"}},
                                   {"same_container_class": False}, {"parity": {"all": False, "reason": "an arm failed"}}])
def test_a_repeat_with_a_dirty_tree_errors_or_a_real_parity_failure_is_rejected(tmp_path, spoil):
    one = raw("lab", 200.0, 200, status="FAILED")
    one.update(parity={"all": False, "this_run": "one arm"}, clean_tree=True, errors={}, same_container_class=True)
    one.update(spoil)
    p = tmp_path / "a.json"
    p.write_text(json.dumps(one))
    with pytest.raises(ValueError, match="need MEASURED or a clean single-arm run"):
        rp.extract(p, "lab")


def test_summary_is_median_and_range_and_the_ratio_range_is_the_extreme_pairs():
    reps = {"lab": [{"wall_per_study_s": v} for v in (0.9, 0.8, 1.0)],
            "fused": [{"wall_per_study_s": v} for v in (0.05, 0.04, 0.045)]}
    s = rp.summarize(reps)
    assert s["lab"] == {"n": 3, "median": 0.9, "min": 0.8, "max": 1.0}
    assert s["fused"]["median"] == pytest.approx(0.045)
    assert s["ratio"]["median"] == pytest.approx(0.9 / 0.045)
    assert s["ratio"]["min"] == pytest.approx(0.8 / 0.05) and s["ratio"]["max"] == pytest.approx(1.0 / 0.04)


def test_the_committed_summary_matches_its_raw_repeats():
    found = sorted((Path(__file__).parent / "results").glob("repeats-*.json"))
    if not found:
        pytest.skip("no committed repeats result yet")
    r = json.loads(found[-1].read_text())
    assert r["summary"] == rp.summarize(r["repeats"])
    assert all(len(v) == 3 for v in r["repeats"].values())
    assert r["status"] == rp.STATUS and all(x["parity_vs_prior"]["all"] for v in r["repeats"].values() for x in v)
    raw_dir = Path(__file__).parent / "results" / "repeats-raw"
    for arm, reps in r["repeats"].items():  # every stored repeat is exactly what its committed raw file says
        for x in reps:
            again = rp.extract(raw_dir / x["file"], "fused" if arm == rp.PREVIOUS else arm)
            assert {k: v for k, v in x.items() if k != "parity_vs_prior"} == again
            assert x["parity_vs_prior"]["against"] == r["parity"]["prior"]["file"]
    for arm, reps in r["repeats"].items():
        for x in reps:
            assert x["wall_per_study_s"] == pytest.approx(x["studies_wall_s"] / x["evaluated_studies"])


def test_every_raw_repeat_is_in_the_spend_ledger_and_the_totals_split_actual_from_estimate():
    found = sorted((Path(__file__).parent / "results").glob("repeats-*.json"))
    if not found:
        pytest.skip("no committed repeats result yet")
    r = json.loads(found[-1].read_text())
    ledger = json.loads((Path(__file__).parent.parent / r["spend"]["log"]).read_text())
    ran = {Path(e["actual"]["result"]).name for e in ledger["entries"] if "result" in e["actual"]}
    for v in r["repeats"].values():  # a lost ledger line would under-report the spend
        for x in v:
            assert x["file"].split("-", 3)[3] in ran, x["file"]
    done, aborted = ledger["completed_usd"], ledger["aborted_usd"]
    assert "MEASURED" in done["label"] and "ESTIMATE" in aborted["label"]
    assert ledger["running_total_usd"] == pytest.approx(done["value"] + aborted["value"])
    assert ledger["running_total_usd"] == pytest.approx(sum(e["actual"]["est_usd"] for e in ledger["entries"]))
    assert r["spend"] == {"log": r["spend"]["log"], **{k: ledger[k] for k in
                          ("running_total_usd", "completed_usd", "aborted_usd")}}
