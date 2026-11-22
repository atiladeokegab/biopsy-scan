import copy
import json

import pytest

from bench import sweep_repeats as s

TASK = {"fleet": lambda d: d["kernel"]["gpu"], "matched": lambda d: d["host"],
        "scaling": lambda d: d["points"]["1"]["host"]}
COMMIT = {"fleet": "commit", "matched": "source_commit", "scaling": "source_commit"}


def repeat(kind, k, edit=None):
    d = copy.deepcopy(s.load_prior(kind))
    if kind == "fleet":  # what integration's stats.screen gives on this synthetic null (the prior pinned 8add7f9)
        for b in d["screen"]["backends"].values():
            if isinstance(b, dict):
                b["p"] = s.SCREEN_P_AT_HEAD
        d["gpu_requested"] = s.GPU_REQUESTED  # the prior asked for any A100 and recorded no per-container models
        for n in s.FLEET:
            d["fleet"]["runs"][n].update(gpus=[s.GPU_MODEL], gpu_models=[s.GPU_MODEL] * int(n), same_gpu_model=True)
    TASK[kind](d)["modal_task_id"] = f"ta-{k}"
    d[COMMIT[kind]] = "c0ffee"
    if edit:
        edit(d)
    return d


def raw(tmp_path, kind, docs):
    (tmp_path / kind).mkdir()
    for k, d in enumerate(docs):
        (tmp_path / kind / f"r{k}.json").write_text(json.dumps(d))
    (tmp_path / "commits.json").write_text(json.dumps({kind: {"commit": "c0ffee", "repeats": len(docs)}}))
    return tmp_path


@pytest.mark.parametrize("kind", ["fleet", "scaling", "matched"])
def test_three_clean_repeats_of_the_prior_reduce_to_its_own_numbers(tmp_path, kind):
    out = s.reduce(kind, sorted((raw(tmp_path, kind, [repeat(kind, k) for k in range(3)]) / kind).glob("*")), "c0ffee")
    prior = s.REDUCE[kind](repeat(kind, 9), s.load_prior(kind), kind)["values"]
    assert {k: v["median"] for k, v in out["spread"].items()} == prior


def test_headlines_match_the_single_sweeps_they_repeat():
    v = {k: s.REDUCE[k](repeat(k, 0), s.load_prior(k), k)["values"] for k in ("fleet", "scaling", "matched")}
    assert round(v["fleet"]["speedup_compute_span"], 1) == 6.6
    assert round(v["scaling"]["speedup_64_cores"], 1) == 38.2
    assert round(v["matched"]["ratio"], 1) == 42.1


def lab_p(d):
    d["per_study"][3]["lab_p"] += 0.01


def maxT_far(d):
    d["per_study"][3]["preflight_maxT"] += 1e-6


def test_preflight_maxT_may_differ_by_rounding_across_hosts_but_no_more():
    d = repeat("matched", 0)
    d["per_study"][3]["preflight_maxT"] += 2e-15  # what a fresh host gave (fair runs, 1.8e-15)
    s.matched(d, s.load_prior("matched"), "matched")
    maxT_far(d)
    with pytest.raises(s.Bad, match="differs from the prior"):
        s.matched(d, s.load_prior("matched"), "matched")


def test_the_screen_hooks_maxT_may_move_by_an_ulp_across_hosts():
    d = repeat("fleet", 0)
    d["screen"]["backends"]["numpy"]["maxT"] = 3.058938978093486  # fleet repeat 3's host
    s.fleet(d, s.load_prior("fleet"), "fleet")
    d["screen"]["backends"]["numpy"]["maxT"] = 3.06
    with pytest.raises(s.Bad, match="maxT differs"):
        s.fleet(d, s.load_prior("fleet"), "fleet")


def screen_old_p(d):
    d["screen"]["backends"]["cupy"]["p"] = 0.789


def fleet_bits(d):
    d["fleet"]["runs"]["8"]["bit_identical_to_1"] = False


def scaling_x(d):
    d["studies_by_cores"]["64"][5]["x_sha"] = "0" * 64


def scaling_dup(d):
    d["studies_by_cores"]["8"][1] = d["studies_by_cores"]["8"][0]


def nan_wall(d):
    d["arms"]["lab"]["stages"]["studies_wall_s"] = float("nan")


# Atlas's review of #134: each claim must follow from the rows and operands, not from cached totals or flags
def all_agree(d):
    d["decisions"]["agree"] = d["decisions"]["compared"]


def setup_ratio_99(d):
    d["wall"]["with_setup"]["ratio"] = 99.0


def ratio_cached(d):
    d["wall"]["ratio"] *= 2
    d["wall"]["lab_s_per_study"] *= 2


def features_differ(d):
    d["per_study"][-1]["x_equal"] = False


def call_not_from_p(d):
    r = d["per_study"][0]
    r["lab_call"] = not r["lab_call"]


def mixed_gpus(d):
    r = d["fleet"]["runs"]["8"]
    r.update(same_gpu_model=False, gpu_models=["NVIDIA A100-SXM4-40GB"] * 7 + ["NVIDIA A100-SXM4-80GB"],
             gpus=["NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB"])


def other_gpu_request(d):
    d["gpu_requested"] = "A100"


def scaling_empty(d):
    d["studies_by_cores"] = {}


def scaling_point_missing(d):
    d["points"].pop("64")


def scaling_curve_dup(d):
    d["curve"].append(d["curve"][0])


def other_class(d):
    d["modal"]["cpu"] = 16.0


def fewer_containers(d):
    d["fleet"]["runs"]["8"]["containers_seen"] = 7


def missing_lab_study(d):
    d["per_study"].pop()


@pytest.mark.parametrize("kind,edit,why", [
    ("matched", lab_p, "differs from the prior"),
    ("matched", maxT_far, "differs from the prior"),
    ("matched", nan_wall, "positive and finite"),
    ("matched", other_class, "container class"),
    ("matched", missing_lab_study, "exactly once"),
    ("fleet", fleet_bits, "a check failed"),
    ("fleet", screen_old_p, "screen hook's p"),
    ("fleet", fewer_containers, "a check failed"),
    ("scaling", scaling_x, "differs from the prior"),
    ("scaling", scaling_dup, "exactly once"),
    ("matched", all_agree, "decision totals"),
    ("matched", setup_ratio_99, "does not follow"),
    ("matched", ratio_cached, "does not follow"),
    ("matched", features_differ, "features differ"),
    ("matched", call_not_from_p, "do not follow from its p"),
    ("fleet", mixed_gpus, "GPUs"),
    ("fleet", other_gpu_request, "GPUs"),
    ("scaling", scaling_empty, "one per core count"),
    ("scaling", scaling_point_missing, "one per core count"),
    ("scaling", scaling_curve_dup, "cores"),
])
def test_a_bad_repeat_fails_closed_and_writes_no_file(tmp_path, capsys, kind, edit, why):
    d = raw(tmp_path, kind, [repeat(kind, 0), repeat(kind, 1), repeat(kind, 2, edit)])
    out = tmp_path / "out.json"
    assert s.main([str(d), str(out)]) == 1
    assert not out.exists() and why in capsys.readouterr().err


def test_duplicate_runs_wrong_commit_and_too_few_repeats_are_refused(tmp_path):
    same = repeat("matched", 0)
    with pytest.raises(s.Bad, match="appears twice"):
        s.reduce("matched", sorted((raw(tmp_path, "matched", [same, repeat("matched", 1), same]) / "matched").glob("*")), "c0ffee")
    with pytest.raises(s.Bad, match="ran c0ffee, not deadbee"):
        s.reduce("matched", sorted((tmp_path / "matched").glob("*")), "deadbeef")
    with pytest.raises(s.Bad, match="2 repeats, not the declared 3"):
        s.reduce("matched", sorted((tmp_path / "matched").glob("*"))[:2], "c0ffee")
    with pytest.raises(s.Bad, match="1 repeats"):
        s.reduce("matched", sorted((tmp_path / "matched").glob("*"))[:1], "c0ffee", want=1)


def test_a_failed_attempt_is_listed_with_its_reason_and_kept_out_of_the_spread(tmp_path):
    d = raw(tmp_path, "fleet", [repeat("fleet", 0), repeat("fleet", 1)])
    (d / "fleet-failed").mkdir()
    (d / "fleet-failed" / "f.json").write_text(json.dumps(
        {"status": "FAILED", "commit": "c0ffee", "parity_failed": ["fleet.runs.8.same_gpu_model"]}))
    assert s.main([str(d), str(tmp_path / "out.json")]) == 0
    out = json.loads((tmp_path / "out.json").read_text())["fleet"]
    assert len(out["repeats"]) == 2 and out["failed_attempts"][0]["why"] == ["fleet.runs.8.same_gpu_model"]
    (d / "fleet-failed" / "f.json").write_text(json.dumps({"status": "MEASURED", "commit": "c0ffee"}))
    assert s.main([str(d), str(tmp_path / "out2.json")]) == 1


def test_the_fair_run_must_be_the_declared_8_lab_workers():
    d = repeat("matched", 0)
    with pytest.raises(s.Bad, match="workers"):
        s.matched(d, s.load_prior("matched"), "fair")


def test_a_fair_receipt_must_carry_the_declared_400_lab_studies():
    d = repeat("matched", 0)  # 100 lab studies, relabelled as an 8-worker run
    d["arms"]["lab"]["workers"], d["arms"]["lab"]["affinity"] = 8, d["host"]["affinity"]
    with pytest.raises(s.Bad, match="not the declared"):
        s.matched(d, s.load_prior("matched"), "fair")


def test_a_prior_that_is_not_the_pinned_bytes_is_refused(monkeypatch):
    monkeypatch.setitem(s.PRIORS, "matched", (s.PRIORS["matched"][0], "0" * 64))
    with pytest.raises(s.Bad, match="pinned"):
        s.load_prior("matched")


def test_the_committed_result_is_what_the_reducer_makes_of_the_committed_raw_files(tmp_path):
    out = tmp_path / "again.json"
    assert s.main([str(s.RESULTS / "sweep-repeats-raw"), str(out)]) == 0
    assert out.read_text() == (s.RESULTS / "sweep-repeats.json").read_text()
