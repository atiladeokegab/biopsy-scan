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
    TASK[kind](d)["modal_task_id"] = f"ta-{k}"
    d[COMMIT[kind]] = "c0ffee"
    if edit:
        edit(d)
    return d


def raw(tmp_path, kind, docs):
    (tmp_path / kind).mkdir()
    for k, d in enumerate(docs):
        (tmp_path / kind / f"r{k}.json").write_text(json.dumps(d))
    (tmp_path / "commits.json").write_text(json.dumps({kind: "c0ffee"}))
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


def screen_old_p(d):
    d["screen"]["backends"]["cupy"]["p"] = 0.789


def fleet_bits(d):
    d["fleet"]["runs"]["8"]["bit_identical_to_1"] = False


def scaling_x(d):
    d["studies_by_cores"]["64"][5]["x_sha"] = "0" * 64


def scaling_dup(d):
    d["studies_by_cores"]["8"][1] = d["studies_by_cores"]["8"][0]


def nan_wall(d):
    d["wall"]["lab_s_per_study"] = float("nan")


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
    with pytest.raises(s.Bad, match="2 repeats"):
        s.reduce("matched", sorted((tmp_path / "matched").glob("*"))[:2], "c0ffee")


def test_the_fair_run_must_be_the_declared_8_lab_workers():
    d = repeat("matched", 0)
    with pytest.raises(s.Bad, match="workers"):
        s.matched(d, s.load_prior("matched"), "fair")


def test_a_prior_that_is_not_the_pinned_bytes_is_refused(monkeypatch):
    monkeypatch.setitem(s.PRIORS, "matched", (s.PRIORS["matched"][0], "0" * 64))
    with pytest.raises(s.Bad, match="pinned"):
        s.load_prior("matched")
