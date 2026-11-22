import json

from bench.upmc_replication import OUT, build, pooled


def test_build_reproduces_charville_headline_and_matches_committed_result():
    res = build()
    at200 = next(x for x in res["curve_83"]["matched"] if x["patients"] == 200)
    assert round(at200["charville"]["analytic_power"], 3) == 0.801 and at200["charville"]["sim_power"] == 0.27
    assert res["curve_83"]["charville"]["pooled_fwer"]["rejections"] == 28  # 28/700, the fact of record
    assert res["curve_83"]["upmc"]["pooled_fwer"]["nulls"] == 600
    assert res["production"]["charville"]["formula_overstates"] == res["production"]["charville"]["design_points"]
    committed = json.loads(OUT.read_text())
    assert res["replicated"] is committed["replicated"] is False
    assert res["answer"] == committed["answer"]


def test_pooled_fwer_interval_brackets_rate():
    p = pooled([{"patients": 1, "k": 3, "n": 100}, {"patients": 2, "k": 2, "n": 100}], "k", "n")
    assert (p["rejections"], p["nulls"]) == (5, 200) and p["lo"] < 0.025 < p["hi"]
