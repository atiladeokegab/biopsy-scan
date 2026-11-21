import json
import math
from pathlib import Path

import pytest

from bench.vs_schoenfeld import compare, schoenfeld_power, svg

META = {"source_events": 38, "source_n": 159, "inference": "model-based Monte Carlo"}


def row(p, power, lo, hi, **kw):
    return {"cohort": "charville", "patients": p, "hr_per_source_sd": 2.0, "alpha": 0.05, "regions_per_patient": 2,
            "fov_um": 400.0, "eps": 0.1, "backend": "cpu", "power": power, "power_lo": lo, "power_hi": hi,
            "power_n": 200, "identification_rate": 0.0, "alternative_mean_events": 19.0, "fwer": 0.05, "fwer_lo": 0.02, "fwer_hi": 0.09, **kw}


def test_formula_hits_the_textbook_80_percent_point():
    # sqrt(D) * log(HR) = z_.975 + z_.80 gives 80% power at m=1
    d = ((1.959964 + 0.841621) / math.log(2.0)) ** 2
    assert schoenfeld_power(d, 1.0, 2.0, 0.05, 1) == pytest.approx(0.80, abs=1e-4)
    assert schoenfeld_power(d, 1.0, 2.0, 0.05, 630) < 0.80  # Bonferroni costs power
    assert schoenfeld_power(d, 1.0, 0.5, 0.05, 1) == pytest.approx(0.80, abs=1e-4)  # two-sided


def test_verdict_follows_the_data_either_way():
    hi = compare([row(80, 0.0, 0.0, 0.01), row(300, 0.1, 0.05, 0.15)], META, 630, cohort="charville")
    assert hi["verdict"].startswith("The formula overstates power at 2 of 2")
    lo = compare([row(80, 1.0, 0.99, 1.0), row(300, 1.0, 0.99, 1.0)], META, 630, cohort="charville")
    assert lo["verdict"].startswith("The formula understates power at 2 of 2")
    mixed = compare([row(80, 0.0, 0.0, 0.01), row(300, 0.97, 0.9, 1.0)], META, 630, cohort="charville")
    assert mixed["verdict"].endswith("at 1 of 2 patient counts (outside the simulation's 95% interval); inside it at the other 1.")
    s = svg(hi)
    assert "MEASURED (sim)" in s and "ANALYTIC (formula)" in s and s.count("<polyline") == 2


def test_one_curve_per_design():
    with pytest.raises(ValueError, match="mix eps"):
        compare([row(80, 0.1, 0, 0.2), row(300, 0.1, 0, 0.2, eps=0.2)], META, 630, cohort="charville")
    with pytest.raises(ValueError, match="repeat"):
        compare([row(80, 0.1, 0, 0.2), row(80, 0.1, 0, 0.2)], META, 630, cohort="charville")


def test_control_design_is_a_third_line_at_the_same_points():
    ctl = [row(80, 0.3, 0.2, 0.4, fov_um=None, eps=0.0), row(300, 0.99, 0.95, 1.0, fov_um=None, eps=0.0)]
    res = compare([row(80, 0.1, 0.0, 0.2), row(300, 0.9, 0.8, 0.95)], META, 630, cohort="charville", control=ctl)
    assert res["control"]["design"]["fov_um"] == "full core" and len(res["control"]["points"]) == 2
    assert svg(res).count("<polyline") == 3 and "control" in svg(res)
    part = compare([row(80, 0.1, 0.0, 0.2), row(300, 0.9, 0.8, 0.95)], META, 630, cohort="charville", control=ctl[:1])
    assert [p["patients"] for p in part["control"]["points"]] == [80]  # a stopped run keeps its finished points
    assert "1 of 2 points" in svg(part) and "of 2 points" not in svg(res)
    with pytest.raises(ValueError, match="main design's patient counts"):
        compare([row(80, 0.1, 0.0, 0.2)], META, 630, cohort="charville", control=ctl)


def test_status_labels_json_and_chart():
    res = compare([row(80, 0.1, 0.0, 0.2), row(300, 0.9, 0.8, 0.95)], META, 630, cohort="charville", status="ESTIMATE")
    assert res["status"] == "ESTIMATE" and ">ESTIMATE · Textbook" in svg(res)
    assert "status" not in compare([row(80, 0.1, 0.0, 0.2)], META, 630, cohort="charville")


RESULTS = sorted((Path(__file__).parent / "results").glob("vs-schoenfeld*.json"))


@pytest.mark.parametrize("path", RESULTS, ids=[p.name for p in RESULTS])
def test_committed_result_gives_both_powers_per_design_point(path):
    # PRD A-7.1: per design point, Schoenfeld power (ANALYTIC) and simulated power with its interval
    res = json.loads(path.read_text())
    assert res["schema"] == "preflight-vs-schoenfeld/1" and res["cohort"] == "charville"
    assert res["analytic"]["label"] == "ANALYTIC (formula)" and res["simulated"]["label"] == "MEASURED (sim)"
    assert res["simulated"]["method"].startswith("model-based Monte Carlo")
    assert len(res["points"]) >= 2
    for p in res["points"]:
        assert 0 <= p["sim_lo"] <= p["sim_power"] <= p["sim_hi"] <= 1
        assert p["analytic_power"] == pytest.approx(schoenfeld_power(p["patients"], res["event_fraction"], res["hr_per_sd"],
                                                                     res["alpha"], res["features_m"]))
    svg_text = path.with_suffix(".svg").read_text()
    assert "MEASURED (sim)" in svg_text and "ANALYTIC (formula)" in svg_text
    if path.name == "vs-schoenfeld-300.json":
        assert res["status"] == "ESTIMATE"
        assert ">ESTIMATE · Textbook" in svg_text
