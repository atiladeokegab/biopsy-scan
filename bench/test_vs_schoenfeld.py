import math

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
    s = svg(hi)
    assert "MEASURED (sim)" in s and "ANALYTIC (formula)" in s and s.count("<polyline") == 2


def test_one_curve_per_design():
    with pytest.raises(ValueError, match="mix eps"):
        compare([row(80, 0.1, 0, 0.2), row(300, 0.1, 0, 0.2, eps=0.2)], META, 630, cohort="charville")
    with pytest.raises(ValueError, match="repeat"):
        compare([row(80, 0.1, 0, 0.2), row(80, 0.1, 0, 0.2)], META, 630, cohort="charville")
