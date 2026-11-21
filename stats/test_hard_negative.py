"""A-15.6: family-wise false positives at alpha on a clinically confounded null (SYNTHETIC).

200 null datasets, B = 19 resamples each. The plus-one rule gives p-values on a
1/20 grid (Monte Carlo resolution); it does not make the fitted parametric
bootstrap exact, since the observed and fitted-bootstrap statistics are not
exchangeable. The recorded larger run (1,000 datasets, B = 99) is
stats/results/hard-negative.json.
"""

from stats.hard_negative import run

N_DATASETS = 200


def test_bootstrap_holds_alpha_and_methods_that_ignore_c_are_seen():
    result = run(N_DATASETS, B=19)
    rates = result["rates"]
    print({k: (v["rate"], v["clopper_pearson_95"]) for k, v in rates.items()})
    lo, hi = rates["bootstrap"]["clopper_pearson_95"]
    assert lo <= 0.05 <= hi, rates["bootstrap"]
    lo, _ = rates["unadjusted"]["clopper_pearson_95"]
    assert lo > 0.05, rates["unadjusted"]  # red on the screen that ignores C
    assert {"raw_residual_shuffle", "unadjusted"} <= set(rates) and result["label"] == "SYNTHETIC"
