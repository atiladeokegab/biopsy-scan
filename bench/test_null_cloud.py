import json
import numpy as np
import pytest

from bench.null_cloud import AUDIT as AUDIT_PATH, OUT, summarise, svg

AUDIT = json.loads(AUDIT_PATH.read_text())


def test_committed_cloud_matches_v2_joint_screen_and_is_labelled_measured():
    res = json.loads(OUT.read_text())
    assert res["label"] == "MEASURED" and res["p"] == AUDIT["maxT"]["p"]
    assert res["observed_maxT"] == AUDIT["maxT"]["statistic"] and res["n_resamples"] == AUDIT["maxT"]["n_resamples"]
    assert res["features_tested"] == AUDIT["n_features_tested"] == 3780
    assert sum(res["histogram"]["counts"]) == res["n_resamples"]
    # the real point sits inside the cloud, below the cutoff: not significant after correction
    assert res["null_quantiles"]["0.5"] < res["observed_maxT"] < res["cutoff_alpha_05"]
    text = OUT.with_suffix(".svg").read_text()
    assert text == svg(res) and "MEASURED" in text and f"p = {AUDIT['maxT']['p']:.3f}" in text
    assert "not significant after correction" in text and "chance explains" not in text.lower() and "no effect" not in text.lower()


def _null(at_least, n=AUDIT["maxT"]["n_resamples"]):
    # a synthetic null with exactly at_least draws at or above the observed statistic
    obs = AUDIT["maxT"]["statistic"]
    return np.r_[np.linspace(obs + 0.01, obs + 1, at_least), np.linspace(1, obs - 0.01, n - at_least)]


def test_summarise_guards_draw_count_and_cutoff():
    at_least = round(AUDIT["maxT"]["p"] * (AUDIT["maxT"]["n_resamples"] + 1)) - 1
    obs = AUDIT["maxT"]["statistic"]
    with pytest.raises(ValueError, match="finite null draws"):
        summarise(_null(at_least)[:999], obs, AUDIT, seed=0)
    with pytest.raises(ValueError, match="finite null draws"):
        summarise(np.r_[_null(at_least)[:-1], np.nan], obs, AUDIT, seed=0)
    with pytest.raises(ValueError, match="cutoff"):  # p and statistic match, the cutoff does not
        summarise(_null(at_least), obs, AUDIT, seed=0)
    from audit.runner import conditional_permutation_cutoff
    ok = {**AUDIT, "maxT": {**AUDIT["maxT"], "conditional_cutoff_alpha_05":
                            conditional_permutation_cutoff(_null(at_least)[:, None])}}
    assert summarise(_null(at_least), obs, ok, seed=0)["p"] == AUDIT["maxT"]["p"]


def test_summarise_refuses_a_null_that_does_not_reproduce_the_audit_p():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="do not match the audit"):
        summarise(rng.normal(2, 0.1, 9999), AUDIT["maxT"]["statistic"], AUDIT, seed=0)
