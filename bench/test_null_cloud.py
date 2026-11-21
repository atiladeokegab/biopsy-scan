import json
from pathlib import Path

import numpy as np
import pytest

from bench.null_cloud import OUT, summarise, svg

AUDIT = json.loads((Path(__file__).resolve().parents[1] / "audit" / "results" / "charville.json").read_text())


def test_committed_cloud_matches_audit_and_is_labelled_measured():
    res = json.loads(OUT.read_text())
    assert res["label"] == "MEASURED" and res["p"] == AUDIT["maxT"]["p"]
    assert res["observed_maxT"] == AUDIT["maxT"]["statistic"] and res["n_resamples"] == AUDIT["maxT"]["n_resamples"]
    assert sum(res["histogram"]["counts"]) == res["n_resamples"]
    # the real point sits inside the cloud, below the cutoff: chance explains it
    assert res["null_quantiles"]["0.5"] < res["observed_maxT"] < res["cutoff_alpha_05"]
    text = OUT.with_suffix(".svg").read_text()
    assert text == svg(res) and "MEASURED" in text and "p = 0.13" in text


def test_summarise_refuses_a_null_that_does_not_reproduce_the_audit_p():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="do not match the audit"):
        summarise(rng.normal(2, 0.1, 9999), AUDIT["maxT"]["statistic"], AUDIT, seed=0)
