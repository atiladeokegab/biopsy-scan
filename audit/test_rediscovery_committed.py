"""A-16.2: the committed results/upmc-rediscovery.json (#109) meets the contract the scorer's fixture tests pin.

Reads the result and its opening receipt as committed; reruns nothing and reads no label, outcome or manifest file.
"""
import copy
import json
import math

import pytest
from scipy.stats import norm

from audit import rediscovery as rs

COMMITTED = rs.ROOT / "results" / "upmc-rediscovery.json"
RECEIPT = rs.opening_record(rs.RULE_SHA256, rs.APPROVED_SPLIT["manifest_sha256"])
# what score_feature returns once the Cox fit ran: present when MEASURED or underpowered, absent otherwise
ESTIMATES = ("hr_per_sd", "hr_per_sd_ci95", "log_hr_per_sd", "se", "wald_z", "p_one_sided", "c_index",
             "locked_test_passes")
Z95 = 1.959964  # rediscovery.score_feature's CI multiplier


def _strict(text):
    def bad(c):
        raise ValueError(f"non-strict JSON constant {c}")
    return json.loads(text, parse_constant=bad)


def _finite(*vals):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals)


def check(doc, opened):
    """Raises AssertionError on any breach of the A-16.2 result contract; `opened` is the opening receipt."""
    assert doc["cohort"] == "upmc" and doc["label"] == rs.LABEL
    assert doc["rule_sha256"] == doc["locked"]["rule_sha256"] == rs.RULE_SHA256 == rs.PREREG_SHA256
    assert doc["manifest_sha256"] == rs.APPROVED_SPLIT["manifest_sha256"]
    assert doc["opening_record"] == RECEIPT.name

    # the receipt froze the discovery rank before the locked set was opened: the result must repeat it exactly
    assert opened["rule_sha256"] == rs.RULE_SHA256
    assert opened["manifest_sha256"] == opened["design"]["manifest_sha256"] == rs.APPROVED_SPLIT["manifest_sha256"]
    assert opened["design"] == rs.APPROVED_SPLIT and opened["code_sha"] == doc["code_sha"]
    assert doc["discovery"] == opened["discovery"]

    d = doc["discovery"]
    assert d["status"] == "MEASURED" and 1 <= d["rank"] <= d["of"] <= d["of_all"]
    assert _finite(d["z"]) and d["direction_ok"] is (d["z"] > 0)  # the rule's direction: higher is worse
    assert d["label"].startswith("MEASURED on discovery patients only") and rs.LABEL in d["label"]

    lk = doc["locked"]
    assert lk["feature"] == rs.RULE["feature"] and lk["label"] == rs.LABEL
    assert (lk["n_patients"], lk["n_events"]) == (rs.APPROVED_SPLIT["locked_patients"], rs.APPROVED_SPLIT["locked_events"])
    pw = lk["power"]
    assert pw["label"].startswith("SIMULATED") and "no outcomes read" in pw["label"]
    assert (pw["effect_hr_per_sd"], pw["min_power"], pw["alpha"]) == (rs.EFFECT_HR_PER_SD, rs.MIN_POWER, rs.ALPHA)
    assert _finite(pw["value"]) and 0 <= pw["value"] <= 1 and pw["sims"] > 0
    assert lk["status"] in ("MEASURED", "UNMEASURED")
    if pw["value"] < rs.MIN_POWER:  # underpowered: never MEASURED, whatever the p
        assert lk["status"] == "UNMEASURED"
    fitted = lk["status"] == "MEASURED" or lk.get("reason", "").startswith("underpowered")
    if not fitted:  # not estimable, fit failed, overflow: a reason and no numbers
        assert lk["status"] == "UNMEASURED" and lk.get("reason") and not set(ESTIMATES) & set(lk)
    else:
        assert set(ESTIMATES) <= set(lk)
        hr, ci, beta, se, z, p = (lk[k] for k in ("hr_per_sd", "hr_per_sd_ci95", "log_hr_per_sd", "se", "wald_z",
                                                   "p_one_sided"))
        assert len(ci) == 2 and _finite(hr, *ci, beta, se, z, p) and se > 0
        assert 0 < ci[0] <= hr <= ci[1] and 0 <= p <= 1 and lk["alpha"] == rs.ALPHA
        assert math.isclose(hr, math.exp(beta), rel_tol=1e-9)
        assert math.isclose(ci[0], math.exp(beta - Z95 * se), rel_tol=1e-9)
        assert math.isclose(ci[1], math.exp(beta + Z95 * se), rel_tol=1e-9)
        assert math.isclose(z, beta / se, rel_tol=1e-9) and math.isclose(p, norm.sf(z), rel_tol=1e-9)
        assert lk["locked_test_passes"] is (p < rs.ALPHA)
        c = lk["c_index"]
        assert (c is None and lk["c_index_reason"]) or (_finite(c) and 0 <= c <= 1)

    assert doc["charville_top_hit"] == {**rs.upmc_marker_check(rs.CHARVILLE_TOP_HIT), **rs.CHARVILLE_TOP_HIT_SOURCE}
    assert doc["charville_top_hit"]["status"] == "NOT TESTABLE"


def _load():
    return _strict(COMMITTED.read_text()), _strict(RECEIPT.read_text())


def test_committed_result_meets_the_contract():
    check(*_load())


def test_a_not_estimable_locked_result_with_a_reason_and_no_numbers_is_legitimate():
    doc, opened = _load()
    doc["locked"] = {k: v for k, v in doc["locked"].items() if k not in ESTIMATES + ("feature_sd", "alpha")}
    doc["locked"]["reason"] = "not estimable: 1 events, feature SD 0.03 on the locked patients"
    check(doc, opened)


def _set(path, value):
    def edit(doc):
        *head, last = path
        for k in head:
            doc = doc[k]
        doc[last] = value
    edit.id = "set:" + "/".join(map(str, path)) + f"={value!r}"[:30]
    return edit


def _del(path):
    def edit(doc):
        *head, last = path
        for k in head:
            doc = doc[k]
        del doc[last]
    edit.id = "del:" + "/".join(map(str, path))
    return edit


def _scale(path, f):
    def edit(doc):
        *head, last = path
        for k in head:
            doc = doc[k]
        doc[last] = f(doc[last])
    edit.id = "flip:" + "/".join(map(str, path))
    return edit


RESULT_MUTANTS = [
    # identity, labels, design
    _set(["rule_sha256"], "0" * 64), _set(["locked", "rule_sha256"], "0" * 64), _set(["manifest_sha256"], "0" * 64),
    _set(["opening_record"], "upmc-locked-opened.other.json"), _set(["label"], "confirmatory"),
    _set(["locked", "feature"], rs.CHARVILLE_TOP_HIT), _set(["locked", "n_events"], 5),
    _set(["locked", "power", "label"], "achieved power"), _set(["charville_top_hit", "status"], "FAILED REPLICATION"),
    # deletions of the principal estimates
    *(_del(["locked", k]) for k in ESTIMATES), _del(["locked", "reason"]),
    # out of range
    _set(["locked", "p_one_sided"], 2.5), _set(["locked", "p_one_sided"], -0.1), _set(["locked", "power", "value"], -1),
    _set(["locked", "power", "value"], 1.5), _set(["locked", "c_index"], 1.2), _set(["locked", "hr_per_sd"], -1.8),
    _scale(["locked", "hr_per_sd_ci95"], lambda ci: ci[::-1]), _set(["locked", "se"], 0.0),
    _set(["locked", "hr_per_sd"], float("inf")),
    # flips and changed values
    _set(["locked", "status"], "MEASURED"), _set(["locked", "locked_test_passes"], True),
    _scale(["locked", "wald_z"], lambda z: -z), _scale(["locked", "p_one_sided"], lambda p: 1 - p),
    _scale(["locked", "hr_per_sd"], lambda hr: 1 / hr),
    _set(["discovery", "status"], "UNMEASURED"), _scale(["discovery", "rank"], lambda r: 1 if r != 1 else 2),
    _scale(["discovery", "direction_ok"], lambda ok: not ok), _scale(["discovery", "z"], lambda z: -z),
]


@pytest.mark.parametrize("edit", RESULT_MUTANTS, ids=lambda e: e.id)
def test_a_broken_result_fails_the_contract(edit):
    doc, opened = _load()
    edit(doc)
    with pytest.raises(AssertionError):
        check(doc, opened)


RECEIPT_MUTANTS = [
    _set(["rule_sha256"], "0" * 64), _set(["manifest_sha256"], "0" * 64),
    _set(["design", "locked_events"], 5), _set(["code_sha"], "0" * 40),
    _scale(["discovery", "rank"], lambda r: 1 if r != 1 else 2),
    _scale(["discovery", "direction_ok"], lambda ok: not ok), _scale(["discovery", "z"], lambda z: -z),
]


@pytest.mark.parametrize("edit", RECEIPT_MUTANTS, ids=lambda e: e.id)
def test_a_result_that_disagrees_with_its_opening_receipt_fails(edit):
    doc, opened = _load()
    edit(opened)
    with pytest.raises(AssertionError):
        check(doc, opened)


def test_same_rank_and_direction_changed_in_both_still_fails_on_direction_consistency():
    doc, opened = _load()
    for d in (doc["discovery"], opened["discovery"]):
        d["direction_ok"] = not d["direction_ok"]
    with pytest.raises(AssertionError):
        check(doc, opened)


def test_a_non_strict_number_is_refused():
    doc, _ = _load()
    doc["locked"]["p_one_sided"] = float("nan")
    with pytest.raises(ValueError):
        _strict(json.dumps(doc))
