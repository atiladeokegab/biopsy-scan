"""A-16.2: the committed results/upmc-rediscovery.json (#109) meets the contract the scorer's fixture tests pin.

Reads the file as it is; reruns nothing and reads no label or outcome file.
"""
import copy
import json

import pytest

from audit import rediscovery as rs

COMMITTED = rs.ROOT / "results" / "upmc-rediscovery.json"


def _strict(text):
    def bad(c):
        raise ValueError(f"non-strict JSON constant {c}")
    return json.loads(text, parse_constant=bad)


def check(doc):
    """Raises AssertionError on any breach of the A-16.2 result contract."""
    assert doc["cohort"] == "upmc" and doc["label"] == rs.LABEL
    assert doc["rule_sha256"] == doc["locked"]["rule_sha256"] == rs.RULE_SHA256 == rs.PREREG_SHA256
    assert doc["manifest_sha256"] == rs.APPROVED_SPLIT["manifest_sha256"]
    marker = rs.opening_record(rs.RULE_SHA256, rs.APPROVED_SPLIT["manifest_sha256"])
    assert doc["opening_record"] == marker.name and marker.exists()
    opened = _strict(marker.read_text())
    assert opened["code_sha"] == doc["code_sha"] and opened["design"] == rs.APPROVED_SPLIT

    d = doc["discovery"]
    assert d["status"] == "MEASURED" and 1 <= d["rank"] <= d["of"] and isinstance(d["direction_ok"], bool)
    assert d["label"].startswith("MEASURED on discovery patients only") and rs.LABEL in d["label"]

    lk = doc["locked"]
    assert lk["feature"] == rs.RULE["feature"] and lk["label"] == rs.LABEL and lk["alpha"] == rs.ALPHA
    assert (lk["n_patients"], lk["n_events"]) == (rs.APPROVED_SPLIT["locked_patients"], rs.APPROVED_SPLIT["locked_events"])
    pw = lk["power"]
    assert pw["label"].startswith("SIMULATED") and "no outcomes read" in pw["label"]
    assert (pw["effect_hr_per_sd"], pw["min_power"]) == (rs.EFFECT_HR_PER_SD, rs.MIN_POWER)
    assert lk["status"] in ("MEASURED", "UNMEASURED")
    if pw["value"] < rs.MIN_POWER:  # underpowered: never MEASURED, whatever the p
        assert lk["status"] == "UNMEASURED" and lk["reason"].startswith("underpowered")
    assert lk["locked_test_passes"] == (lk["p_one_sided"] < rs.ALPHA)

    assert doc["charville_top_hit"] == {**rs.upmc_marker_check(rs.CHARVILLE_TOP_HIT), **rs.CHARVILLE_TOP_HIT_SOURCE}
    assert doc["charville_top_hit"]["status"] == "NOT TESTABLE"


def test_committed_result_meets_the_contract():
    check(_strict(COMMITTED.read_text()))


def _set(path, value):
    def edit(doc):
        *head, last = path
        for k in head:
            doc = doc[k]
        doc[last] = value
    return edit


@pytest.mark.parametrize("edit", [
    _set(["rule_sha256"], "0" * 64),
    _set(["locked", "rule_sha256"], "0" * 64),
    _set(["manifest_sha256"], "0" * 64),
    _set(["opening_record"], "upmc-locked-opened.other.json"),
    _set(["label"], "confirmatory"),
    _set(["discovery", "status"], "UNMEASURED"),
    _set(["locked", "status"], "MEASURED"),
    _set(["locked", "feature"], rs.CHARVILLE_TOP_HIT),
    _set(["locked", "n_events"], 5),
    _set(["locked", "locked_test_passes"], True),
    _set(["locked", "power", "label"], "achieved power"),
    _set(["charville_top_hit", "status"], "FAILED REPLICATION"),
], ids=lambda e: "/".join(map(str, e.__closure__[0].cell_contents)))
def test_a_broken_copy_fails_the_contract(edit):
    doc = copy.deepcopy(_strict(COMMITTED.read_text()))
    edit(doc)
    with pytest.raises(AssertionError):
        check(doc)


def test_a_non_strict_number_is_refused():
    with pytest.raises(ValueError):
        _strict(COMMITTED.read_text().replace('"p_one_sided": 0.19255938147949075', '"p_one_sided": NaN'))
