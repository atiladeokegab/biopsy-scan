"""A-16.1: the rediscovery candidate is a committed, hashed rule that UPMC can express."""
import io
import os
import zipfile
from pathlib import Path

import pytest

from audit.rediscovery import M, RULE, RULE_SHA256, UPMC_MARKERS, UPMC_TYPES
from core.grammar import feature_names, grammar

# Frozen when the rule was committed, before the search or any outcome. Never update it to make this test pass.
COMMITTED_SHA256 = "a70424447055052d19610332a478fb46da68d87a9f2f30b350cc3c14a4bada1e"


def test_rule_is_unchanged_since_commit():
    assert RULE_SHA256 == COMMITTED_SHA256


def test_rule_is_a_column_of_the_upmc_screen():
    assert RULE["expressible"] and not RULE["outcomes_opened"]
    assert RULE["feature"] in feature_names(grammar(UPMC_TYPES))
    assert RULE["feature"] == f"{RULE['focal']}|{RULE['neighbour']}|{RULE['radius_um']}um|{RULE['statistic']}"


def test_marker_count_for_r17():
    assert M == 22 and len(set(UPMC_MARKERS)) == M


DATA = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[1] / "data")) / "upmc_raw_data.zip"


@pytest.mark.skipif(not DATA.exists(), reason="UPMC raw data not fetched")
def test_types_and_markers_match_disk():
    """Reads one region's headers and types only; no label or outcome file."""
    with zipfile.ZipFile(DATA) as z:
        region = next(n for n in z.namelist() if n.endswith(".expression.csv"))
        header = io.TextIOWrapper(z.open(region)).readline().strip().split(",")
        assert header[1:] == UPMC_MARKERS
    assert RULE["focal"] in UPMC_TYPES and RULE["neighbour"] in UPMC_TYPES
