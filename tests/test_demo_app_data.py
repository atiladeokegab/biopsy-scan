"""demo/app: data.json is a fresh build of the result files, and the page code hardcodes none of its numbers or names."""
import importlib.util
import json
import re

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_data", ROOT / "demo" / "app" / "build_data.py")
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)
DATA = json.loads((ROOT / "demo/app/data.json").read_text())
UI = {p: (ROOT / "demo/app" / p).read_text() for p in ("index.html", "app.js")}


def test_data_json_matches_the_sources():
    assert json.loads(json.dumps(bd.build())) == DATA


def test_race_numbers_follow_the_founder_ruling():
    r, cur, ours = DATA["race"], DATA["current"], DATA["ours"]
    assert DATA["studies"] == 1000 and cur["workers"] == 8 and ours["gpus"] == 1
    assert ours["source"] == "bench/results/gpu-fused-h100-20261122T141815Z.json" and ours["source"] in DATA["sources"]
    assert abs(ours["s_per_study"] - 0.0020690) < 1e-6 and 2.0 < ours["total_s"] < 2.2  # about 2.1 s for 1,000 on one H100
    assert abs(DATA["studies"] * ours["s_per_study"] * r["race_s_per_real_s"] - r["ours_finish_s"]) < 1e-9
    assert r["current_studies_at_end"] == 1  # about 1 of 1,000 by 10 s
    assert 6 * 3600 + 10 * 60 < cur["total_s"] < 6 * 3600 + 15 * 60  # about 6 h 12 m (same-host 22.35 s/study)
    assert set(DATA["kept"]) <= set(DATA["markers"])


def forbidden():
    c = DATA["cohort"]
    nums = {c["patients"], c["events"], c["regions"], c["cores_shown"], DATA["studies"], DATA["race"]["current_studies_at_end"],
            round(DATA["ratio"])}
    words = set(DATA["markers"]) | {p["gene"] for p in DATA["ihc"]} | {c["name"]}
    pats = [rf"(?<![\w.]){n:,}(?![\w.])".replace(",", ",?") for n in nums if n >= 10]  # 1 and the like are not tell-tales
    pats += [rf"(?<![\w.]){re.escape(f'{v:.4g}')}" for v in (DATA["current"]["s_per_study"], DATA["ours"]["s_per_study"])]
    pats += [rf"(?i)\b{re.escape(w)}\b" for w in words]
    return pats


def test_ui_hardcodes_no_cohort_numbers_or_marker_names():
    hits = [(p, pat, m.group()) for p, text in UI.items() for pat in forbidden() for m in [re.search(pat, text)] if m]
    assert not hits, hits


def test_the_grep_catches_a_planted_number_and_marker():
    planted = "<b>159 patients</b> <span>1,000</span> PanCK"
    assert sum(bool(re.search(p, planted)) for p in forbidden()) >= 3


def test_current_receipt_gate_rejects_a_doctored_receipt():
    good = json.loads((ROOT / bd.CURRENT).read_text())
    bd.check_current(good)
    for key, bad in (("status", "FAILED"), ("errors", ["boom"])):
        with pytest.raises(AssertionError):
            bd.check_current({**good, key: bad})
    with pytest.raises(AssertionError):
        bd.check_current({**good, "checks": {**good["checks"], "same_cells_both_arms": False}})
