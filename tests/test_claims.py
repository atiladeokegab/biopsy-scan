"""The claim ledger (results/claims.json): every headline number still matches its source file, and no doc carries a stale one."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_claims", ROOT / "scripts" / "check_claims.py")
check_claims = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_claims)


def test_every_claim_matches_its_source_file():
    assert check_claims.drift() == []


def test_no_doc_or_demo_page_carries_a_stale_value():
    assert check_claims.stale() == []


@pytest.fixture
def copy(tmp_path):
    """The ledger, its sources and the scanned docs, copied so a test can plant a bad value."""
    ledger = json.loads((ROOT / check_claims.LEDGER).read_text())
    files = {check_claims.LEDGER, *(c["source"] for c in ledger["claims"])}
    files |= {q["file"] for c in ledger["claims"] for q in c["quotes"]}
    files |= {p.relative_to(ROOT).as_posix() for p in check_claims.doc_files()}
    for f in files:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, tmp_path / f)
    assert check_claims.main(["check_claims", str(tmp_path)]) == 0
    return tmp_path


def test_a_planted_stale_value_fails(copy):
    readme = copy / "README.md"
    readme.write_text(readme.read_text() + "\nKernel: NumPy 2.23 s vs GPU 0.70 s.\n")
    hits = check_claims.stale(copy)
    assert [h[0] for h in hits] == [f"README.md:{len(readme.read_text().splitlines())}"]
    assert check_claims.main(["check_claims", str(copy)]) == 1


def test_a_source_that_moves_fails(copy):
    src = copy / "bench/results/modal-a100-20261121T183709Z.json"
    data = json.loads(src.read_text())
    data["kernel"]["rows"]["numpy_reference"]["median_s"] = 2.2349
    src.write_text(json.dumps(data))
    bad = check_claims.drift(copy)
    assert bad[0] == ("kernel-3.19x", "numpy", "2.24", "2.23")
    assert [b[1] for b in bad[1:]] == ["README.md"]  # the README quote no longer matches the source
    assert check_claims.main(["check_claims", str(copy)]) == 1


def test_a_relabelled_source_fails(copy):
    src = copy / "stats/results/hard-negative-f458ad1.json"
    data = json.loads(src.read_text())
    data["label"] = "SIMULATED"
    src.write_text(json.dumps(data))
    assert check_claims.drift(copy)[0] == ("fwer-0.060", "label", "SYNTHETIC", "SIMULATED")


MUTANTS = [  # (kind, file, published text, mutant): each must make the check fail
    ("value", "README.md", "`68.3%`", "`98.3%`"),
    ("value", "README.md", "NumPy `2.24 s`", "NumPy `2.34 s`"),
    ("value", "README.md", "843.7→128.2 s", "843.7→118.2 s"),
    ("value", "docs/JUDGING.md", "0.060 (60 of 1,000)", "0.050 (50 of 1,000)"),
    ("value", "docs/JUDGING.md", "gives 27% (95% CI 22 to 32%)", "gives 37% (95% CI 22 to 32%)"),
    ("value", "demo/data/claims.json", '"sim": 0.27,', '"sim": 0.37,'),
    ("value", "demo/data/claims.json", '"p": 0.0981,', '"p": 0.0481,'),
    ("label", "README.md", "(**ESTIMATE** (SIMULATED))", "(**MEASURED** (SIMULATED))"),
    ("label", "README.md", "unadjusted (**SYNTHETIC**)", "unadjusted (**MEASURED**)"),
    ("label", "docs/JUDGING.md", "| MEASURED, exploratory, negative |", "| ESTIMATE, exploratory, negative |"),
    ("label", "demo/data/claims.json", '"formula_label": "ANALYTIC"', '"formula_label": "MEASURED"'),
    ("prose", "README.md", "interpolated between", "measured between"),
    ("prose", "README.md", "exact parity", "close parity"),
    ("prose", "README.md", "inference outside timing", "inference inside timing"),
    ("prose", "docs/JUDGING.md", "consistent with chance", "a real effect"),
]


@pytest.mark.parametrize("kind,file,text,mutant", MUTANTS, ids=[f"{m[0]}-{i}" for i, m in enumerate(MUTANTS)])
def test_a_mutated_published_claim_fails(copy, kind, file, text, mutant):
    doc = copy / file
    body = doc.read_text(encoding="utf-8")
    assert body.count(text) == 1, f"{text!r} must appear once in {file} for the mutant to mean anything"
    doc.write_text(body.replace(text, mutant), encoding="utf-8")
    assert any(b[1].startswith(file) for b in check_claims.drift(copy))
    assert check_claims.main(["check_claims", str(copy)]) == 1


def test_list_selector_needs_exactly_one_match():
    with pytest.raises(KeyError):
        check_claims.lookup({"points": [{"n": 1}, {"n": 1}]}, "points[n=1].x")
    assert check_claims.lookup({"points": [{"n": 1, "x": 5}, {"n": 2, "x": 6}]}, "points[n=2].x") == 6
