"""The claim ledger (results/claims.json): every headline number still matches its source file, and no doc carries a stale one."""
import importlib.util
import json
import re
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
    assert [b[1] for b in bad[1:]] == ["RESULTS.md"]  # the README quote no longer matches the source
    assert check_claims.main(["check_claims", str(copy)]) == 1


def test_a_relabelled_source_fails(copy):
    src = copy / "stats/results/hard-negative-f458ad1.json"
    data = json.loads(src.read_text())
    data["label"] = "SIMULATED"
    src.write_text(json.dumps(data))
    assert check_claims.drift(copy)[0] == ("fwer-0.060", "label", "SYNTHETIC", "SIMULATED")


def test_a_derived_range_follows_its_source(copy):
    """The Numba range is derived (per-mode median of numpy/numba wall); slowing one mode's Numba moves it."""
    src = copy / "bench/results/modal-quiet-cpu-20261121T180327Z.json"
    data = json.loads(src.read_text())
    for run in data["runs"]:
        if run["mode"].endswith("workqueue"):
            for s in run["studies"]:
                s["numba"]["wall_s"] *= 2
    src.write_text(json.dumps(data))
    bad = check_claims.drift(copy)
    assert ("numba-quiet-2.5-4.1x", "lo", "2.5", "1.2") in bad
    assert any(b[:2] == ("numba-quiet-2.5-4.1x", "RESULTS.md") for b in bad)


MUTANTS = [  # (kind, file, published text, mutant): each must make the check fail
    ("value", "README.md", "**534× faster**", "**634× faster**"),
    ("prose", "README.md", "the timing windows differ", "the timing windows match"),
    ("value", "README.md", "**21× faster**", "**31× faster**"),
    ("prose", "README.md", "**MEASURED**, cross-run)", "**MEASURED**, same run)"),
    ("value", "README.md", "| 22.35 s |", "| 12.35 s |"),
    ("value", "README.md", "| 0.881 s |", "| 0.781 s |"),
    ("value", "README.md", "312× on the same 48 studies", "412× on the same 48 studies"),
    ("prose", "README.md", "; different statistics;", "; same statistics;"),
    ("value", "README.md", "| **27.0%** (95% CI 22.1–32.4%) |", "| **37.0%** (95% CI 22.1–32.4%) |"),
    ("value", "README.md", "| 80.1% |", "| 90.1% |"),
    ("label", "README.md", "Schoenfeld formula (**ANALYTIC**)", "Schoenfeld formula (**MEASURED**)"),
    ("value", "README.md", "8 CPU workers per side", "16 CPU workers per side"),
    ("value", "RESULTS.md", "`68.3%`", "`98.3%`"),
    ("value", "RESULTS.md", "NumPy `2.24 s`", "NumPy `2.34 s`"),
    ("value", "RESULTS.md", "843.7→128.2 s", "843.7→118.2 s"),
    ("value", "docs/JUDGING.md", "0.060 (60 of 1,000)", "0.050 (50 of 1,000)"),
    ("value", "docs/JUDGING.md", "gives 27% (95% CI 22 to 32%)", "gives 37% (95% CI 22 to 32%)"),
    ("value", "demo/data/claims.json", '"sim": 0.27,', '"sim": 0.37,'),
    ("value", "demo/data/claims.json", '"p": 0.0981,', '"p": 0.0481,'),
    ("label", "RESULTS.md", "(**ESTIMATE** (SIMULATED))", "(**MEASURED** (SIMULATED))"),
    ("label", "RESULTS.md", "unadjusted (**SYNTHETIC**)", "unadjusted (**MEASURED**)"),
    ("label", "docs/JUDGING.md", "| MEASURED, exploratory, negative |", "| ESTIMATE, exploratory, negative |"),
    ("label", "demo/data/claims.json", '"formula_label": "ANALYTIC"', '"formula_label": "MEASURED"'),
    ("value", "RESULTS.md", "Discovery rank `366`", "Discovery rank `36`"),
    ("value", "docs/JUDGING.md", "one-sided p 0.193", "one-sided p 0.019"),
    ("label", "RESULTS.md", "**UNMEASURED** (underpowered", "**MEASURED** (underpowered"),
    ("label", "docs/SUBMISSION.md", "which is UNMEASURED (underpowered)", "which is MEASURED (underpowered)"),
    ("value", "RESULTS.md", "Numba `2.5–4.1×`", "Numba `2.5–5.1×`"),
    ("label", "RESULTS.md", "by threading mode (**MEASURED**)", "by threading mode (**ESTIMATE**)"),
    ("value", "demo/data/baseline.json", '"seconds": 1761.88,', '"seconds": 1661.88,'),
    ("value", "RESULTS.md", "`19.5×` (median", "`29.5×` (median"),
    ("value", "RESULTS.md", "calls agree `2,000/2,000`", "calls agree `1,999/2,000`"),
    ("prose", "RESULTS.md", "the calibrated cutoff and calls are outside the timing, as are load", "the calibrated cutoff and calls are inside the timing, as are load"),
    ("prose", "docs/SUBMISSION.md", "the cutoff and calls are outside the timing; #120", "the cutoff and calls are inside the timing; #120"),
    ("value", "demo/index.html", "<b>21×</b> less compute wall", "<b>31×</b> less compute wall"),
    ("value", "README.md", "**5.5× faster**", "**6.5× faster**"),
    ("value", "README.md", "decisions agree on 362 of 400", "decisions agree on 352 of 400"),
    ("value", "README.md", "**19.5× faster**", "**29.5× faster**"),
    ("value", "demo/data/baseline.json", '"seconds": 377.67,', '"seconds": 37.67,'),
    ("label", "demo/data/baseline.json", '"label": "MEASURED",\n      "seconds": 375.26,', '"label": "ESTIMATE",\n      "seconds": 375.26,'),
    ("value", "RESULTS.md", "`7.0×` per evaluated study", "`7.9×` per evaluated study"),
    ("value", "RESULTS.md", "`4.70×`", "`7.04×`"),
    ("value", "RESULTS.md", "agree to `2.3e-15` relative", "agree to `2.3e-12` relative"),
    ("value", "RESULTS.md", "`0.126 s` per study, a tie", "`0.026 s` per study, a tie"),
    ("value", "RESULTS.md", "vs `20.3%` (`203/1,000`)", "vs `30.3%` (`203/1,000`)"),
    ("value", "RESULTS.md", "paired gain `+0.2` pp", "paired gain `+2.2` pp"),
    ("label", "RESULTS.md", "each study's maxT (**MEASURED**)", "each study's maxT (**ESTIMATE**)"),
    ("label", "RESULTS.md", "(rates **ESTIMATE**, compute **MEASURED**)", "(rates **MEASURED**, compute **MEASURED**)"),
    ("label", "RESULTS.md", "no clear recovery of the power gap (**ESTIMATE**)", "no clear recovery of the power gap (**MEASURED**)"),
    ("prose", "RESULTS.md", "the lab's `2,000` have none", "the lab's `2,000` include them"),
    ("prose", "RESULTS.md", "Different procedures: per-study permutation", "Same procedure: per-study permutation"),
    ("prose", "RESULTS.md", "no clear recovery", "a clear recovery"),
    ("value", "RESULTS.md", "(`n = 200`, `HR = 2.0/SD`", "(`n = 300`, `HR = 2.0/SD`"),
    ("value", "demo/data/claims.json", '"patients": 200,', '"patients": 300,'),
    ("value", "demo/data/claims.json", '"p": 0.0981,', '"p": "0.098",'),
    ("value", "demo/data/claims.json", '"tests": 3780,', '"tests": "3,780",'),
    ("value", "demo/data/claims.json", '"tests": 3780,', '"tests": 3780.0,'),
    ("prose", "RESULTS.md", "top hit is not testable on UPMC (PD1 absent)", "top hit is testable on UPMC (PD1 present)"),
    ("prose", "docs/JUDGING.md", "top hit is not testable on UPMC (PD1 absent)", "top hit replicates on UPMC"),
    ("prose", "RESULTS.md", "interpolated between", "measured between"),
    ("prose", "RESULTS.md", "the locked test does not pass", "the locked test passes"),
    ("prose", "RESULTS.md", "exact parity", "close parity"),
    ("prose", "RESULTS.md", "inference outside timing", "inference inside timing"),
    ("prose", "docs/JUDGING.md", "consistent with chance", "a real effect"),
    ("value", "RESULTS.md", "`0.0301 s` on 64 cores", "`0.0401 s` on 64 cores"),
    ("value", "RESULTS.md", "64 cores: `38.2×`", "64 cores: `48.2×`"),
    ("label", "RESULTS.md", "of the 8-core z (**MEASURED**)", "of the 8-core z (**ESTIMATE**)"),
    ("value", "RESULTS.md", "per decided study (`2,000` decided", "per decided study (`3,000` decided"),
    ("value", "RESULTS.md", "`1` lab core vs `8` Biopsy Scan workers", "`1` lab core vs `4` Biopsy Scan workers"),
    ("prose", "RESULTS.md", "different procedures (per-study permutation", "the same procedure (per-study permutation"),
    ("prose", "RESULTS.md", "not a guaranteed speed-up", "a guaranteed speed-up"),
    ("value", "RESULTS.md", "the same `1,000` null studies", "the same `10,000` null studies"),
    ("value", "RESULTS.md", "`1,000` null studies at", "`1,000` alternative studies at"),
    ("value", "RESULTS.md", "at `1–64` Modal cores", "at `1–128` Modal cores"),
    ("prose", "RESULTS.md", "not separated from host variance", "caused by host variance"),
    ("prose", "RESULTS.md", "host, cache and scheduling effects not isolated", "host effects ruled out"),
    ("link", "RESULTS.md", "([caveats](bench/results/modal-scaling-20261121T201008Z.caveats.json))", "([caveats](missing.json))"),
    ("link", "RESULTS.md", "(bench/results/matched-e2e-20261121T201925Z.json) |", "(missing.json) |"),
    ("prose", "RESULTS.md", "| **Matched end to end** (lab vs Biopsy Scan, same container, design and seeds, #123)",
     "| **Matched end to end** (lab vs Biopsy Scan, #123)"),
    ("value", "RESULTS.md", "paid for): `42.1×`", "paid for): `24.1×`"),
    ("value", "RESULTS.md", "decisions agree `89/100`", "decisions agree `98/100`"),
    ("label", "RESULTS.md", "(**MEASURED**, outcomes SYNTHETIC)", "(**ESTIMATE**, outcomes SYNTHETIC)"),
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


SOURCE_MUTANTS = [  # (source, path, new value, claim, value name): a timing or workload input that moves must drift
    ("bench/results/modal-scaling-20261121T201008Z.json", ["points", "4", "studies_wall_s"], 400.0, "cpu-scaling-38.2x", "speedup_4"),
    ("bench/results/modal-scaling-20261121T201008Z.json", ["points", "1", "studies_wall_s"], 1300.0, "cpu-scaling-38.2x", "wall_1"),
    ("bench/results/modal-scaling-20261121T201008Z.json", ["points", "64", "studies_wall_s"], 40.0, "cpu-scaling-38.2x", "speedup_64"),
    ("bench/results/modal-scaling-20261121T201008Z.json", ["workload", "studies"], 10000, "cpu-scaling-38.2x", "studies"),
    ("bench/results/modal-scaling-20261121T201008Z.json", ["workload", "stream"], "alternative", "cpu-scaling-38.2x", "stream"),
    ("bench/results/matched-e2e-20261121T201925Z.json", ["wall", "with_setup", "lab_s_per_study"], 6.0, "matched-e2e-42.1x", "ratio_setup"),
    ("bench/results/matched-e2e-20261121T201925Z.json", ["arms", "lab", "stages", "studies_wall_s"], 600.0, "matched-e2e-42.1x", "lab"),
    ("bench/results/matched-e2e-20261121T201925Z.json", ["arms", "preflight", "stages", "studies_wall_s"], 600.0, "matched-e2e-42.1x", "pf"),
    ("bench/results/matched-e2e-20261121T201925Z.json", ["wall", "lab_s_per_study"], 6.0, "matched-e2e-42.1x", "ratio"),
    ("bench/results/matched-e2e-20261121T201925Z.json", ["wall", "studies", "lab"], 200, "matched-e2e-42.1x", "lab_n"),
]


@pytest.mark.parametrize("src,keys,value,cid,name", SOURCE_MUTANTS, ids=[f"{m[3]}-{m[4]}" for m in SOURCE_MUTANTS])
def test_a_moved_timing_or_workload_input_fails(copy, src, keys, value, cid, name):
    path = copy / src
    data = json.loads(path.read_text())
    node = data
    for k in keys[:-1]:
        node = node[k]
    assert keys[-1] in node
    node[keys[-1]] = value
    path.write_text(json.dumps(data))
    assert any(b[:2] == (cid, name) for b in check_claims.drift(copy))
    assert check_claims.main(["check_claims", str(copy)]) == 1


def test_every_local_link_in_a_bound_quote_exists():
    ledger = json.loads((ROOT / check_claims.LEDGER).read_text())
    links = [m for c in ledger["claims"] for q in c["quotes"] if "text" in q
             for m in re.findall(r"\]\(([^)#]+)\)", q["text"]) if "://" not in m]
    assert links and [x for x in links if not (ROOT / x).is_file()] == []
