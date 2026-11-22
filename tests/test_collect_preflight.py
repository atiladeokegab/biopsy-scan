"""The "Preflight your study" collector refuses source files that do not say what the panel would print (Prometheus, #130)."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collect_preflight", ROOT / "demo" / "collect_preflight.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


@pytest.fixture
def root(tmp_path):
    for p in (cp.CURVE, cp.VS):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / p, tmp_path / p)
    return tmp_path


def edit(root, rel, change):
    path = root / rel
    d = json.loads(path.read_text())
    change(d)
    path.write_text(json.dumps(d))  # NaN is written as NaN, as a careless producer would


def test_the_committed_sources_build_and_match_the_committed_file(root):
    assert cp.same(cp.build(root), json.loads((ROOT / "demo/data/preflight.json").read_text()))


def both_500(root):
    edit(root, cp.CURVE, lambda d: d["design"].update(fov_um=500.0))
    edit(root, cp.VS, lambda d: d["design"].update(fov_um=500.0))


@pytest.mark.parametrize("name, mutate, msg", [
    ("both designs 500 um, still published as 400um", both_500, "design is not"),
    ("empty control series", lambda r: edit(r, cp.VS, lambda d: d["control"].update(points=[])), "no points"),
    ("curve cohort changed alone", lambda r: edit(r, cp.CURVE, lambda d: d.update(cohort="upmc")), "disagree on cohort"),
    ("curve planted HR changed alone", lambda r: edit(r, cp.CURVE, lambda d: d.update(hr_per_sd=1.5)), "disagree on hr_per_sd"),
    ("curve simulation model changed alone",
     lambda r: edit(r, cp.CURVE, lambda d: d["simulated"].update(source_hash="0" * 64)), "simulated.source_hash"),
    ("NaN at the 400-patient point (not shared with the other file)",
     lambda r: edit(r, cp.CURVE, lambda d: next(p for p in d["points"] if p["patients"] == 400).update(power=float("nan"))),
     "not finite"),
    ("duplicate patient count",
     lambda r: edit(r, cp.VS, lambda d: d["control"]["points"].append(dict(d["control"]["points"][-1]))), "strictly increasing"),
    ("zero studies at a point",
     lambda r: edit(r, cp.CURVE, lambda d: d["points"][-1].update(power_n=0)), "positive integer"),
    ("power outside its interval",
     lambda r: edit(r, cp.CURVE, lambda d: d["points"][-1].update(power_lo=1.0)), "outside its interval"),
])
def test_bad_sources_are_refused(root, name, mutate, msg):
    mutate(root)
    with pytest.raises(cp.Bad, match=msg):
        cp.build(root)
