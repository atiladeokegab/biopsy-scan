"""The demo's dials read Atlas's committed Charville grid (atlas 3faac76), labelled for what it is."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_data", ROOT / "demo" / "check_data.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)
GRID = ROOT / "demo" / "data" / "grid.json"


def test_grid_is_atlas_3faac76_verbatim():
    assert hashlib.sha256(GRID.read_bytes()).hexdigest() == "638534d2cef80b06cc1d1883959fb8c840a5f2fb7340fa1e7f80c916ce66c654"
    g = json.loads(GRID.read_text())
    assert g["source"] == "MEASURED" and g["meta"]["simulation_label"] == "SYNTHETIC"
    assert len(g["points"]) == 48 and "legacy R4" in g["note"] and "not an R15" in g["note"]
    cd.check_grid(g, "grid.json")


@pytest.mark.parametrize("bad", [
    lambda p: p.update(power=p["power_hi"] + 0.01),          # outside its own interval
    lambda p: p.update(power_count=p["power_count"] + 1),    # rate not count / studies
    lambda p: p.update(fwer=1.5),
    lambda p: p.pop("power_lo"),
])
def test_measured_grid_numbers_are_checked(bad):
    g = json.loads(GRID.read_text())
    bad(g["points"][5])
    with pytest.raises(SystemExit):
        cd.check_grid(g, "grid.json")


def test_measured_grid_must_say_which_model_it_is():
    g = json.loads(GRID.read_text())
    for note in ("", "SYNTHETIC Weibull outcomes, HR2"):
        h = copy.deepcopy(g)
        h["note"] = note
        with pytest.raises(SystemExit):
            cd.check_grid(h, "grid.json")
