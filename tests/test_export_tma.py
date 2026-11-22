"""demo/data/tma.json (demo/collect_tma.py): real Charville cores in the shape the Preflight workflow section reads."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_data", ROOT / "demo" / "check_data.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)
PATH = ROOT / "demo" / "data" / "tma.json"


def test_tma_json_shape():
    assert PATH.stat().st_size < 2_500_000
    d = json.loads(PATH.read_text())
    cd.check_tma(d, "tma.json")
    cores = d["cores"]
    assert len(cores) == 36 and len({c["patient"] for c in cores}) == 36
    assert d["markers"] == ["DAPI", "PanCK", "CD8", "CD68", "aSMA", "CD20"]
    hero = cores[d["pick"]]
    assert hero["region"] == "Charville_c003_v001_r001_reg112" and hero["patient"] == "CHV-24133"
    assert len(hero["cells"]) == hero["n_cells"]
    assert all(len(c["cells"]) <= 1200 for i, c in enumerate(cores) if i != d["pick"])


def test_check_tma_refuses_a_cell_outside_its_core():
    d = json.loads(PATH.read_text())
    d["cores"][1]["cells"][0][0] = d["cores"][1]["size_um"][0] + 5
    with pytest.raises(SystemExit):
        cd.check_tma(d, "tma.json")
