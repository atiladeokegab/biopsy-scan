"""demo/data/tma.json (scripts/export_tma.py): the shape the Preflight workflow section reads."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "demo" / "data" / "tma.json"


def test_tma_json_shape():
    assert PATH.stat().st_size < 3_000_000
    d = json.loads(PATH.read_text())
    assert d["schema"] == "preflight-tma/1" and d["source"] == "MEASURED"
    assert d["meta"]["label"] == "MEASURED cells" and "13179600" in d["meta"]["source"] and "seed" in d["meta"]
    assert len(d["types"]) == 15 and {t["group"] for t in d["types"]} == {"tumour", "stroma", "immune"}
    markers = list(d["meta"]["markers"])
    assert markers == ["DAPI", "PanCK", "CD45", "Vimentin"]
    cores = d["cores"]
    assert len(cores) == 36 and len({c["patient"] for c in cores}) == 36 and len({c["region"] for c in cores}) == 36
    assert all(c["region"].startswith("Charville_") for c in cores)
    heroes = [c for c in cores if c["hero"]]
    assert len(heroes) == 1 and heroes[0]["region"] == d["hero"] == "Charville_c003_v001_r001_reg112"
    assert heroes[0]["patient"] == "CHV-24133" and len(heroes[0]["t"]) == heroes[0]["cells_in_region"]
    for c in cores:
        n = len(c["t"])
        assert 0 < n <= c["cells_in_region"] and (c["hero"] or n <= 2500)
        assert len(c["x"]) == len(c["y"]) == n and min(c["x"]) == 0 and min(c["y"]) == 0
        assert all(0 <= t < 15 for t in c["t"])
        assert list(c["m"]) == markers and all(len(v) == n and min(v) >= 0 and max(v) <= 255 for v in c["m"].values())
