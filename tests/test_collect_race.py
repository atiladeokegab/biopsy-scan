"""The speed race's collector: bench files feed rows by id, and the committed race matches its source file."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collect_race", ROOT / "demo" / "collect_race.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def test_race_list_feeds_rows_by_id():
    d = {"race": [{"id": "preflight-a100", "seconds": 0.8, "label": "MEASURED", "hardware": "Modal", "gpu": "1x A100"}]}
    [(rid, f)] = cr.rows_from(Path("modal-a100.json"), d)
    assert rid == "preflight-a100" and f["seconds"] == 0.8 and f["source"] == "bench/results/modal-a100.json"


def test_impl_a_feeds_standard_with_most_workers():
    d = {"cpu_model": "X", "cohorts": [{"cohort": "charville", "candidates": 3780, "timings": [
        {"workers": 1, "wall_s": 300.0, "label": "MEASURED"}, {"workers": 8, "wall_s": 64.0, "label": "MEASURED"}]}]}
    [(rid, f)] = cr.rows_from(Path("impl-a.json"), d)
    assert rid == "standard" and f["seconds"] == 64.0 and f["cores"] == 8 and f["note"] == "300.0 s on 1 core"


def test_other_bench_files_feed_nothing():
    assert list(cr.rows_from(Path("ladder.json"), {"rungs": []})) == []


def test_committed_race_matches_bench_results():
    race = {r["id"]: r for r in json.loads((ROOT / "demo" / "data" / "baseline.json").read_text())["rows"]}
    for path in sorted((ROOT / "bench" / "results").glob("*.json")):
        d = json.loads(path.read_text())
        for rid, fields in (cr.rows_from(path, d) if isinstance(d, dict) else []):
            assert race[rid]["seconds"] == fields["seconds"], f"{rid}: rerun python3 -I demo/collect_race.py"
