"""Spread for the race's speed claims: independent repeats of #120's lab and fused arms, summarised.

    uv run python -m bench.repeats --lab a.json b.json c.json --fused d.json e.json f.json \\
        --spend bench/results/modal-spend-repeats.json --output bench/results/repeats-<utc>.json

Each input is a bench/results/modal-design-*.json written by #120's runner (bench/modal_design.py on
reimage-fast-b9, run unchanged) on its container class (Modal A100 + 8 cores + 64 GiB), one arm per run.
Per arm: median and min-max of wall per evaluated study over the repeats. Ratio lab/fused: the ratio of
medians, with the range from the extreme pairs (min lab / max fused to max lab / min fused).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

HARNESS = {"pr": 120, "branch": "reimage-fast-b9", "commit": "a29f1b7bb3c1f0d747121b52c04b285aa1bd6d93",
           "runner": "bench/modal_design.py (unchanged, run from a detached checkout)"}


def extract(path, arm: str) -> dict:
    data = Path(path).read_bytes()
    r = json.loads(data)
    if r["status"] != "MEASURED":
        raise ValueError(f"{path}: status {r['status']}, need MEASURED")
    a = r["arms"][arm]
    st = a["stages"]
    return {"file": Path(path).name, "file_sha256": hashlib.sha256(data).hexdigest(), "source_commit": r["source_commit"],
            "workers": a["workers"], "evaluated_studies": st["evaluated_studies"], "studies_wall_s": st["studies_wall_s"],
            "compute_wall_s": st["compute_wall_s"], "wall_per_study_s": st["wall_per_study_s"],
            "end_to_end_in_container_s": st["end_to_end_in_container_s"], "warm_s": a["warm_s"],
            "host": a["host"], "modal": r["modal"]}


def _spread(values) -> dict:
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(repeats: dict) -> dict:
    out = {arm: _spread([x["wall_per_study_s"] for x in reps]) for arm, reps in repeats.items()}
    lab, fused = out["lab"], out["fused"]
    out["ratio"] = {"median": lab["median"] / fused["median"], "min": lab["min"] / fused["max"],
                    "max": lab["max"] / fused["min"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", nargs="+", required=True)
    ap.add_argument("--fused", nargs="+", required=True)
    ap.add_argument("--spend", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f"preserving existing evidence: {out}")
    repeats = {"lab": [extract(p, "lab") for p in a.lab], "fused": [extract(p, "fused") for p in a.fused]}
    commits = {x["source_commit"] for reps in repeats.values() for x in reps}
    if commits != {HARNESS["commit"]}:
        raise ValueError(f"every repeat must run #120 at {HARNESS['commit']}, got {commits}")
    s = summarize(repeats)
    spend = json.loads(Path(a.spend).read_text())
    doc = {
        "status": "MEASURED", "harness": HARNESS,
        "scope": "wall per evaluated study = studies_wall_s / evaluated_studies inside the container (re-imaging and "
                 "scoring; excludes data load, outcome prep, adjacency and warm-up), as the race reports it. lab: "
                 "8 single-thread workers, cKDTree + 1,000-shuffle permutation test, null + alternative studies. "
                 "fused: 8 workers, fused Numba re-imaging + Cox screen, calibration + null + alternative studies.",
        "container_class": "Modal A100 + 8 cores + 64 GiB, one fresh container per repeat",
        "headline": (f"Over {s['lab']['n']} independent repeats each: lab {s['lab']['median']:.3f} s/study "
                     f"(range {s['lab']['min']:.3f}-{s['lab']['max']:.3f}), fused {s['fused']['median']:.4f} s/study "
                     f"(range {s['fused']['min']:.4f}-{s['fused']['max']:.4f}); ratio {s['ratio']['median']:.1f}x "
                     f"(range {s['ratio']['min']:.1f}-{s['ratio']['max']:.1f}x)."),
        "summary": s, "repeats": repeats,
        "spend": {"log": a.spend, "running_total_usd": spend["running_total_usd"]},
    }
    out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(doc["headline"])


if __name__ == "__main__":
    main()
