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

HARNESS = {"pr": 120, "branch": "reimage-fast-b9", "commit": "0b27ea1f79db7d45a7be0c35b2866f0dce338d3d",
           "previous_commit": "a29f1b7bb3c1f0d747121b52c04b285aa1bd6d93",
           "runner": "bench/modal_design.py (unchanged, run from a detached checkout)"}
# 0b27ea1 is the head that merges (fused crop fix) and is what the published ratio uses; a29f1b7's fused runs sit beside it.
PREVIOUS = "fused_a29f1b7"
STATUS = "MEASURED timing; parity checked against #113's four-arm run, not within the run"


ONE_ARM = {"all": False, "this_run": "one arm"}  # the runner's parity for a single-arm run without --compare-with
Z_TOL = 1e-9


def parity_vs_prior(studies: list, arm: str, prior: dict) -> dict:
    """X sha256 against every prior arm on each shared study; lab p exactly equal to the prior lab's; maxT vs numba."""
    keyed = {a: {(s["stream"], s["i"]): s for s in v["studies"]} for a, v in prior["arms"].items()}
    x_n = x_bad = 0
    stat_n, stat_worst = 0, 0.0
    for s in studies:
        k = (s["stream"], s["i"])
        for a, ks in keyed.items():
            if k in ks:
                x_n += 1
                x_bad += ks[k]["x_sha"] != s["x_sha"]
        ref = keyed["lab" if arm == "lab" else "numba"].get(k)
        if ref is not None:
            stat_n += 1
            if arm == "lab":
                stat_worst = max(stat_worst, 0.0 if ref["p"] == s["p"] else float("inf"))
            else:
                stat_worst = max(stat_worst, abs(s["maxT"] - ref["maxT"]) / max(1.0, abs(ref["maxT"])))
    stat = "p, exact" if arm == "lab" else f"maxT vs numba, rel <= {Z_TOL}"
    ok = x_n > 0 and x_bad == 0 and stat_n == len(studies) and stat_worst <= (0.0 if arm == "lab" else Z_TOL)
    return {"x_compared": x_n, "x_mismatched": x_bad, "statistic": stat, "statistic_compared": stat_n,
            "statistic_max_rel_diff": stat_worst, "all": bool(ok)}


def extract(path, arm: str, prior: dict | None = None) -> dict:
    """One arm's walls from a runner result. A single-arm run is FAILED by the runner only for want of a parity
    partner; it is accepted when its tree was clean and nothing failed, and checked against `prior` instead."""
    data = Path(path).read_bytes()
    r = json.loads(data)
    one_arm_only = (r["status"] == "FAILED" and r.get("parity") == ONE_ARM and r.get("clean_tree")
                    and not r.get("errors") and r.get("same_container_class"))
    if r["status"] != "MEASURED" and not one_arm_only:
        raise ValueError(f"{path}: status {r['status']}, need MEASURED or a clean single-arm run")
    a = r["arms"][arm]
    st = a["stages"]
    out = {"file": Path(path).name, "file_sha256": hashlib.sha256(data).hexdigest(), "source_commit": r["source_commit"],
           "runner_status": r["status"], "runner_parity": r.get("parity"),
           "runner_status_note": "MEASURED" if r["status"] == "MEASURED" else "FAILED: single-arm, no in-run parity",
           "workers": a["workers"], "evaluated_studies": st["evaluated_studies"],
           "studies_wall_s": st["studies_wall_s"], "compute_wall_s": st["compute_wall_s"],
           "wall_per_study_s": st["wall_per_study_s"], "end_to_end_in_container_s": st["end_to_end_in_container_s"],
           "warm_s": a["warm_s"], "host": a["host"], "modal": r["modal"]}
    if prior is not None:
        out["parity_vs_prior"] = {"against": prior.get("_file", "prior"), **parity_vs_prior(a["studies"], arm, prior)}
    return out


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
    ap.add_argument("--fused-previous", nargs="*", default=[], help="fused runs at HARNESS previous_commit")
    ap.add_argument("--spend", required=True)
    ap.add_argument("--prior", required=True, help="a MEASURED modal-design result from #113/#120 at the same seeds")
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f"preserving existing evidence: {out}")
    prior_bytes = Path(a.prior).read_bytes()
    prior = json.loads(prior_bytes)
    prior["_file"] = Path(a.prior).name
    if prior["status"] != "MEASURED":
        raise ValueError("--prior must be MEASURED")
    repeats = {"lab": [extract(p, "lab", prior) for p in a.lab], "fused": [extract(p, "fused", prior) for p in a.fused]}
    if a.fused_previous:
        repeats[PREVIOUS] = [extract(p, "fused", prior) for p in a.fused_previous]
    for arm, reps in repeats.items():
        want = HARNESS["previous_commit" if arm == PREVIOUS else "commit"]
        if {x["source_commit"] for x in reps} != {want}:
            raise ValueError(f"{arm} repeats must run #120 at {want}")
    s = summarize(repeats)
    spend = json.loads(Path(a.spend).read_text())
    doc = {
        "status": (STATUS if all(x["parity_vs_prior"]["all"] for v in repeats.values() for x in v)
                   else "FAILED: a repeat failed the external parity check"),
        "labels": {"compute": "MEASURED", "outcomes": "SYNTHETIC (atlas/simulation.py at 79a7792)"},
        "harness": HARNESS,
        "parity": {"prior": {"file": Path(a.prior).name, "sha256": hashlib.sha256(prior_bytes).hexdigest(),
                             "source_commit": prior["source_commit"], "where": "reimage-fast-b9 (#120), from #113"},
                   "rule": "each repeat: X sha256 equal to every prior arm on every shared study (same seeds); lab p "
                           "exactly equal to the prior lab arm's; fused maxT within 1e-9 of the prior numba arm's. "
                           "Single-arm runs have no in-run partner, so the runner marks them FAILED with parity "
                           "'one arm'; this check replaces that."},
        "scope": "wall per evaluated study = studies_wall_s / evaluated_studies inside the container (re-imaging and "
                 "scoring; excludes data load, outcome prep, adjacency and warm-up), as the race reports it. lab: "
                 "8 single-thread workers, cKDTree + 1,000-shuffle permutation test, null + alternative studies. "
                 "fused: 8 workers, fused Numba re-imaging + Cox screen, calibration + null + alternative studies.",
        "container_class": "Modal A100 + 8 cores + 64 GiB, one fresh container per repeat",
        "headline": (f"Over {s['lab']['n']} independent repeats each: lab {s['lab']['median']:.3f} s/study "
                     f"(range {s['lab']['min']:.3f}-{s['lab']['max']:.3f}), fused {s['fused']['median']:.4f} s/study "
                     f"(range {s['fused']['min']:.4f}-{s['fused']['max']:.4f}); ratio {s['ratio']['median']:.1f}x "
                     f"(range {s['ratio']['min']:.1f}-{s['ratio']['max']:.1f}x), all at #120 0b27ea1."
                     + (f" Fused at a29f1b7 (before the crop fix): {s[PREVIOUS]['median']:.4f} s/study "
                        f"(range {s[PREVIOUS]['min']:.4f}-{s[PREVIOUS]['max']:.4f}, {s[PREVIOUS]['n']} repeats)."
                        if PREVIOUS in s else "")),
        "summary": s, "repeats": repeats,
        "spend": {"log": a.spend, **{k: spend[k] for k in ("running_total_usd", "completed_usd", "aborted_usd")}},
    }
    out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(doc["headline"])


if __name__ == "__main__":
    main()
