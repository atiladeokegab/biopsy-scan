"""Spread for the race's speed claims: fresh-container timing reruns of #120's lab and fused arms, summarised.

    uv run python -m bench.repeats --lab a.json b.json c.json --fused d.json e.json f.json \\
        --fused-previous g.json h.json i.json --prior bench/results/modal-design-20261121T194458Z.json \\
        --spend bench/results/modal-spend-repeats.json --output bench/results/repeats-<utc>.json

Each input is a bench/results/modal-design-*.json written by #120's runner (bench/modal_design.py on
reimage-fast-b9, run unchanged) on its container class (Modal A100 + 8 cores + 64 GiB), one arm per run.
--prior is #113's four-arm run (committed on integration since #120, sha256 PRIOR_SHA256); every repeat's
studies are checked against it. Per arm: median and min-max of wall per evaluated study over the repeats.
Ratio lab/fused: the ratio of medians, with the range from the extreme pairs. Descriptive only.
Exits 1 and writes nothing when any repeat fails the external parity check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HARNESS = {"pr": 120, "branch": "reimage-fast-b9", "commit": "0b27ea1f79db7d45a7be0c35b2866f0dce338d3d",
           "previous_commit": "a29f1b7bb3c1f0d747121b52c04b285aa1bd6d93",
           "runner": "bench/modal_design.py (unchanged, run from a detached checkout)"}
# 0b27ea1 is the head that merges (fused crop fix) and is what the published ratio uses; a29f1b7's fused runs sit beside it.
PREVIOUS = "fused_a29f1b7"
STATUS = "MEASURED timing; parity checked against #113's four-arm run, not within the run"
PRIOR_SHA256 = "2edfce37ab1c6a04550ab393f210a7504ca455b27b98503cbe098e07ff500326"  # modal-design-20261121T194458Z.json

ONE_ARM = {"all": False, "this_run": "one arm"}  # the runner's parity for a single-arm run without --compare-with
Z_TOL = 1e-9
TIMINGS = ("studies_wall_s", "compute_wall_s", "wall_per_study_s", "end_to_end_in_container_s")
STREAMS = {"lab": ("null", "alternative"), "fused": ("calibration", "null", "alternative")}  # the lab skips calibration
SCOPE = {"workers": 8, "container_class": "Modal A100 + 8 cores + 64 GiB",
         "modal": {"gpu": "A100", "cpu": 8.0, "memory_mib": 65536}}  # the race's container class; every run must match


def planned(design: dict, arm: str) -> dict:
    """The run's declared plan, from its design, not from a summary the runner derives from its own records."""
    n = design["studies_per_stream"]
    if not (isinstance(n, int) and not isinstance(n, bool) and n > 0):
        raise ValueError("studies_per_stream must be a positive integer")
    return {st: n if st in STREAMS["lab" if arm == "lab" else "fused"] else 0 for st in STREAMS["fused"]}


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_studies(studies: list, plan: dict, arm: str, where: str) -> None:
    """Exactly the planned (stream, i) set, each once, with finite statistics (lab p also in [0, 1])."""
    keys = [(s["stream"], s["i"]) for s in studies]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{where}: duplicate study IDs")
    want = {(st, i) for st, n in plan.items() for i in range(n)}
    if set(keys) != want:
        raise ValueError(f"{where}: studies do not cover the plan {plan} exactly "
                         f"({len(want - set(keys))} missing, {len(set(keys) - want)} unplanned)")
    for s in studies:
        if not _finite(s["maxT"]) or (arm == "lab" and not (_finite(s.get("p")) and 0 <= s["p"] <= 1)):
            raise ValueError(f"{where}: non-finite statistic at {s['stream']} {s['i']}")


def load_prior(path) -> dict:
    data = Path(path).read_bytes()
    prior = json.loads(data)
    if prior["status"] != "MEASURED":
        raise ValueError("--prior must be MEASURED")
    digest = hashlib.sha256(data).hexdigest()
    if digest != PRIOR_SHA256:  # only #113's own file may stand as the reference
        raise ValueError(f"--prior sha256 {digest} is not the pinned {PRIOR_SHA256}")
    for arm, v in prior["arms"].items():  # validated before it becomes a dict, so a duplicate cannot hide
        plan = planned(prior["design"], arm)
        if v["summary"]["studies"] != plan:
            raise ValueError(f"prior {arm}: summary {v['summary']['studies']} is not the declared plan {plan}")
        check_studies(v["studies"], plan, "lab" if arm == "lab" else arm, f"prior {arm}")
    prior["_file"], prior["_sha256"] = Path(path).name, digest
    return prior


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
    n = st["evaluated_studies"]
    if not (isinstance(n, int) and not isinstance(n, bool) and n > 0):
        raise ValueError(f"{path}: evaluated_studies must be a positive integer")
    if not all(_finite(st[k]) and st[k] > 0 for k in TIMINGS):
        raise ValueError(f"{path}: every timing must be positive and finite")
    if not math.isclose(st["wall_per_study_s"], st["studies_wall_s"] / n, rel_tol=1e-9):
        raise ValueError(f"{path}: wall_per_study_s is not studies_wall_s / evaluated_studies")
    plan = planned(r["design"], arm)
    if a["summary"]["studies"] != plan:
        raise ValueError(f"{path}: summary {a['summary']['studies']} is not the declared plan {plan}")
    check_studies(a["studies"], plan, arm, str(path))
    if len(a["studies"]) != n:
        raise ValueError(f"{path}: {len(a['studies'])} study records for {n} evaluated studies")
    out = {"file": Path(path).name, "file_sha256": hashlib.sha256(data).hexdigest(), "source_commit": r["source_commit"],
           "run_id": a["host"]["modal_task_id"], "container_class": a["container_class"],
           "runner_status": r["status"], "runner_parity": r.get("parity"),
           "runner_status_note": "MEASURED" if r["status"] == "MEASURED" else "FAILED: single-arm, no in-run parity",
           "workers": a["workers"], "evaluated_studies": n,
           "studies_wall_s": st["studies_wall_s"], "compute_wall_s": st["compute_wall_s"],
           "wall_per_study_s": st["wall_per_study_s"], "end_to_end_in_container_s": st["end_to_end_in_container_s"],
           "warm_s": a["warm_s"], "host": a["host"], "modal": r["modal"]}
    if prior is not None:
        design = {k: v for k, v in r["design"].items() if k != "studies_per_stream"}
        if design != {k: v for k, v in prior["design"].items() if k != "studies_per_stream"}:
            raise ValueError(f"{path}: design or seeds differ from the prior's")
        out["parity_vs_prior"] = {"against": prior.get("_file", "prior"), **parity_vs_prior(a["studies"], arm, prior)}
    return out


def check_repeats(repeats: dict) -> None:
    """Distinct runs (by file bytes and by Modal task), each on the declared SCOPE's workers, class and hardware."""
    every = [x for v in repeats.values() for x in v]
    for key in ("file_sha256", "run_id"):
        if len({x[key] for x in every}) != len(every):
            raise ValueError(f"the same run is counted twice ({key})")
    for x in every:
        got = {"workers": x["workers"], "container_class": x["container_class"],
               "modal": {k: x["modal"].get(k) for k in SCOPE["modal"]}}
        if got != SCOPE:
            raise ValueError(f"{x['file']}: {got} is not the declared scope {SCOPE}")


def _spread(values) -> dict:
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(repeats: dict) -> dict:
    out = {arm: _spread([x["wall_per_study_s"] for x in reps]) for arm, reps in repeats.items()}
    lab, fused = out["lab"], out["fused"]
    out["ratio"] = {"median": lab["median"] / fused["median"], "min": lab["min"] / fused["max"],
                    "max": lab["max"] / fused["min"]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", nargs="+", required=True)
    ap.add_argument("--fused", nargs="+", required=True)
    ap.add_argument("--fused-previous", nargs="*", default=[], help="fused runs at HARNESS previous_commit")
    ap.add_argument("--spend", required=True)
    ap.add_argument("--prior", required=True, help=f"#113's MEASURED four-arm result, sha256 {PRIOR_SHA256}")
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f"preserving existing evidence: {out}")
    prior = load_prior(a.prior)
    repeats = {"lab": [extract(p, "lab", prior) for p in a.lab], "fused": [extract(p, "fused", prior) for p in a.fused]}
    if a.fused_previous:
        repeats[PREVIOUS] = [extract(p, "fused", prior) for p in a.fused_previous]
    for arm, reps in repeats.items():
        want = HARNESS["previous_commit" if arm == PREVIOUS else "commit"]
        if {x["source_commit"] for x in reps} != {want}:
            raise ValueError(f"{arm} repeats must run #120 at {want}")
    check_repeats(repeats)
    failed = {x["file"]: x["parity_vs_prior"] for v in repeats.values() for x in v if not x["parity_vs_prior"]["all"]}
    if failed:  # fail closed: no MEASURED headline after a parity failure
        print(json.dumps({"status": "FAILED: a repeat failed the external parity check", "failed": failed}, indent=2),
              file=sys.stderr)
        return 1
    s = summarize(repeats)
    spend = json.loads(Path(a.spend).read_text())
    counts = {k: {x["evaluated_studies"] for x in v} for k, v in repeats.items()}
    if any(len(c) != 1 for c in counts.values()):
        raise ValueError(f"reruns of one arm must evaluate the same studies: {counts}")
    n_lab, n_fused = (f"{min(counts[k]):,}" for k in ("lab", "fused"))
    doc = {
        "status": STATUS,
        "labels": {"compute": "MEASURED", "outcomes": "SYNTHETIC (atlas/simulation.py at 79a7792)",
                   "spend": "ESTIMATE (elapsed wall x list rate, not billing)"},
        "harness": HARNESS,
        "parity": {"prior": {"file": prior["_file"], "sha256": prior["_sha256"],
                             "source_commit": prior["source_commit"],
                             "where": "bench/results/ on integration since #120; written by #113"},
                   "rule": "each repeat: exactly its planned study IDs, each once, with finite statistics; design and "
                           "seeds equal to the prior's; X sha256 equal to every prior arm on every shared study; lab p "
                           "exactly equal to the prior lab arm's; fused maxT within 1e-9 of the prior numba arm's. "
                           "Single-arm runs have no in-run partner, so the runner marks them FAILED with parity "
                           "'one arm'; this check replaces that."},
        "scope": (f"Fresh-container timing reruns of the same seeded studies: lab {n_lab} evaluated studies per repeat "
                  f"(null + alternative), fused {n_fused} (calibration + null + alternative). Wall per evaluated study = "
                  "studies_wall_s / evaluated_studies inside the container (re-imaging and scoring; excludes data load, "
                  "outcome prep, adjacency and warm-up), as the race reports it. lab: 8 single-thread workers, cKDTree "
                  "+ 1,000-shuffle permutation test. fused: 8 workers, fused Numba re-imaging + Cox screen."),
        "ratio_note": ("Descriptive: ratio of median walls per study, range from the extreme pairs. It is not an "
                       "uncertainty interval around the race's 21x, whose lab batch was 2,000 studies, not 200."),
        "container_class": f"{SCOPE['container_class']}, {SCOPE['workers']} workers, one fresh container per repeat",
        "headline": (f"{s['lab']['n']} fresh-container reruns each at #120 0b27ea1: lab {s['lab']['median']:.3f} s/study "
                     f"(range {s['lab']['min']:.3f}-{s['lab']['max']:.3f}, {n_lab} studies per rerun), fused "
                     f"{s['fused']['median']:.4f} s/study (range {s['fused']['min']:.4f}-{s['fused']['max']:.4f}, "
                     f"{n_fused} studies per rerun); ratio of medians {s['ratio']['median']:.1f}x "
                     f"(extreme pairs {s['ratio']['min']:.1f}-{s['ratio']['max']:.1f}x)."
                     + (f" Kept separate: fused at a29f1b7 (before the crop fix) {s[PREVIOUS]['median']:.4f} s/study "
                        f"(range {s[PREVIOUS]['min']:.4f}-{s[PREVIOUS]['max']:.4f}, {s[PREVIOUS]['n']} reruns)."
                        if PREVIOUS in s else "")),
        "summary": s, "repeats": repeats,
        "spend": {"log": a.spend, **{k: spend[k] for k in ("running_total_usd", "completed_usd", "aborted_usd")}},
    }
    out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(doc["headline"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
