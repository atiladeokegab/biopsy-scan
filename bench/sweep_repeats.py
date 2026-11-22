"""Three fresh repeats of each single-sweep result (#75 fleet, #115 strong scaling, #123 matched e2e) and the
equal-hardware matched run (lab on 8 workers), reduced to median and range, as bench/repeats.py does for #120.

    uv run python -m bench.sweep_repeats [bench/results/sweep-repeats-raw] [out.json]

Every raw runner file is checked before any number is used: the prior's pinned sha256; the run's own status and
parity flags; the planned studies covered exactly once; positive finite timings; a unique run (file sha256 and Modal
task id); the declared container class; and parity against the prior run on the studies both ran. Any failure prints
why, exits 1 and writes no file: no headline over a parity failure. Ranges are descriptive (3 runs), not intervals.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
PRIORS = {  # the single sweeps the task repeats, pinned by content
    "fleet": ("modal-a100-20261121T183709Z.json", "fd184953cf1db869b09dd99cccf518f72148d0d8bf0009c39399cc6a1d72e696"),
    "scaling": ("modal-scaling-20261121T201008Z.json", "5318583c0aa5890f63f662640b2167457373d64c897c34e9d784b93e916383de"),
    "matched": ("matched-e2e-20261121T201925Z.json", "2af4fba937c59678d43f85f9dcef491ca3892cbf0132be9f49613ba5ceb5aab5"),
}
PRIORS["fair"] = PRIORS["matched"]  # same design and seeds; parity on the 100 studies both lab samples share
LAB_WORKERS = {"matched": 1, "fair": 8}
LAB_PLAN = {"matched": {"null": 50, "alternative": 50}, "fair": {"null": 200, "alternative": 200}}  # as launched
ALPHA = 0.05  # bench/b9_design_batch.ALPHA: the lab calls p <= ALPHA
MATCHED_CLASS = {"cpu": 8.0, "memory_mib": 65536, "gpu": None}
FLEET = ("1", "8")
# The screen hook's p is a 999-resample count: #80 (7660440) changed stats.screen's resample streams after the prior
# ran (it used the pinned 8add7f9 file). Locally on CPU, numpy backend: 8add7f9 gives p 0.789 (the prior's), 3647816
# gives 0.769; maxT 3.0589389780934853 at both. So maxT is checked against the prior (to rounding), p against the code.
SCREEN_P_AT_HEAD = 0.769
GPU_REQUESTED, GPU_MODEL = "A100-40GB", "NVIDIA A100-SXM4-40GB"  # the pinned request, and the one model it may yield
Z_TOL = 1e-9  # bench/scaling.Z_TOL: features bitwise, scores to rounding


class Bad(ValueError):
    pass


def need(ok, why: str) -> None:
    if not ok:
        raise Bad(why)


def pos(*v) -> bool:
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in v)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prior(kind: str) -> dict:
    name, want = PRIORS[kind]
    path = RESULTS / name
    need(sha(path) == want, f"prior {name} sha256 is not the pinned {want[:12]}")
    return json.loads(path.read_text())


def matched(doc: dict, prior: dict, kind: str) -> dict:
    lw = LAB_WORKERS[kind]
    need(doc["status"] == "MEASURED" and doc["parity"]["x_bitwise"], "status or feature parity is not MEASURED/bitwise")
    need({k: doc["modal"].get(k) for k in MATCHED_CLASS} == MATCHED_CLASS, f"container class {doc['modal']}")
    lab, pre = doc["arms"]["lab"], doc["arms"]["preflight"]
    pinned = len(lab["affinity"]) == 1 if lw == 1 else lab["affinity"] == doc["host"]["affinity"]  # 8: unpinned, as Preflight
    need(pre["workers"] == 8 and lab["workers"] == lw and pinned, f"lab workers {lab['workers']}, affinity {lab['affinity']}")
    need(doc["plan"]["preflight"] == prior["plan"]["preflight"], "Preflight plan differs from the prior's")
    need(doc["plan"]["lab"] == LAB_PLAN[kind], f"lab plan {doc['plan']['lab']} is not the declared {LAB_PLAN[kind]}")
    want = {(s, i) for s, n in LAB_PLAN[kind].items() for i in range(n)}
    rows = doc["per_study"]
    ids = [(r["stream"], r["i"]) for r in rows]
    need(len(ids) == len(set(ids)) and set(ids) == want, "lab studies are not the plan, exactly once")
    need(all(r["x_equal"] is True for r in rows) and doc["parity"]["x_compared"] == len(rows),
         "a study's features differ between the arms")
    # every claim is rebuilt from the rows and the arms' own operands, never from cached ratios or totals
    cut = doc["threshold"]
    need(pos(cut) and all(r["lab_call"] == (r["lab_p"] <= ALPHA) and r["preflight_call"] == (r["preflight_maxT"] > cut)
                          and r["agree"] == (r["lab_call"] == r["preflight_call"]) for r in rows),
         "a study's calls or agreement do not follow from its p, maxT and the threshold")
    calls = {st: {"lab": sum(r["lab_call"] for r in rows if r["stream"] == st),
                  "preflight": sum(r["preflight_call"] for r in rows if r["stream"] == st),
                  "n": sum(r["stream"] == st for r in rows)} for st in ("null", "alternative")}
    decisions = {"agree": sum(r["agree"] for r in rows), "compared": len(rows), "calls": calls}
    need(doc["decisions"] == decisions, f"decision totals {doc['decisions']} are not the rows' {decisions}")
    n_lab, n_pre = len(rows), sum(doc["plan"]["preflight"].values())
    decided = n_pre - doc["plan"]["preflight"]["calibration"]
    need(lab["stages"]["evaluated_studies"] == n_lab and pre["stages"]["evaluated_studies"] == n_pre,
         "an arm's evaluated studies do not match the plan")
    lab_wall, pre_wall = lab["stages"]["studies_wall_s"], pre["stages"]["studies_wall_s"]
    setup = lambda o: o["setup"]["prep_s"] + o["setup"]["adjacency_s"] + (o["warm_s"] or 0)  # noqa: E731 - as compare()
    need(pos(lab_wall, pre_wall) and all(v >= 0 and math.isfinite(v) for v in (setup(lab), setup(pre))),
         "a timing is not positive and finite")
    lab_s = lab_wall / n_lab
    ws = {"lab_s_per_study": (lab_wall + setup(lab)) / n_lab, "preflight_s_per_decided_study": (pre_wall + setup(pre)) / decided}
    ws["ratio"] = ws["lab_s_per_study"] / ws["preflight_s_per_decided_study"]
    w = {"lab_s_per_study": lab_s, "preflight_s_per_decided_study": pre_wall / decided,
         "ratio": lab_s / (pre_wall / decided), "with_setup": ws}
    rec = doc["wall"]
    need(all(math.isclose(rec[k], w[k], rel_tol=1e-12) for k in ("lab_s_per_study", "preflight_s_per_decided_study", "ratio"))
         and all(math.isclose(rec["with_setup"][k], ws[k], rel_tol=1e-12) for k in ws)
         and rec["studies"] == {"lab": n_lab, "preflight_evaluated": n_pre, "preflight_decided": decided,
                                "preflight_calibration": n_pre - decided},
         "a recorded wall or ratio does not follow from the arms' walls, setup and study counts")
    near = lambda a, b: abs(a - b) <= Z_TOL * max(1, abs(b))  # noqa: E731 - scores to rounding, as bench/scaling.py
    need(near(cut, prior["threshold"]), f"threshold {cut} != prior {prior['threshold']}")
    old = {(r["stream"], r["i"]): r for r in prior["per_study"]}
    shared = [r for r in doc["per_study"] if (r["stream"], r["i"]) in old]
    need(len(shared) == len(old), f"{len(shared)} of the prior's {len(old)} studies shared")
    for r in shared:
        o = old[(r["stream"], r["i"])]
        need(r["x_sha"] == o["x_sha"] and r["lab_p"] == o["lab_p"] and near(r["preflight_maxT"], o["preflight_maxT"])
             and (r["lab_call"], r["preflight_call"]) == (o["lab_call"], o["preflight_call"]),
             f"study {r['stream']} {r['i']} differs from the prior run")
    return {"run_id": doc["host"]["modal_task_id"], "source_commit": doc["source_commit"],
            "values": {"ratio": w["ratio"], "ratio_with_setup": w["with_setup"]["ratio"],
                       "lab_s_per_study": w["lab_s_per_study"],
                       "preflight_s_per_decided_study": w["preflight_s_per_decided_study"]},
            "decisions_agree": [decisions["agree"], decisions["compared"]],
            "parity_vs_prior": f"{len(shared)} shared studies: features bitwise, lab p identical, Preflight maxT and the "
                               f"threshold within {Z_TOL} relative, both arms' calls identical"}


def fleet(doc: dict, prior: dict, kind: str) -> dict:
    need(doc["status"] == "MEASURED" and doc["kind"] == "ladder" and not doc["dirty"], "status, kind or dirty tree")
    f, pf = doc["fleet"], prior["fleet"]
    need((f["studies"], f["seeds"], f["design"]) == (pf["studies"], pf["seeds"], pf["design"]), "fleet plan differs")
    need(set(FLEET) <= set(f["runs"]), f"fleet sizes {sorted(f['runs'])}")
    for n in FLEET:
        r = f["runs"][n]
        need(r["containers_seen"] == r["containers_requested"] == int(n) and r["seed_coverage_exact"]
             and r["distinct_task_ids"] and r["bit_identical_to_1"] and r["all_overlapped"], f"fleet {n}: a check failed")
        need(sum(r["studies_per_container"]) == f["studies"], f"fleet {n}: studies not covered")
        need(pos(r["compute_span_s"], r["modal_wall_s"]), f"fleet {n}: a timing is not positive and finite")
        models = r.get("gpu_models", [])
        need(r.get("same_gpu_model") is True and len(models) == int(n) and set(models) == set(r["gpus"]) == {GPU_MODEL}
             and doc["gpu_requested"] == GPU_REQUESTED, f"fleet {n}: GPUs {models or r['gpus']}, not {n} x {GPU_MODEL}")
    k, pk = doc["kernel"], prior["kernel"]
    need(k["parity_exact"] and k["seeds"] == pk["seeds"], "kernel parity or seeds differ")
    s, ps = doc["screen"]["backends"], prior["screen"]["backends"]
    ran = [b for b in s.values() if isinstance(b, dict)]
    need("numpy" in s and all(abs(b["maxT"] - ps["numpy"]["maxT"]) <= Z_TOL for b in ran),  # hosts differ by an ulp
         "screen hook's maxT differs from the prior run")
    need(all(b["p"] == SCREEN_P_AT_HEAD for b in ran), f"screen hook's p is not {SCREEN_P_AT_HEAD} on every backend")
    one, eight = f["runs"]["1"], f["runs"]["8"]
    return {"run_id": doc["kernel"]["gpu"]["modal_task_id"], "source_commit": doc["commit"],
            "values": {"speedup_compute_span": one["compute_span_s"] / eight["compute_span_s"],
                       "speedup_modal_wall": one["modal_wall_s"] / eight["modal_wall_s"],
                       "compute_span_1_s": one["compute_span_s"], "compute_span_8_s": eight["compute_span_s"]},
            "parity_vs_prior": "same studies, seeds and design; each fleet bit-identical to its own 1-container run; "
                               f"kernel parity exact on the prior's seeds; screen hook maxT within {Z_TOL} of the prior's, its p "
                               f"{SCREEN_P_AT_HEAD} on every backend (the current code's; see SCREEN_P_AT_HEAD) "
                               "(the prior file stores no per-study fleet outputs to compare)"}


def scaling(doc: dict, prior: dict, kind: str) -> dict:
    need(doc["status"] == "MEASURED" and not doc["failures"] and not doc["errors"] and doc["parity"]["all"],
         "status, failures or within-run parity")
    need({k: v for k, v in doc["workload"].items() if k != "arm"} == {k: v for k, v in prior["workload"].items() if k != "arm"},
         "workload (studies, seeds, design) differs from the prior's")
    need(doc["container_class"] == prior["container_class"], "container classes differ from the prior's")
    curve = {r["cores"]: r for r in doc["curve"]}
    cores = sorted(r["cores"] for r in prior["curve"])
    need(len(doc["curve"]) == len(curve) and sorted(curve) == cores, f"cores {[r['cores'] for r in doc['curve']]}")
    need(set(doc["points"]) == set(doc["studies_by_cores"]) == {str(c) for c in cores},
         f"points {sorted(doc['points'])} or study lists {sorted(doc['studies_by_cores'])} are not one per core count {cores}")
    tasks = [p["host"]["modal_task_id"] for p in doc["points"].values()]
    need(all(tasks) and len(set(tasks)) == len(tasks), "point task ids missing or repeated")
    n = doc["workload"]["studies"]
    old = {r["i"]: r for r in prior["studies_by_cores"]["1"]}
    for c, rows in doc["studies_by_cores"].items():
        need(sorted(r["i"] for r in rows) == list(range(n)), f"{c} cores: studies not 0..{n - 1} exactly once")
        for r in rows:
            o = old[r["i"]]
            need(r["x_sha"] == o["x_sha"] and abs(r["maxT"] - o["maxT"]) <= Z_TOL * max(1, abs(o["maxT"])),
                 f"{c} cores, study {r['i']}: differs from the prior's 1-core run")
    need(all(pos(r["studies_wall_s"]) for r in curve.values()), "a timing is not positive and finite")
    for c, r in curve.items():  # the curve is a summary: each row must be its point's own wall, never edited apart from it
        p = doc["points"][str(c)]
        need(r["studies_wall_s"] == p["studies_wall_s"] and r["workers"] == p["workers"] and r["studies"] == n
             and math.isclose(r["wall_per_study_s"], r["studies_wall_s"] / n, rel_tol=1e-12)
             and math.isclose(r["speedup_vs_1_core"], curve[1]["studies_wall_s"] / r["studies_wall_s"], rel_tol=1e-12),
             f"{c} cores: the curve row does not follow from its point's studies_wall_s")
    top = max(curve)
    return {"run_id": tasks[0], "source_commit": doc["source_commit"],
            "values": {f"speedup_{top}_cores": curve[1]["studies_wall_s"] / curve[top]["studies_wall_s"],
                       "wall_per_study_1_core_s": curve[1]["wall_per_study_s"],
                       f"wall_per_study_{top}_cores_s": curve[top]["wall_per_study_s"]},
            "parity_vs_prior": f"every core count: all {n} studies' features bitwise and maxT within {Z_TOL} of the "
                               "prior sweep's 1-core run"}


REDUCE = {"fleet": fleet, "scaling": scaling, "matched": matched, "fair": matched}


def failed_attempt(path: Path, commit: str) -> dict:
    """A run the runner itself marked FAILED (no timing published): kept as evidence, never in the spread."""
    doc = json.loads(path.read_text())
    need(doc.get("status") == "FAILED" and doc.get("commit", doc.get("source_commit")) == commit,
         f"{path.name}: a failed attempt must be a FAILED run at {commit[:7]}")
    return {"file": path.name, "file_sha256": sha(path), "status": "FAILED",
            "why": doc.get("parity_failed") or doc.get("failures") or doc.get("note")}


def reduce(kind: str, paths: list[Path], commit: str, want: int = 3) -> dict:
    prior = load_prior(kind)
    runs = []
    for p in paths:
        try:
            r = REDUCE[kind](json.loads(p.read_text()), prior, kind)
            need(r["source_commit"] == commit, f"ran {r['source_commit'][:7]}, not {commit[:7]}")
        except (Bad, KeyError, TypeError) as e:
            raise Bad(f"{kind} {p.name}: {e!r}") from None
        runs.append({"file": p.name, "file_sha256": sha(p), **r})
    need(len(runs) == want >= 2, f"{kind}: {len(runs)} repeats, not the declared {want}")
    for key in ("file_sha256", "run_id"):
        need(len({r[key] for r in runs}) == want, f"{kind}: a repeat appears twice ({key})")
    spread = {k: {"median": statistics.median(v), "min": min(v), "max": max(v)}
              for k in runs[0]["values"] for v in [[r["values"][k] for r in runs]]}
    return {"status": "MEASURED", "prior": PRIORS[kind][0], "repeats": runs, "spread": spread}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    raw = Path(argv[0]) if argv else RESULTS / "sweep-repeats-raw"
    commits = json.loads((raw / "commits.json").read_text())  # {kind: {commit each repeat ran, repeats declared}}
    try:
        out = {}
        for kind, c in commits.items():
            out[kind] = reduce(kind, sorted((raw / kind).glob("*.json")), c["commit"], c["repeats"])
            out[kind]["failed_attempts"] = [failed_attempt(p, c["commit"])
                                            for p in sorted((raw / f"{kind}-failed").glob("*.json"))]
    except Bad as e:
        print(f"no result written: {e}", file=sys.stderr)
        return 1
    out = {"labels": {"timings": "MEASURED", "outcomes": "SYNTHETIC", "spread": "median and min-max of each kind's declared fresh runs; "
                      "descriptive, not a confidence interval"}, **out}
    path = Path(argv[1]) if len(argv) > 1 else RESULTS / "sweep-repeats.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    for kind in commits:
        print(kind, json.dumps({k: [round(v["median"], 4), round(v["min"], 4), round(v["max"], 4)]
                                for k, v in out[kind]["spread"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
