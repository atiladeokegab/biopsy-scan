"""Matched comparison, compute through scoring: the lab's workflow vs Preflight's, same studies, seeds, design and container.

Why: bench/results/cpu-ladder.json's lab_today (7.52 s/study) re-images AND runs a 1,000-shuffle permutation
test, while its ladder rows are feature-only, so it marks their quotient UNMEASURED ("Feature-only and full
legacy inference have different scope"). Here both arms do features and the per-study decision.

  lab        build-9's lab arm = bench.lab_today's path: fresh cKDTree re-imaging, then stats.permutation_test
             with 1,000 within-strata shuffles, call p <= 0.05. One study at a time, one pool worker pinned
             to one core. Runs a fixed sample: the first `lab_per_stream` null and alternative studies.
  preflight  build-9's numba arm: CSR Numba re-imaging, then the Cox score screen, call maxT > the design
             threshold from the calibration nulls. 8 single-thread workers. Its wall includes the calibration
             studies the threshold needs.

Both arms run bench/b9_design_batch.py, an unedited copy of build-9's harness (B9), in one container.
"""
from __future__ import annotations

import math
import os

import numpy as np

from bench import b9_design_batch as db

B9 = {"branch": "modal-design-b9", "commit": "093c1eea6eda7e0433c0f506a95e54d8401ed79d",
      "path": "bench/design_batch.py", "sha256": "5a5df91b50c39e8c045aacb6984fcfb6c0cae3abd41b8d4c62d2611d993f9130"}
# demo/data/grid.json (Atlas, #5): 48 design points, each 10,000 calibration + 10,000 null + 1,000 planted studies.
GRID = {"points": 48, "calibration": 10000, "null": 10000, "alternative": 1000, "source": "demo/data/grid.json"}


LABELS = {"compute": "MEASURED", "outcomes": "SYNTHETIC (atlas/simulation.py at 79a7792)",
          "rates": "ESTIMATE (FWER, power and calls: binomial rates with Wilson 95% intervals)"}
# the ceil((B+1)(1-alpha)) rank rule needs B >= 19 at alpha 0.05, else the threshold is +inf and nothing rejects
MIN_CALIBRATION = 19


def plans(lab_per_stream: int, calibration: int, per_stream: int) -> dict:
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (lab_per_stream, calibration, per_stream)):
        raise ValueError("study counts must be integers")
    if not 0 < lab_per_stream <= per_stream:
        raise ValueError("the lab sample must be a non-empty subset of the preflight arm's null and alternative studies")
    if calibration < MIN_CALIBRATION:
        raise ValueError(f"need at least {MIN_CALIBRATION} calibration studies for a finite threshold")
    return {"lab": {"null": lab_per_stream, "alternative": lab_per_stream},
            "preflight": {"calibration": calibration, "null": per_stream, "alternative": per_stream}}


def check_cap(spent: float, ceiling: float, cap: float) -> None:
    """Refuse a NaN, infinite or non-positive cap (NaN compares false, so it would pass), and any run over it."""
    if not (math.isfinite(cap) and cap > 0 and math.isfinite(spent) and math.isfinite(ceiling)):
        raise ValueError(f"cap, spend and ceiling must be finite and the cap positive: {cap}, {spent}, {ceiling}")
    if spent + ceiling > cap:
        raise ValueError(f"refused: spent {spent} + ceiling {ceiling} > cap {cap} USD")


def _coverage(records: list, plan: dict, arm: str) -> None:
    """Every planned (stream, i) exactly once and nothing else: duplicates or gaps are not a matched run."""
    got = [(r["stream"], r["i"]) for r in records]
    want = {(s, i) for s, n in plan.items() for i in range(n)}
    if len(got) != len(set(got)):
        raise ValueError(f"duplicate study IDs in the {arm} arm")
    if set(got) != want:
        raise ValueError(f"{arm} arm studies do not cover its plan exactly: "
                         f"missing {sorted(want - set(got))[:3]}, extra {sorted(set(got) - want)[:3]}")


def run(load, plan: dict, workers: int = 8) -> dict:
    """Preflight on `workers` cores, then the lab on one pinned core; fresh cells and harness state per arm."""
    db._S.clear()
    pre = db.run_arm("numba", load(), plan["preflight"], workers=workers)
    db._S.clear()  # else the lab would reuse the numba kernel cached in _S
    keep = os.sched_getaffinity(0)
    core = max(keep)
    os.sched_setaffinity(0, {core})  # the forked worker inherits it, as bench.lab_today pins itself
    try:
        lab = db.run_arm("lab", load(), plan["lab"], workers=1)
    finally:
        os.sched_setaffinity(0, keep)
    lab["affinity"] = [core]
    return {"lab": lab, "preflight": pre}


def compare(lab: dict, pre: dict, plan: dict) -> dict:
    """Per-study features and decisions on the shared studies; compute-through-scoring wall per study and ratio."""
    from stats import threshold

    if lab["workers"] != 1:
        raise ValueError("the lab arm must run one worker: one study at a time on one core")
    if set(plan["lab"]) != {"null", "alternative"} or any(plan["lab"][s] > plan["preflight"][s] for s in plan["lab"]):
        raise ValueError("the lab studies are not in the preflight arm's plan")
    _coverage(pre["records"], plan["preflight"], "preflight")
    P = {(r["stream"], r["i"]): r for r in pre["records"]}
    missing = [(r["stream"], r["i"]) for r in lab["records"] if (r["stream"], r["i"]) not in P]
    if missing:
        raise ValueError(f"lab studies not in the preflight arm: {missing[:3]}")
    _coverage(lab["records"], plan["lab"], "lab")
    cal = [r["z"] for r in pre["records"] if r["stream"] == "calibration"]
    cut = float(threshold({"alpha": db.ALPHA}, np.stack(cal)))
    rows = []
    for r in sorted(lab["records"], key=lambda r: (r["stream"], r["i"])):
        p = P[(r["stream"], r["i"])]
        mt = float(np.abs(p["z"]).max())
        lc, pc = r["p"] <= db.ALPHA, mt > cut
        rows.append({"stream": r["stream"], "i": r["i"], "x_sha": r["x_sha"], "x_equal": r["x_sha"] == p["x_sha"],
                     "lab_p": r["p"], "lab_call": lc, "preflight_maxT": mt, "preflight_call": bool(pc),
                     "agree": bool(lc == pc)})
    x_ok = all(r["x_equal"] for r in rows)
    calls = {s: {"lab": sum(r["lab_call"] for r in rows if r["stream"] == s),
                 "preflight": sum(r["preflight_call"] for r in rows if r["stream"] == s),
                 "n": sum(r["stream"] == s for r in rows)} for s in ("null", "alternative")}
    n_pre = len(pre["records"])
    decided = n_pre - len(cal)
    lab_s = lab["studies_wall_s"] / len(lab["records"])
    setup = lambda o: o["setup"]["prep_s"] + o["setup"]["adjacency_s"] + (o["warm_s"] or 0)
    wall = {"lab_s_per_study": lab_s,
            "preflight_s_per_evaluated_study": pre["studies_wall_s"] / n_pre,
            "preflight_s_per_decided_study": pre["studies_wall_s"] / decided,
            "ratio": lab_s / (pre["studies_wall_s"] / decided),
            "with_setup": {"lab_s_per_study": (lab["studies_wall_s"] + setup(lab)) / len(lab["records"]),
                           "preflight_s_per_decided_study": (pre["studies_wall_s"] + setup(pre)) / decided},
            "studies": {"lab": len(lab["records"]), "preflight_evaluated": n_pre, "preflight_decided": decided,
                        "preflight_calibration": len(cal)}}
    wall["with_setup"]["ratio"] = wall["with_setup"]["lab_s_per_study"] / wall["with_setup"]["preflight_s_per_decided_study"]
    return {"status": "MEASURED" if x_ok else "FAILED", "threshold": cut,
            "parity": {"x_bitwise": x_ok, "x_compared": len(rows)},
            "decisions": {"agree": sum(r["agree"] for r in rows), "compared": len(rows), "calls": calls},
            "wall": wall, "per_study": rows}


def project(wall: dict, grid: dict = GRID) -> dict:
    """ANALYTIC: every grid point priced at the measured point's per-study wall; the lab needs no calibration."""
    lab = grid["points"] * (grid["null"] + grid["alternative"]) * wall["lab_s_per_study"]
    pre = grid["points"] * (grid["calibration"] + grid["null"] + grid["alternative"]) * wall["preflight_s_per_evaluated_study"]
    return {"status": "ANALYTIC", "grid": grid, "lab_s": lab, "preflight_s": pre, "ratio": lab / pre,
            "lab_days": lab / 86400, "preflight_hours": pre / 3600,
            "assumes": "each design point costs what the measured point (bench.real_cpu.DESIGN) costs per study; "
                       "real points differ in regions, field of view and mis-call rate. A projection, not a run."}
