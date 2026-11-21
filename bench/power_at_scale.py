"""How many patients for 80% power? Merge vs-schoenfeld-300 with the Modal GPU points into one answer.

    uv run python -I bench/power_at_scale.py bench/results/modal-power-a100-*.json

Writes bench/results/power-at-scale.json, status ESTIMATE. The points up to 300 patients come from
bench/results/vs-schoenfeld-300.json (same design, seed and budget, run on the CPU backend); the larger ones from
the raw Modal runs named on the command line (gpu/power_at_scale.py). n for 80% power is the first crossing of the
simulated power curve, interpolated linearly in patients between the two bracketing points; its band is the same
crossing of the per-point 95% Clopper-Pearson bounds (upper bound: fewest patients, lower bound: most).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
TARGET = 0.80
LABEL = "ESTIMATE: simulated"


def crossing(ns: list[int], ys: list[float], target: float = TARGET) -> float | None:
    """Patients where ys first reaches target, linear between bracketing points; None if it never does."""
    for i, (n, y) in enumerate(zip(ns, ys)):
        if y >= target:
            if i == 0:
                return float(n)
            n0, y0 = ns[i - 1], ys[i - 1]
            return n0 + (target - y0) * (n - n0) / (y - y0)
    return None


def answer(points: list[dict]) -> dict:
    ns = [p["patients"] for p in points]
    n80, lo, hi = (crossing(ns, [p[k] for p in points]) for k in ("power", "power_hi", "power_lo"))
    top = max(points, key=lambda p: p["patients"])
    cleared = [p["patients"] for p in points if p["power_lo"] >= TARGET]
    clear = f" Smallest simulated count whose lower 95% bound clears 80%: {min(cleared)}." if cleared else (
        f" No simulated count up to {top['patients']} has its lower 95% bound above 80% (beyond {top['patients']}).")
    if n80 is None:
        statement = (f"Simulated power never reached 80% up to {top['patients']} patients: "
                     f"{top['power']:.0%} [{top['power_lo']:.0%}, {top['power_hi']:.0%}] there.")
    else:
        band = f"{lo:.0f} to {hi:.0f}" if hi is not None else f"{lo:.0f} to more than {top['patients']}"
        statement = f"About {n80:.0f} patients for 80% simulated power (95% band {band})."
    statement += clear
    return {"target_power": TARGET, "n_80": n80, "n_80_lo": lo, "n_80_hi": hi,
            "n_80_hi_beyond_range": hi is None,
            "smallest_patients_lower_bound_clears_80": min(cleared) if cleared else None,
            "max_patients_simulated": top["patients"], "statement": statement,
            "method": "first crossing of 80%, linear in patients between bracketing points; band: the same crossing "
                      "of the per-point 95% Clopper-Pearson upper bound (lo) and lower bound (hi)"}


def build(runs: list[dict]) -> dict:
    base = json.loads((RESULTS / "vs-schoenfeld-300.json").read_text())
    points = [{"patients": p["patients"], "power": p["sim_power"], "power_lo": p["sim_lo"], "power_hi": p["sim_hi"],
               "power_n": p["sim_n"], "fwer": p["sim_fwer"], "fwer_lo": p["sim_fwer_lo"], "fwer_hi": p["sim_fwer_hi"],
               "mean_events": p["sim_mean_events"], "backend": "cpu", "hardware": "team VM CPU (not recorded per point)",
               "wall_s": None, "source": "bench/results/vs-schoenfeld-300.json"} for p in base["points"]]
    seen = {p["patients"] for p in points}
    for run in runs:
        for r in run["points"]:
            if "error" in r or r["patients"] in seen:
                continue
            s, m = r["summary"], r["metadata"]
            if (m["seed"] != base["simulated"]["seed"] or m["source_hash"] != base["simulated"]["source_hash"]
                    or m["grammar_hash"] != base["simulated"]["grammar_hash"]
                    or {k: m["budget"][k] for k in base["simulated"]["budget"]} != base["simulated"]["budget"]):
                raise ValueError(f"P={r['patients']} was not run on the 300 run's seed, source, grammar and budget")
            if (s["regions_per_patient"], s["fov_um"], s["eps"], s["hr_per_source_sd"]) != (
                    base["design"]["regions_per_patient"], base["design"]["fov_um"], base["design"]["eps"], base["hr_per_sd"]):
                raise ValueError(f"P={r['patients']} is a different design")
            points.append({"patients": r["patients"], "power": s["power"], "power_lo": s["power_lo"],
                           "power_hi": s["power_hi"], "power_n": s["power_n"], "fwer": s["fwer"], "fwer_lo": s["fwer_lo"],
                           "fwer_hi": s["fwer_hi"], "mean_events": s["alternative_mean_events"],
                           "resample_ratio": s["resample_ratio"], "backend": s["backend"],
                           "hardware": r["gpu"]["model"], "wall_s": s["runtime_s"],
                           "in_container_s": r["in_function_s"], "source": f"bench/results/{run['_file']}"})
            seen.add(r["patients"])
    points.sort(key=lambda p: p["patients"])
    if len({p["hardware"] for p in points if p["backend"] == "gpu"}) > 1:
        raise ValueError("the GPU points ran on more than one GPU model; pin one for the whole curve")
    return {"schema": "preflight-power-at-scale/1", "status": "ESTIMATE", "label": LABEL,
            "question": "How many patients does this design need for 80% power?",
            "answer": answer(points),
            "cohort": base["cohort"], "hr_per_sd": base["hr_per_sd"], "alpha": base["alpha"],
            "design": {k: v for k, v in base["design"].items() if k != "backend"},
            "patients_source": base["patients_source"], "events_source": base["events"],
            "resampling": "patients drawn with replacement from the source cohort; every count above "
                          f"{base['patients_source']} repeats source patients (resample_ratio = patients / source)",
            "simulated": {"method": base["simulated"]["method"], "budget": base["simulated"]["budget"],
                          "seed": base["simulated"]["seed"],
                          "seed_scheme": "SeedSequence([seed, design_id, stage, index]); design_id leaves out the "
                                         "backend, so CPU and GPU runs draw the same studies",
                          "source_hash": base["simulated"]["source_hash"],
                          "grammar_hash": base["simulated"]["grammar_hash"], "atlas_code": base["simulated"]["atlas_code"],
                          "test": "maxT against one threshold calibrated per design point on the calibration studies; "
                                  "nested 0, so no per-study permutations run"},
            "wall_s_note": "wall_s is the atlas's own runtime_s for the whole point (calibration, null and planted "
                           "studies), inside the container; container start-up and image build are excluded",
            "points": points}


def main(argv: list[str]) -> int:
    runs = []
    for name in argv:
        run = json.loads(Path(name).read_text())
        run["_file"] = Path(name).name
        runs.append(run)
    result = build(runs)
    out = RESULTS / "power-at-scale.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(out, result["answer"]["statement"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
