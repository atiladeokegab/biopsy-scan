"""How many patients for 80% power? Merge vs-schoenfeld-300 with the Modal GPU points into one answer.

    uv run python -I bench/power_at_scale.py --planned 400,500,600 bench/results/modal-power-a100-*.json

Writes bench/results/power-at-scale.json, status ESTIMATE. The points up to 300 patients come from
bench/results/vs-schoenfeld-300.json (same design, seed and budget, run on the CPU backend); the larger ones from
the raw Modal runs named on the command line (gpu/power_at_scale.py). n for 80% power is the first crossing of the
simulated power curve, interpolated linearly in patients between the two bracketing points. The same crossing of the
per-point 95% Clopper-Pearson bounds is reported as pointwise only: it is not a confidence interval for n.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
TARGET = 0.80
LABEL = "ESTIMATE: simulated"
PIN = "1f1bc7457eaf98e5cff600c0e17001d7e2fc44bf"  # gpu/power_at_scale.py's PIN: the code the 300 run used
# The model every shard must have run: plant, hazard shape, source normalisation and censoring, as the P=40 GPU
# replicate that reproduced #53 to every digit reported them. vs-schoenfeld-300.json does not record these per point.
FROZEN_MODEL = {
    "shape": 1.5, "plant_feature": 0, "plant_mean": 0.022642723251553813, "plant_sd": 0.021889184398124352,
    "target_event_fraction": 0.2389937106918239, "source_events": 38, "source_regions": 286, "source_cells": 619645,
    "max_regions_per_patient": 2,
    "censoring_tail_policy": "administrative horizon = stratum maximum observed follow-up",
    "administrative_horizons_json": '{"0": 1562.0, "1": 1675.0, "2": 5812.0, "3": 4821.0}',
    "error_control_scope": "unconditional empirical-source global null; separate maxT and omnibus tests",
}


def crossing(ns: list[int], ys: list[float], target: float = TARGET) -> float | None:
    """Patients where ys first reaches target, linear between bracketing points; None if it never does."""
    for i, (n, y) in enumerate(zip(ns, ys)):
        if y >= target:
            if i == 0:
                return float(n)
            n0, y0 = ns[i - 1], ys[i - 1]
            return n0 + (target - y0) * (n - n0) / (y - y0)
    return None


def answer(points: list[dict], missing: list[int] = ()) -> dict:
    ns = [p["patients"] for p in points]
    n80, lo, hi = (crossing(ns, [p[k] for p in points]) for k in ("power", "power_hi", "power_lo"))
    top = max(points, key=lambda p: p["patients"])
    cleared = [p["patients"] for p in points if p["power_lo"] >= TARGET]
    if n80 is None:
        statement = (f"Simulated power never reached 80% up to {top['patients']} patients: "
                     f"{top['power']:.0%} [{top['power_lo']:.0%}, {top['power_hi']:.0%}] there.")
    else:
        statement = (f"Simulated power first crosses 80% at about {n80:.0f} patients (interpolated). The pointwise "
                     f"95% CI bounds cross 80% at {lo:.0f} (upper) and "
                     + (f"{hi:.0f} (lower)" if hi is not None else f"not by {top['patients']} (lower)")
                     + "; these are pointwise intervals, not a confidence interval for n.")
    if missing:
        statement = (f"PARTIAL: P={', '.join(map(str, missing))} failed and has no result; among the points that "
                     "finished: " + statement.replace("Simulated power never reached", "simulated power did not reach"))
    statement += (f" Smallest simulated count whose pointwise lower 95% bound is at least 80%: {min(cleared)}."
                  if cleared else f" No simulated count up to {top['patients']} has its pointwise lower 95% bound "
                                  f"at 80% or above: beyond {top['patients']}.")
    return {"target_power": TARGET, "missing_patients": list(missing), "n_80": n80, "n_80_pointwise_ci_crossings": [lo, hi],
            "smallest_patients_pointwise_lower_bound_clears_80": min(cleared) if cleared else None,
            "max_patients_simulated": top["patients"], "statement": statement,
            "method": "n_80: first crossing of 80% by the simulated power, linear in patients between the bracketing "
                      "points, no extrapolation. n_80_pointwise_ci_crossings: the same crossing of the per-point 95% "
                      "Clopper-Pearson upper and lower bounds; pointwise, so not a confidence interval for n_80"}


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    """Two-sided 95% Clopper-Pearson, as the atlas computes it (atlas.runner.interval)."""
    from scipy.stats import beta

    return (0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)))


def check_point(r: dict, base: dict, source: str) -> dict:
    """Validate one producer point's own summary against the #53 design; return the curve row."""
    s, m, sim, budget = r["summary"], r["metadata"], base["simulated"], base["simulated"]["budget"]
    P = r["patients"]
    if s["patients"] != P:
        raise ValueError(f"point P={P}: its summary is for P={s['patients']}")
    if not isinstance(P, int) or P < 2:
        raise ValueError(f"P={P} is not a patient count")
    if s["seed"] != sim["seed"] or m["seed"] != sim["seed"]:
        raise ValueError(f"P={P}: seed {s['seed']}/{m['seed']} is not the 300 run's {sim['seed']}")
    if (s["source_hash"], m["source_hash"], s["grammar_hash"], m["grammar_hash"]) != (
            sim["source_hash"], sim["source_hash"], sim["grammar_hash"], sim["grammar_hash"]):
        raise ValueError(f"P={P} was not run on the 300 run's source and grammar")
    if {k: m["budget"][k] for k in budget} != budget:
        raise ValueError(f"P={P}: budget {m['budget']} is not the 300 run's")
    if (s["cohort"], s["alpha"], s["regions_per_patient"], s["fov_um"], s["eps"], s["hr_per_source_sd"]) != (
            base["cohort"], base["alpha"], base["design"]["regions_per_patient"], base["design"]["fov_um"],
            base["design"]["eps"], base["hr_per_sd"]):
        raise ValueError(f"P={P} is a different design")
    if {k: (type(s.get(k)), s.get(k)) for k in FROZEN_MODEL} != {k: (type(v), v) for k, v in FROZEN_MODEL.items()}:
        bad = sorted(k for k, v in FROZEN_MODEL.items() if (type(s.get(k)), s.get(k)) != (type(v), v))
        raise ValueError(f"P={P}: {', '.join(bad)} differ from the frozen model of the 300 run")
    t = s.get("threshold")
    if not (isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)):
        raise ValueError(f"P={P}: calibrated cutoff {t!r} is not a finite number")
    if (s["backend"], m["backend"]) != ("gpu", "gpu"):
        raise ValueError(f"P={P}: backend {s['backend']}/{m['backend']}, not gpu, so the GPU model is not its hardware")
    if (s["calibration_n"], s["fwer_n"], s["power_n"]) != (budget["calibration"], budget["null"], budget["alternative"]):
        raise ValueError(f"P={P}: study counts {s['calibration_n']}/{s['fwer_n']}/{s['power_n']} are not the budget's")
    for rate in ("power", "fwer"):
        k, n, v, lo, hi = (s[rate + x] for x in ("_count", "_n", "", "_lo", "_hi"))
        if not all(isinstance(c, int) or (isinstance(c, float) and c.is_integer()) for c in (k, n)):
            raise ValueError(f"P={P}: {rate} counts {k}/{n} are not integer binomial counts")
        k, n = int(k), int(n)
        want_lo, want_hi = clopper_pearson(k, n)
        if not (0 <= k <= n and math.isclose(v, k / n, abs_tol=1e-12) and math.isclose(lo, want_lo, abs_tol=1e-9)
                and math.isclose(hi, want_hi, abs_tol=1e-9) and 0 <= lo <= v <= hi <= 1):
            raise ValueError(f"P={P}: {rate} rates are not {k}/{n} with their 95% Clopper-Pearson interval")
    for t in (s["runtime_s"], r["in_function_s"]):
        if not (isinstance(t, (int, float)) and math.isfinite(t) and t > 0):
            raise ValueError(f"P={P}: timing {t} is not a positive finite number of seconds")
    if not (math.isfinite(s["alternative_mean_events"]) and 0 <= s["alternative_mean_events"] <= P):
        raise ValueError(f"P={P}: mean events {s['alternative_mean_events']} is not a finite number in [0, P]")
    if s["source_n"] != base["patients_source"] or not math.isclose(s["resample_ratio"], P / base["patients_source"]):
        raise ValueError(f"P={P}: resampling {s['resample_ratio']} of {s['source_n']} is not P / {base['patients_source']}")
    return {"patients": P, "power": s["power"], "power_count": s["power_count"], "power_lo": s["power_lo"],
            "power_hi": s["power_hi"], "power_n": s["power_n"], "fwer": s["fwer"], "fwer_lo": s["fwer_lo"],
            "fwer_hi": s["fwer_hi"], "mean_events": s["alternative_mean_events"], "resample_ratio": s["resample_ratio"],
            "backend": s["backend"], "hardware": r["gpu"]["model"], "wall_s": s["runtime_s"],
            "in_container_s": r["in_function_s"], "source": source}


def build(runs: list[dict], planned: list[int] | None = None) -> dict:
    base = json.loads((RESULTS / "vs-schoenfeld-300.json").read_text())
    points = [{"patients": p["patients"], "power": p["sim_power"], "power_lo": p["sim_lo"], "power_hi": p["sim_hi"],
               "power_n": p["sim_n"], "fwer": p["sim_fwer"], "fwer_lo": p["sim_fwer_lo"], "fwer_hi": p["sim_fwer_hi"],
               "mean_events": p["sim_mean_events"], "backend": "cpu", "hardware": "team VM CPU (not recorded per point)",
               "wall_s": None, "source": "bench/results/vs-schoenfeld-300.json"} for p in base["points"]]
    for p in points:
        p["power_count"] = round(p["power"] * p["power_n"])
    by_p = {p["patients"]: p for p in points}
    failed = []
    for run in sorted(runs, key=lambda r: r["_file"]):  # deterministic provenance whatever the argument order
        if (run.get("schema"), run.get("status"), run.get("pin"), run.get("backend")) != (
                "preflight-modal-power/1", "MEASURED", PIN, "gpu"):
            raise ValueError(f"{run['_file']} is not a MEASURED preflight-modal-power/1 GPU run of the pinned code {PIN[:12]}")
        for r in run["points"]:
            if "error" in r:
                failed.append({"patients": r["patients"], "error": r["error"], "source": f"bench/results/{run['_file']}"})
                continue
            point = check_point(r, base, f"bench/results/{run['_file']}")
            if r["patients"] in by_p:  # a replicate: same seeds, so it must give the same studies' answer
                kept = by_p[r["patients"]]
                science = ("power_count", "power_n", "fwer")
                same = (all(kept[k] == point[k] for k in science)
                        and math.isclose(kept["mean_events"], point["mean_events"], rel_tol=1e-12)
                        # #53's CPU points are compared on the science only; GPU replicates on the GPU model too
                        and (kept["backend"] == "cpu" or (kept["hardware"], kept["backend"]) == (point["hardware"], point["backend"])))
                if not same:
                    raise ValueError(f"P={r['patients']} twice with different results: {kept['source']} and {point['source']}")
                kept.setdefault("replicated_by", []).append(point["source"])
                continue
            by_p[r["patients"]] = point
    points = list(by_p.values())
    points.sort(key=lambda p: p["patients"])
    missing = sorted(({f["patients"] for f in failed} | set(planned or ())) - set(by_p))
    if len({p["hardware"] for p in points if p["backend"] == "gpu"}) > 1:
        raise ValueError("the GPU points ran on more than one GPU model; pin one for the whole curve")
    return {"schema": "preflight-power-at-scale/1", "status": "ESTIMATE", "label": LABEL,
            "question": "How many patients does this design need for 80% power?",
            "planned_patients": planned, "complete": not missing, "failed_points": failed,
            "complete_scope": "complete means every planned and every attempted count has a result; with no planned "
                              "list it covers only the counts in the supplied run files",
            "answer": answer(points, missing),
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
    planned = None
    if argv[:1] == ["--planned"]:
        planned, argv = [int(p) for p in argv[1].split(",")], argv[2:]
    runs = []
    for name in argv:
        run = json.loads(Path(name).read_text())
        run["_file"] = Path(name).name
        runs.append(run)
    result = build(runs, planned)
    out = RESULTS / "power-at-scale.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(out, result["answer"]["statement"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
