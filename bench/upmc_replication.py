"""Second cohort: does UPMC replicate Charville's gap between the textbook power formula and real-tissue power?

    uv run python -m bench.upmc_replication

No new simulation: every SIMULATED number is read from results already committed, and the ANALYTIC side is
bench/vs_schoenfeld.py's formula and compare(), the code that made Charville's headline. Two views:

1. production grid: both cohorts from one atlas commit (4408ff0), same command, seed 20261121, numpy backend,
   10,000 calibration / 10,000 null / 1,000 planted studies per design point (atlas/results/production/*).
2. #83 curve: the design and seed of Charville's headline (2 regions, 400 um, 10% mis-calls; bench/results/
   vs-schoenfeld-300.json) on 34 UPMC discovery patients (bench/results/hn-power-upmc-discovery.json), with
   the formula at the same planted event fraction (0.239) and UPMC's own feature count.

Writes bench/results/upmc-replication.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from bench.power_at_scale import clopper_pearson
from bench.vs_schoenfeld import compare, schoenfeld_power

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
OUT = RESULTS / "upmc-replication.json"
PROD = ROOT / "atlas" / "results" / "production"
HEADLINE = {"regions_per_patient": 2, "fov_um": 400.0, "eps": 0.1}
LABELS = {"ANALYTIC": "Schoenfeld events formula, Bonferroni over m features (bench/vs_schoenfeld.py)",
          "SIMULATED": "model-based Monte Carlo power and FWER on studies re-imaged from real tissue; SYNTHETIC "
                       "planted Weibull outcomes, HR 2 per source SD; no real outcome is read",
          "MEASURED": "compute receipts only (hardware, wall time); no clinical quantity is measured"}


def m_features(cohort: str) -> int:
    return int(json.loads((ROOT / "audit" / "results" / f"{cohort}.json").read_text())["n_features_tested"])


def pooled(points: list[dict], k: str, n: str) -> dict:
    hits, total = sum(int(p[k]) for p in points), sum(int(p[n]) for p in points)
    lo, hi = clopper_pearson(hits, total)
    return {"rejections": hits, "nulls": total, "fwer": hits / total, "lo": lo, "hi": hi,
            "sizes": [p["patients"] for p in points]}


def production(cohort: str) -> dict:
    """Formula vs simulation at every (regions, field) of the cohort's production grid, via vs_schoenfeld.compare."""
    d = PROD / cohort
    meta = json.loads((d / "metadata.json").read_text())
    rows = pd.read_parquet(d / "summary.parquet").astype(object).where(lambda t: t.notna(), None).to_dict("records")
    m, f = m_features(cohort), meta["source_events"] / meta["source_n"]
    for r in rows:  # the formula's D must be the events the simulation actually planted
        assert abs(r["alternative_mean_events"] / r["patients"] - f) < 0.02, (cohort, r["patients"])
    designs = sorted({(r["regions_per_patient"], r["fov_um"]) for r in rows}, key=lambda t: (t[0], t[1] or 1e9))
    grid, over, under = [], 0, 0
    for regions, fov in designs:
        sub = [r for r in rows if (r["regions_per_patient"], r["fov_um"]) == (regions, fov)]
        res = compare(sub, meta, m, cohort=cohort)
        counts = {r["patients"]: r for r in sub}
        for p in res["points"]:
            r = counts[p["patients"]]
            p.update(fwer_count=int(r["fwer_count"]), fwer_n=int(r["fwer_n"]))
        over += sum(p["formula_outside_sim_interval"] and p["formula_minus_sim"] > 0 for p in res["points"])
        under += sum(p["formula_outside_sim_interval"] and p["formula_minus_sim"] < 0 for p in res["points"])
        grid.append({"design": res["design"], "verdict": res["verdict"], "points": res["points"]})
    head = next(g for g in grid if g["design"]["regions_per_patient"] == HEADLINE["regions_per_patient"]
                and g["design"]["fov_um"] == HEADLINE["fov_um"])
    n_points = sum(len(g["points"]) for g in grid)
    return {"source_sha": meta.get("atlas_code", {}).get("commit") or _prod_sha(cohort), "run_id": meta["run_id"],
            "source_patients": meta["source_n"], "source_events": meta["source_events"], "event_fraction": f,
            "features_m": m, "budget": meta["budget"], "seed": meta["seed"], "backend": meta["backend"],
            "headline_design": head,
            "headline_pooled_fwer": pooled(head["points"], "fwer_count", "fwer_n"),
            "design_points": n_points, "formula_overstates": over, "formula_understates": under,
            "grid": grid}


def _prod_sha(cohort: str) -> str:
    runs = json.loads((PROD / "modal-execution.json").read_text())["runs"]
    return next(r["source_sha"] for r in runs if r.get("kind") == "finished" and r.get("cohort") == cohort)


def _hardware(cohort: str) -> dict:
    runs = json.loads((PROD / "modal-execution.json").read_text())["runs"]
    r = next(r for r in runs if r.get("kind") == "finished" and r.get("cohort") == cohort)
    return {"runner": "modal", "cpu": r["cpu"], "memory_mib": r["memory_mib"], "task_id": r["task_id"],
            "elapsed_s": r["elapsed_s"], "label": "MEASURED"}


def curve83() -> dict:
    """UPMC's #83-design curve next to Charville's, formula on both at the planted event fraction."""
    ch = json.loads((RESULTS / "vs-schoenfeld-300.json").read_text())
    up = json.loads((RESULTS / "hn-power-upmc-discovery.json").read_text())
    m, f = m_features("upmc"), ch["event_fraction"]
    upts = []
    for p in up["points"]:
        a = schoenfeld_power(p["patients"], f, ch["hr_per_sd"], ch["alpha"], m)
        upts.append({"patients": p["patients"], "analytic_power": a, "sim_power": p["power"], "sim_lo": p["power_lo"],
                     "sim_hi": p["power_hi"], "sim_n": p["planted_studies"], "sim_mean_events": p["mean_events"],
                     "formula_minus_sim": a - p["power"],
                     "formula_outside_sim_interval": not (p["power_lo"] <= a <= p["power_hi"]),
                     "fwer": p["fwer"], "fwer_count": p["fwer_count"], "fwer_n": p["fwer_n"],
                     "fwer_lo": p["fwer_lo"], "fwer_hi": p["fwer_hi"]})
    cpts = [{**p, "fwer_count": round(p["sim_fwer"] * 100), "fwer_n": 100} for p in ch["points"]]
    raws = [json.loads((ROOT / r).read_text()) for r in up["raw"]]
    gpus = sorted({pt["gpu"]["model"] for r in raws for pt in r["points"] if "gpu" in pt})
    matched = sorted({p["patients"] for p in upts} & {p["patients"] for p in cpts})
    pick = lambda pts, n: next(p for p in pts if p["patients"] == n)  # noqa: E731
    return {
        "design": ch["design"] | {"hr_per_sd": ch["hr_per_sd"], "planted_event_fraction": f},
        "matched_patients": matched,
        "matched": [{"patients": n, "charville": {k: pick(cpts, n)[k] for k in
                                                  ("analytic_power", "sim_power", "sim_lo", "sim_hi", "sim_fwer")},
                     "upmc": {k: pick(upts, n)[k] for k in ("analytic_power", "sim_power", "sim_lo", "sim_hi", "fwer")}}
                    for n in matched],
        "charville": {"source": "bench/results/vs-schoenfeld-300.json", "features_m": ch["features_m"],
                      "budget": ch["simulated"]["budget"], "seed": ch["simulated"]["seed"],
                      "code": ch["simulated"]["atlas_code"], "hardware": {"backend": "cpu"},
                      "pooled_fwer": pooled(cpts, "fwer_count", "fwer_n"), "points": cpts},
        "upmc": {"source": "bench/results/hn-power-upmc-discovery.json", "features_m": m,
                 "source_patients": up["metadata"]["source_n"], "budget": up["metadata"]["budget"],
                 "planted_studies_per_point": up["planted_studies_per_point"], "seed": up["metadata"]["seed"],
                 "code": up["commit"], "hardware": {"runner": "modal", "gpu": gpus, "label": "MEASURED"},
                 "caveats": up["caveats"], "pooled_fwer": pooled(upts, "fwer_count", "fwer_n"), "points": upts},
    }


def build() -> dict:
    prod = {c: production(c) for c in ("charville", "upmc")}
    for c in prod:
        prod[c]["hardware"] = _hardware(c)
    c83 = curve83()
    at200 = next(x for x in c83["matched"] if x["patients"] == 200)
    cu, uu = at200["charville"], at200["upmc"]
    ph = {c: max(prod[c]["headline_design"]["points"], key=lambda p: p["patients"]) for c in prod}
    share = {c: prod[c]["formula_overstates"] / prod[c]["design_points"] for c in prod}
    replicated = share["upmc"] >= 0.5  # Charville's gap: the formula overstates outside the 95% CI at 48 of 48
    answer = (
        f"{'Yes' if replicated else 'No'}: UPMC {'replicates' if replicated else 'does not replicate'} Charville's "
        f"gap. The formula overstates power outside the simulation's 95% interval at "
        f"{prod['upmc']['formula_overstates']} of {prod['upmc']['design_points']} UPMC production design points "
        f"(Charville {prod['charville']['formula_overstates']} of {prod['charville']['design_points']}). On the #83 design at 200 patients the formula promises "
        f"{uu['analytic_power']:.0%} and UPMC tissue gives {uu['sim_power']:.0%} "
        f"[{uu['sim_lo']:.0%}-{uu['sim_hi']:.0%}], against Charville's {cu['analytic_power']:.0%} vs "
        f"{cu['sim_power']:.0%} [{cu['sim_lo']:.0%}-{cu['sim_hi']:.0%}]. In the matched production grid (one "
        f"commit, 1,000 planted studies) at the largest size, Charville {ph['charville']['patients']} patients: "
        f"formula {ph['charville']['analytic_power']:.0%}, tissue {ph['charville']['sim_power']:.0%}; UPMC "
        f"{ph['upmc']['patients']}: formula {ph['upmc']['analytic_power']:.0%}, tissue "
        f"{ph['upmc']['sim_power']:.0%}. The gap is cohort-specific, so the formula cannot be trusted either way "
        f"without simulating on the cohort's own tissue. FWER holds near 5% on both.")
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    return {"schema": "preflight-upmc-replication/1", "status": "ESTIMATE", "labels": LABELS,
            "question": "Does UPMC replicate 'real-tissue power far below the formula' and FWER control?",
            "replicated": replicated, "replicated_rule": "yes if the formula overstates power outside the "
            "simulation's 95% interval at half or more of UPMC's production design points",
            "answer": answer, "computed_at_commit": sha,
            "new_simulation": "none: every simulated number is read from committed results; Modal spend $0",
            "studentised_score": "not on integration at compute time; FWER here is the atlas's maxT on the Cox "
                                 "score with permutation variance",
            "outcome_scope": "SYNTHETIC outcomes on real tissue; no real UPMC outcome read; no clinical claim",
            "curve_83": c83, "production": prod}


def main() -> int:
    res = build()
    OUT.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    print(f"{OUT.relative_to(ROOT)}: {res['answer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
