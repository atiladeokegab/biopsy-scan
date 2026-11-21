"""The simulator's predicted power checked against the real Charville cohort, on real tissue.

    DATA_DIR=... uv run python -I bench/real_check.py [--cache data/atlas-cache] [--calibration 2000] [--alternative 2000]

Both arms bootstrap n patients (with replacement) from the real 159, give each its real whole-tissue features
(the 159 x 630 reference matrix, v1 grammar), plant HR 2 per SD on the simulator's own plant feature, and run the
atlas screen: maxT over stats.score z, rejecting above a threshold calibrated from null studies at the same n.
Study i uses the same patients and seeds in both arms, so the comparison is paired.

MEASURED (sim): outcomes from the simulator, atlas.simulation (Weibull k=1.5 tuned to the cohort's event fraction,
reverse-KM censoring). At the full-tissue design re-imaging is the identity, so this is what run_point predicts there.
MEASURED (plasmode): the same, except that the baseline survival is the real cohort's own Kaplan-Meier curve per
stratum, S_i(t) = S0(t) ** HR**z_i, instead of a fitted Weibull. Censoring is the same real reverse KM.

Real data exists only at the full-tissue design (all regions, full core, no mis-calls), so this checks the outcome
model and calibrated-threshold power, not the imaging-noise model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stats  # noqa: E402
from atlas.runner import interval, paired_interval  # noqa: E402

PATIENTS = (50, 100, 159)
SCOPE = ("Real data exists only at the full-tissue design (all regions, full core, no mis-calls), where re-imaging is "
         "the identity; this checks the simulator's outcome model and calibrated-threshold power against the real "
         "cohort's own survival curve, not the imaging-noise model.")


def kaplan_meier(time, event):
    """Event times and the KM survival just after each (failures before censors at tied times)."""
    time, event = np.asarray(time, float), np.asarray(event)
    at, surv, s = [], [], 1.0
    for t in np.unique(time[event == 1]):
        s *= 1 - np.sum((time == t) & (event == 1)) / np.sum(time >= t)
        at.append(float(t))
        surv.append(s)
    return np.array(at), np.array(surv)


def km_outcomes(source, z, hr, patients, censoring, survival_seed, censor_seed):
    """Planted outcomes on the real KM baseline: P(T_i > t) = S0_stratum(t) ** hr**z_i; T is +inf past the curve.

    censoring is atlas.simulation.fit_model's per-stratum reverse KM ({"time", "mass"}); a tie T == C is an event.
    """
    patients = np.asarray(patients)
    strata = np.asarray(source["strata"])[patients]
    u = np.random.default_rng(survival_seed).random(len(patients))
    rng_c = np.random.default_rng(censor_seed)
    T, C = np.full(len(patients), np.inf), np.empty(len(patients))
    for s in np.unique(strata):
        ix = strata == s
        own = np.asarray(source["strata"]) == s
        at, surv = kaplan_meier(np.asarray(source["time"])[own], np.asarray(source["event"])[own])
        if not len(at) or s not in censoring:
            raise ValueError(f"stratum {s} needs source events and a censoring distribution")
        # first event time where S0 ** r <= u, i.e. S0 <= u ** (1 / r); S0 is non-increasing
        target = u[ix] ** (1 / hr ** np.asarray(z, float)[patients[ix]])
        j = np.searchsorted(-surv, -target, side="left")
        T[ix] = np.where(j < len(at), at[np.minimum(j, len(at) - 1)], np.inf)
        C[ix] = rng_c.choice(censoring[s]["time"], size=int(ix.sum()), p=censoring[s]["mass"])
    event = T <= C
    return np.minimum(T, C), event, strata.copy()


def _seeds(seed, n, stage, i):
    children = np.random.SeedSequence([seed, n, stage, i]).spawn(3)
    return [int(c.generate_state(1, dtype=np.uint64)[0]) for c in children]


def run_arm(X, null_outcomes, alt_outcomes, n, *, calibration, alternative, seed, alpha=0.05):
    """Calibrate maxT on null studies, then count planted-effect rejections. outcomes(patients, s_seed, c_seed)."""
    rows, rejects, events, digest = [], [], [], hashlib.sha256()
    for stage, count, make in ((1, calibration, null_outcomes), (2, alternative, alt_outcomes)):
        for i in range(count):
            p_seed, s_seed, c_seed = _seeds(seed, n, stage, i)
            patients = np.random.default_rng(p_seed).integers(0, len(X), n)
            time_, event, strata = make(patients, s_seed, c_seed)
            z = stats.score(X[patients], time_, event, strata)["z"]
            if stage == 1:
                rows.append(z)
                continue
            if i == 0:
                cut = stats.threshold({"alpha": alpha}, np.array(rows))
            digest.update(patients.astype(np.int64).tobytes())
            rejects.append(bool(stats.maxT(z) > cut))
            events.append(int(event.sum()))
    k = int(sum(rejects))
    lo, hi = interval(k, len(rejects))
    return {"patients": n, "calibration_n": calibration, "threshold": float(cut), "power": k / len(rejects),
            "power_count": k, "power_n": len(rejects), "power_lo": lo, "power_hi": hi,
            "mean_events": float(np.mean(events)), "patients_sha256": digest.hexdigest(), "rejects": rejects}


def compare(sim, emp):
    """Paired difference sim - empirical in rejection rate, with atlas.runner's conservative >=95% bounds."""
    if sim["patients_sha256"] != emp["patients_sha256"] or sim["power_n"] != emp["power_n"]:
        raise ValueError("arms must be paired: the same patient draws in the same order")
    delta = np.array(sim["rejects"], int) - np.array(emp["rejects"], int)
    lo, hi = paired_interval(delta)
    return {"difference": sim["power"] - emp["power"], "difference_lo": lo, "difference_hi": hi,
            "agree": bool(lo <= 0 <= hi), "discordant": int(np.sum(delta != 0)),
            "patients_sha256": sim["patients_sha256"]}


def identity_check(cells, X, seeds, n=3):
    """Re-imaging at the full-tissue design returns the reference rows of the drawn patients (NaN where NaN)."""
    from reimage import run

    design = {"cells": cells, "patients": n, "regions_per_patient": None, "fov_um": None, "eps": 0.0}
    batch = run(design, np.asarray(seeds, np.uint64))
    worst, equal = 0.0, True
    for study, s in zip(batch, seeds):
        ref = X[np.random.default_rng(int(s)).integers(0, len(X), n)]
        equal &= bool(np.array_equal(np.isnan(study), np.isnan(ref)))
        both = ~np.isnan(ref)
        worst = max(worst, float(np.max(np.abs(study[both] - ref[both]), initial=0.0)))
    return {"studies": len(seeds), "patients_per_study": n, "max_abs_difference": worst,
            "equal": bool(equal and worst <= 1e-9)}


def main(argv=None):
    from atlas.__main__ import reference_matrix
    from atlas.artifacts import source_metadata
    from atlas.simulation import fit_model, outcomes
    from core.cohorts import load

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "atlas-cache")
    ap.add_argument("--calibration", type=int, default=2000)
    ap.add_argument("--alternative", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results" / "real-check-charville.json")
    a = ap.parse_args(argv)
    start = time.perf_counter()
    cells = load("charville")
    meta = source_metadata(cells)
    X, grammar = reference_matrix(cells, a.cache / "charville-reference.npz", meta)
    source = {k: np.asarray(cells[k]) for k in ("time", "event", "strata")}
    alt, null = fit_model(cells, X, hr=2.0), fit_model(cells, X, hr=1.0)
    z = alt["source_z"]
    weibull = lambda m: lambda p, s, c: outcomes(m, p, s, c)  # noqa: E731
    km = lambda hr: lambda p, s, c: km_outcomes(source, z, hr, p, alt["censoring"], s, c)  # noqa: E731
    points = []
    for n in PATIENTS:
        budget = {"calibration": a.calibration, "alternative": a.alternative, "seed": a.seed}
        sim = run_arm(X, weibull(null), weibull(alt), n, **budget)
        emp = run_arm(X, km(1.0), km(2.0), n, **budget)
        points.append({"patients": n, **compare(sim, emp),
                       **{k: {key: v for key, v in arm.items() if key not in ("rejects", "patients")}
                          for k, arm in (("simulator", sim), ("empirical", emp))}})
        print(json.dumps({"patients": n, "sim": sim["power"], "empirical": emp["power"]}), flush=True)
    res = {
        "schema": "preflight-real-check/1", "cohort": "charville", "status": "MEASURED",
        "source": {"patients": len(X), "events": int(source["event"].sum()), "strata": int(len(np.unique(source["strata"]))),
                   "features_m": int(X.shape[1]), "source_hash": meta["source_hash"], "grammar_hash": grammar},
        "design": {"regions_per_patient": "all", "fov_um": "full core", "eps": 0.0},
        "hr_per_sd": 2.0, "alpha": 0.05, "plant_feature": alt["feature"], "seed": a.seed,
        "screen": "maxT over stats.score z (legacy R4 Cox U, permutation variance), rejecting above the "
                  "ceil((B+1)(1-alpha)) null order statistic (stats.threshold) calibrated at the same n and arm",
        "arms": {"simulator": {"label": "MEASURED (sim)",
                               "outcomes": "atlas.simulation: Weibull k=1.5 PH, scale tuned to the cohort event "
                                           "fraction, stratum reverse-KM censoring at the administrative horizon"},
                 "empirical": {"label": "MEASURED (plasmode)",
                               "outcomes": "the real cohort's stratum Kaplan-Meier baseline raised to HR**z (planted "
                                           "PH), T past the curve censored; the same stratum reverse-KM censoring"}},
        "bootstrap": "patients drawn with replacement from the real 159; study i shares patients and seeds across arms",
        "comparison": "difference = sim - plasmode power; paired conservative >=95% bounds (atlas.runner.paired_interval)",
        "scope": SCOPE, "points": points,
        "identity_check": identity_check(cells, X, [a.seed + i for i in range(2)]),
        "runtime_s": round(time.perf_counter() - start, 1),
    }
    a.out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    print(a.out)
    return res


if __name__ == "__main__":
    main()
