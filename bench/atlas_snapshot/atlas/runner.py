"""Independent design calibration, held-out validation and study-level evidence."""

import hashlib
import json

import numpy as np
from scipy.stats import beta

import stats
from stats.reference import _components

from .artifacts import source_metadata
from .simulation import fit_model, outcomes

DEFAULT_BUDGET = {"calibration": 10000, "null": 10000, "alternative": 1000,
                  "nested": 20, "permutations": 999, "batch": 8}


def interval(successes, total):
    """Two-sided 95% Clopper-Pearson interval, including zero/all successes."""
    if not 0 <= successes <= total or total < 1:
        raise ValueError("binomial counts must satisfy 0 <= successes <= total, total > 0")
    lo = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1))
    hi = 1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes))
    return lo, hi


def _tails(reference, queries):
    p = np.empty_like(queries, dtype=float)
    for j in range(queries.shape[1]):
        tolerance = 16 * np.finfo(float).eps * np.maximum(1, np.abs(queries[:, j]))
        p[:, j] = (1 + len(reference) - np.searchsorted(np.sort(reference[:, j]),
                                                       queries[:, j] - tolerance)) / (len(reference) + 1)
    return p


def fit_omnibus(z):
    """Reusable joint selection rule using disjoint marginal/joint null splits.

    Reuse the statistics reference's component implementation. Fit three marginal
    tail transforms on the first split, then calibrate their minimum jointly on
    the second. A future null study is exchangeable with the joint split given
    the marginal split; no independence among the three components is assumed.
    """
    z = np.asarray(z, float)
    if z.ndim != 2 or z.shape[1] < 1 or not np.isfinite(z).all() or len(z) < 4:
        raise ValueError("joint omnibus calibration needs at least four null studies")
    components = _components(z)
    middle = len(z) // 2
    marginal = components[:middle]
    return {"marginal": marginal, "joint_min_p": _tails(marginal, components[middle:]).min(axis=1)}


def omnibus_p(calibration, z):
    minimum = _tails(calibration["marginal"], _components(np.atleast_2d(z))).min(axis=1)
    joint = np.sort(calibration["joint_min_p"])
    return (1 + np.searchsorted(joint, minimum, side="right")) / (len(joint) + 1)


def design_id(design):
    scientific = {key: design.get(key) for key in
                  ("cohort", "patients", "regions_per_patient", "fov_um", "eps", "alpha")}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()[:16]


def _seeds(seed, point, stage, index):
    sequence = np.random.SeedSequence([seed, int(point[:8], 16), stage, index])
    return [int(child.generate_state(1, dtype=np.uint64)[0]) for child in sequence.spawn(4)]


def studies(cells, model, design, count, seed, stage, batch_size):
    """Yield fresh noisy studies with outcomes aligned to core's patient draws."""
    from reimage import run

    n = design["patients"]
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 2:
        raise ValueError("a design needs an integer patient count >= 2")
    point = design_id(design)
    for start in range(0, count, batch_size):
        seeds = [_seeds(seed, point, stage, i) for i in range(start, min(start + batch_size, count))]
        batch = run({**design, "cells": cells}, np.array([s[0] for s in seeds], dtype=np.uint64))
        if batch.ndim != 3 or batch.shape[:2] != (len(seeds), n):
            raise ValueError("reimage.run returned an incompatible study batch")
        for offset, (X, streams) in enumerate(zip(batch, seeds)):
            image_seed, survival_seed, censor_seed, permutation_seed = streams
            # This is the first draw in core.reimage_ref.draw for an integer n.
            patients = np.random.default_rng(image_seed).integers(0, len(model["source_z"]), n)
            time, event, strata = outcomes(model, patients, survival_seed, censor_seed)
            z = stats.score(X, time, event, strata)["z"]
            yield {"X": X, "time": time, "event": event, "strata": strata, "z": z,
                   "index": start + offset, "image_seed": image_seed,
                   "survival_seed": survival_seed, "censor_seed": censor_seed,
                   "permutation_seed": permutation_seed, "events": int(event.sum())}


def _record(study, point, stage):
    absolute = np.abs(study["z"])
    top = int(np.argmax(absolute))
    return {"design_id": point, "stage": stage, "study": study["index"],
            **{k: study[k] for k in ("image_seed", "survival_seed", "censor_seed", "permutation_seed", "events")},
            "maxT": float(absolute[top]), "top_feature": top,
            "top_tied": bool(np.sum(np.isclose(absolute, absolute[top], rtol=1e-12, atol=1e-12)) > 1)}


def calibrate(cells, reference, design, budget, seed):
    model = fit_model(cells, reference, hr=1)
    point, rows, records = design_id(design), [], []
    for study in studies(cells, model, design, budget["calibration"], seed, 101, budget["batch"]):
        rows.append(study["z"])
        records.append(_record(study, point, "calibration"))
    z = np.array(rows)
    return {"design_id": point, "threshold": stats.threshold(design, z),
            "omnibus": fit_omnibus(z), "records": records, "n": len(z)}


def _add_rate(summary, name, flags):
    flags = np.asarray(flags, dtype=bool)
    count, total = int(flags.sum()), len(flags)
    lo, hi = interval(count, total)
    summary.update({name: count / total, name + "_count": count,
                    name + "_n": total, name + "_lo": lo, name + "_hi": hi})


def run_point(cells, reference, design, *, hr=2.0, budget=None, seed=0, calibration=None):
    """Calibrate once, then evaluate disjoint null and planted-effect studies.

    maxT and omnibus are separate procedures at alpha; their union is not a
    familywise test. Identification means a unique top feature equals the plant
    and maxT rejects. Off-target correlated top hits count as identification
    errors, not established statistical false positives. Ties count separately.
    """
    budget = {**DEFAULT_BUDGET, **(budget or {})}
    if any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_))
           or n < (0 if key == "nested" else 1) for key, n in budget.items()):
        raise ValueError("Monte Carlo budgets must be positive integers; nested may be zero")
    if budget["calibration"] < 4:
        raise ValueError("at least four calibration studies are required")
    model = fit_model(cells, reference, hr=hr)
    point = design_id(design)
    calibration = calibration or calibrate(cells, reference, design, budget, seed)
    if calibration["design_id"] != point or calibration["n"] != budget["calibration"]:
        raise ValueError("calibration does not match the design/budget")
    alpha = float(design.get("alpha", 0.05))
    summary = {"design_id": point, **source_metadata(cells), **{k: design.get(k) for k in
               ("cohort", "patients", "regions_per_patient", "fov_um", "eps", "backend")},
               "alpha": alpha, "hr_per_source_sd": hr, "shape": model["shape"],
               "plant_feature": model["feature"], "plant_mean": model["source_mean"],
               "plant_sd": model["source_sd"], "target_event_fraction": model["target_event_fraction"],
               "expected_event_fraction": model["expected_event_fraction"],
               "calibration_n": calibration["n"], "threshold": calibration["threshold"],
               "seed": seed, "censoring_tail_policy": model["censoring_tail_policy"],
               "administrative_horizons_json": json.dumps({str(k): d["horizon"] for k, d in model["censoring"].items()}),
               "nested_permutations": budget["permutations"],
               "error_control_scope": "unconditional empirical-source global null; separate maxT and omnibus tests"}
    records = list(calibration["records"])
    for stage, stage_id, count, stage_model in (
        ("null", 202, budget["null"], fit_model(cells, reference, hr=1)),
        ("alternative", 303, budget["alternative"], model),
    ):
        stage_records, z_rows, nested = [], [], []
        for study in studies(cells, stage_model, design, count, seed, stage_id, budget["batch"]):
            row = _record(study, point, stage)
            row["maxT_reject"] = row["maxT"] > calibration["threshold"]
            if study["index"] < budget["nested"]:
                conditional = stats.permutation_test(study["X"], study["time"], study["event"], study["strata"],
                                                     n_resamples=budget["permutations"], seed=study["permutation_seed"])
                row["conditional_p"] = conditional["p"]
                row["conditional_reject"] = conditional["p"] <= alpha
                nested.append(row)
            stage_records.append(row)
            z_rows.append(study["z"])
        omnibus = omnibus_p(calibration["omnibus"], np.array(z_rows))
        for row, p in zip(stage_records, omnibus):
            row["omnibus_p"], row["omnibus_reject"] = float(p), bool(p <= alpha)
        flags = [row["maxT_reject"] for row in stage_records]
        _add_rate(summary, "fwer" if stage == "null" else "power", flags)
        _add_rate(summary, "omnibus_fwer" if stage == "null" else "omnibus_power", omnibus <= alpha)
        summary[stage + "_mean_events"] = float(np.mean([row["events"] for row in stage_records]))
        if stage == "alternative":
            _add_rate(summary, "identification_rate", [r["maxT_reject"] and not r["top_tied"]
                                                       and r["top_feature"] == model["feature"] for r in stage_records])
            _add_rate(summary, "wrong_identification_rate", [r["maxT_reject"] and not r["top_tied"]
                                                             and r["top_feature"] != model["feature"] for r in stage_records])
            _add_rate(summary, "discovery_tie_rate", [r["maxT_reject"] and r["top_tied"] for r in stage_records])
        if nested:
            delta = np.array([int(r["maxT_reject"]) - int(r["conditional_reject"]) for r in nested])
            _add_rate(summary, stage + "_nested_disagreement", delta != 0)
            summary[stage + "_nested_shared_minus_conditional"] = float(delta.mean())
            summary[stage + "_nested_difference_se"] = float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None
        records.extend(stage_records)
    return summary, records, calibration
