"""Contract-based real-data audit runner for UPMC and Charville (Issue #6, aligned to binding CONTRACT.md)."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __name__ == "__main__" and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.leakage import leakage_from_cells
from audit.schema import (
    ALREADY_SEEN_DISCLOSURE,
    AUDIT_SCHEMA_VERSION,
    choose_prepared_outcome,
    validate_audit_result,
)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class CoreAPI:
    load: Callable[..., dict[str, Any]]
    truth: Callable[[dict[str, Any]], np.ndarray]
    grammar: Callable[[list[str]], dict[str, Any]]
    n_features: Callable[[dict[str, Any]], int]
    grammar_hash: Callable[[dict[str, Any]], str]
    feature_names: Callable[[dict[str, Any]], list[str]]
    load_regions: Callable[..., Any] | None = None
    load_patients: Callable[..., Any] | None = None


@dataclass(frozen=True)
class StatsAPI:
    score: Callable[..., dict[str, np.ndarray]]
    permutation_test: Callable[..., dict[str, Any]]
    maxT: Callable[[np.ndarray], Any]
    threshold: Callable[..., float] | None = None


def conditional_permutation_cutoff(null_z: np.ndarray, alpha: float = 0.05) -> float:
    """Compute the conditional within-stratum maxT cutoff from permutation draws (not design-calibrated)."""
    arr = np.asarray(null_z, dtype=np.float64)
    if arr.ndim != 2 or len(arr) < 1:
        raise ValueError("null_z must be a non-empty 2D array")
    maxima = np.max(np.where(np.isfinite(arr), np.abs(arr), 0.0), axis=1)
    k = math.ceil((len(maxima) + 1) * (1.0 - alpha))
    return float(np.partition(maxima, k - 1)[k - 1]) if k <= len(maxima) else float("inf")


def load_default_core() -> CoreAPI:
    from core import cohorts
    from core.grammar import feature_names, grammar, grammar_hash, n_features
    from core.truth import truth

    return CoreAPI(
        load=cohorts.load,
        truth=truth,
        grammar=grammar,
        n_features=n_features,
        grammar_hash=grammar_hash,
        feature_names=feature_names,
        load_regions=getattr(cohorts, "load_regions", None),
        load_patients=getattr(cohorts, "load_patients", None),
    )


def load_default_stats() -> StatsAPI:
    import stats
    from stats import reference

    return StatsAPI(
        score=getattr(stats, "score", reference.score),
        permutation_test=getattr(stats, "permutation_test", reference.permutation_test),
        threshold=getattr(stats, "threshold", reference.threshold),
        maxT=getattr(stats, "maxT", reference.maxT),
    )


def _finite_feature_subset(X: np.ndarray, fnames: list[str]) -> tuple[np.ndarray, list[str], list[int], str]:
    finite_cols = np.isfinite(X).all(axis=0)
    if finite_cols.all():
        return X, fnames, list(range(len(fnames))), "all_finite"
    kept_indices = [int(i) for i, ok in enumerate(finite_cols) if ok]
    kept_names = [fnames[i] for i in kept_indices]
    return X[:, finite_cols], kept_names, kept_indices, "fallback_dropped_nonfinite_cols"


def leave_one_patient_out(
    *,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    strata: np.ndarray,
    patient_names: list[str],
    observed_z: np.ndarray,
    top_indices: list[int],
    score_fn: Callable[..., dict[str, np.ndarray]],
) -> dict[int, dict[str, Any]]:
    """Measure leave-one-patient-out z stability and rank-1 retention for top features."""
    n_patients = len(patient_names)
    overall_top = int(np.argmax(np.abs(observed_z))) if len(observed_z) else -1
    z_traces: dict[int, list[float]] = {idx: [] for idx in top_indices}
    rank1_matches = 0

    for drop_i in range(n_patients):
        keep = np.ones(n_patients, dtype=bool)
        keep[drop_i] = False
        z_sub = np.asarray(score_fn(X[keep], time[keep], event[keep], strata[keep])["z"], dtype=np.float64)
        finite_abs = np.where(np.isfinite(z_sub), np.abs(z_sub), -1.0)
        if len(finite_abs) and int(np.argmax(finite_abs)) == overall_top:
            rank1_matches += 1
        for idx in top_indices:
            z_traces[idx].append(float(z_sub[idx]))

    summary: dict[int, dict[str, Any]] = {}
    for idx in top_indices:
        vals = np.asarray(z_traces[idx], dtype=np.float64)
        obs = float(observed_z[idx])
        deltas = np.abs(vals - obs)
        worst_i = int(np.nanargmax(deltas)) if len(deltas) and np.isfinite(deltas).any() else 0
        sign_flips = int(np.sum((vals * obs) < 0)) if obs != 0.0 else 0
        summary[idx] = {
            "n_patients": n_patients,
            "min_z": float(np.nanmin(vals)),
            "max_z": float(np.nanmax(vals)),
            "max_abs_z_delta": float(np.nanmax(deltas)),
            "sign_flips": sign_flips,
            "rank1_retained_fraction": float(rank1_matches / n_patients) if idx == overall_top else None,
            "most_influential_patient_index": worst_i,
            "most_influential_patient_name": patient_names[worst_i],
        }
    return summary


def run_cohort_audit(
    cohort: str,
    *,
    out_dir: Path | None = DEFAULT_OUT_DIR,
    data_dir: Path | None = None,
    n_resamples: int = 9999,
    seed: int = 0,
    top_k: int = 5,
    core_api: CoreAPI | None = None,
    stats_api: StatsAPI | None = None,
) -> dict[str, Any]:
    """Run the already-seen real-data audit against the binding CONTRACT.md schema."""
    if cohort not in {"upmc", "charville"}:
        raise ValueError(f"unknown cohort {cohort!r}")
    core = core_api or load_default_core()
    stats = stats_api or load_default_stats()
    kwargs = {"data_dir": data_dir} if data_dir is not None else {}

    cells = core.load(cohort, **kwargs)
    g = core.grammar(list(cells["type_names"]))
    fnames = core.feature_names(g)
    x_full = np.asarray(core.truth(cells), dtype=np.float64)

    time = np.asarray(cells["time"], dtype=np.float64)
    event = np.asarray(cells["event"], dtype=np.float64)
    strata = np.asarray(cells["strata"], dtype=np.int32)
    patient_names = [str(p) for p in cells["patient_names"]]

    # Prefer native per-feature complete-case NaN handling in stats/; fallback only if stats rejects NaN.
    nan_count = int(np.isnan(x_full).sum())
    try:
        perm = stats.permutation_test(x_full, time, event, strata, n_resamples=n_resamples, seed=seed)
        x_tested = x_full
        tested_names = fnames
        grammar_indices = list(range(len(fnames)))
        nan_policy = f"complete_case_per_feature ({nan_count} NaN cells preserved)" if nan_count else "all_finite"
    except ValueError:
        x_tested, tested_names, grammar_indices, nan_policy = _finite_feature_subset(x_full, fnames)
        perm = stats.permutation_test(x_tested, time, event, strata, n_resamples=n_resamples, seed=seed)

    z_obs = np.asarray(perm["z"], dtype=np.float64)
    adj_p = np.asarray(perm["adjusted_p"], dtype=np.float64)
    order = sorted(
        (i for i in range(len(tested_names)) if np.isfinite(z_obs[i])),
        key=lambda idx: (-abs(float(z_obs[idx])), idx),
    )[:top_k]

    lopo = leave_one_patient_out(
        X=x_tested,
        time=time,
        event=event,
        strata=strata,
        patient_names=patient_names,
        observed_z=z_obs,
        top_indices=order,
        score_fn=stats.score,
    )
    top_hits = [
        {
            "rank": rank,
            "feature": tested_names[idx],
            "index": int(idx),
            "grammar_index": int(grammar_indices[idx]),
            "z": float(z_obs[idx]),
            "adjusted_p": float(adj_p[idx]),
            "lopo": lopo[idx],
        }
        for rank, idx in enumerate(order, start=1)
    ]

    leakage = leakage_from_cells(cells)
    top_hit_stable = bool(
        top_hits
        and top_hits[0]["lopo"]["sign_flips"] == 0
        and (top_hits[0]["lopo"]["rank1_retained_fraction"] or 0.0) >= 0.8
    )
    maxt_p = float(perm["p"])
    omnibus_p = float(perm["omnibus"]["p"])

    patient_of_region = np.asarray(cells["patient_of_region"], dtype=np.int64)
    regions_per_patient = np.bincount(patient_of_region, minlength=len(patient_names))

    sensitivity: dict[str, Any] | None = None
    if cohort == "upmc" and core.load_regions is not None and core.load_patients is not None:
        reg_tuple = core.load_regions("upmc", identity="tuple", **kwargs)
        pts_tuple = core.load_patients(reg_tuple)
        reg_tc = core.load_regions("upmc", identity="tuple+case", **kwargs)
        pts_tc = core.load_patients(reg_tc)
        sensitivity = {
            "kind": "tuple_vs_tuple+case_descriptive_only",
            "tuple_patients": int(len(pts_tuple)),
            "tuple_events": int(np.nansum(pts_tuple["event"])),
            "tuple_plus_case_patients": int(len(pts_tc)),
            "tuple_plus_case_events": int(np.nansum(pts_tc["event"])),
            "note": "Provisional tuple matching; never used for patient-level inference.",
        }

    excluded_summary = (
        "120 NA-labelled regions excluded from verified case-number cohort."
        if cohort == "upmc"
        else "Excluded cases: 24135, 24136 (no survival dates) and 24180 (spans coverslips c002 and c003)."
    )
    strata_note = (
        "Coverslip groups 0=c001-c003, 1=c004-c007 (7 / 43 patients; no verified case crosses group boundary). "
        "UPMC has 10 events on verified cases and is descriptive only."
        if cohort == "upmc"
        else "Coverslip strata c001-c004 mapped to 0-3 (patient-level cohort; 38 events)."
    )

    result = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "cohort": cohort,
        "exploratory": True,
        "already_seen": True,
        "already_seen_disclosure": ALREADY_SEEN_DISCLOSURE,
        "clinical_claims": False,
        "inference_scope": (
            "descriptive_verified_cases_and_tuple_bounds" if cohort == "upmc" else "patient_level_exploratory"
        ),
        "grammar_hash": core.grammar_hash(g),
        "n_regions": int(len(cells["region_names"])),
        "n_patients": int(len(patient_names)),
        "n_events": int(np.sum(event)),
        "excluded_summary": excluded_summary,
        "n_features_total": int(core.n_features(g)),
        "n_features_tested": int(x_tested.shape[1]),
        "nan_policy": nan_policy,
        "strata_note": strata_note,
        "leakage_audit": leakage,
        "maxT": {
            "statistic": float(perm["maxT"]),
            "p": maxt_p,
            "conditional_cutoff_alpha_05": conditional_permutation_cutoff(perm["null_z"], alpha=0.05),
            "cutoff_scope": "conditional_within_stratum_permutation_not_reusable_design_calibration",
            "n_resamples": int(n_resamples),
            "top_hits": top_hits,
        },
        "omnibus": {
            "statistics": {k: float(v) for k, v in perm["omnibus"]["statistics"].items()},
            "component_p": {k: float(v) for k, v in perm["omnibus"]["component_p"].items()},
            "p": omnibus_p,
        },
        "sensitivity": sensitivity,
        "atlas_placement": {
            "cohort": cohort,
            "n_patients": int(len(patient_names)),
            "n_events": int(np.sum(event)),
            "regions_per_patient_median": float(np.median(regions_per_patient)),
            "atlas_overlay_ready": True,
        },
        "prepared_outcome": choose_prepared_outcome(
            cohort=cohort,
            maxt_p=maxt_p,
            omnibus_p=omnibus_p,
            top_hit_stable=top_hit_stable,
            n_events=int(np.sum(event)),
        ),
    }

    validate_audit_result(result)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{cohort}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def format_summary_table(results: list[dict[str, Any]]) -> str:
    """Format an already-seen exploratory results table across audited cohorts."""
    lines = [
        "| Cohort (already-seen, exploratory) | P (Events) | Regions | Features | maxT (p) | Omnibus p | Coverslip Leakage (mean/max) | Top Hit (LOPO flips) | Prepared Outcome |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in results:
        hit = r["maxT"]["top_hits"][0] if r["maxT"]["top_hits"] else None
        hit_s = f"`{hit['feature']}` (z={hit['z']:.2f}, flips={hit['lopo']['sign_flips']})" if hit else "none"
        leak = r["leakage_audit"]
        lines.append(
            f"| {r['cohort']} | {r['n_patients']} ({r['n_events']}) | {r['n_regions']} | "
            f"{r['n_features_tested']}/{r['n_features_total']} | "
            f"{r['maxT']['statistic']:.2f} ({r['maxT']['p']:.4f}) | {r['omnibus']['p']:.4f} | "
            f"{leak['mean_leakage_fraction']:.2f}/{leak['max_leakage_fraction']:.2f} | "
            f"{hit_s} | {r['prepared_outcome']['key']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run already-seen real-data audit (Issue #6).")
    parser.add_argument("--cohort", nargs="+", default=["upmc", "charville"], choices=["upmc", "charville"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--n-resamples", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    results = [
        run_cohort_audit(
            coh,
            out_dir=args.out,
            data_dir=args.data_dir,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
        for coh in args.cohort
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(format_summary_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
