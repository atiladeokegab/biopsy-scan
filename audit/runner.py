"""Contract-based real-data audit runner for UPMC and Charville (Issue #6)."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit.leakage import compute_leakage_audit
from audit.schema import (
    ALREADY_SEEN_DISCLOSURE,
    AUDIT_SCHEMA_VERSION,
    choose_prepared_outcome,
    validate_audit_result,
)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class CoreAPI:
    load_regions: Callable[..., Any]
    load_patients: Callable[..., Any]
    load_cells: Callable[..., dict[str, Any]]
    grammar: Callable[[list[str]], dict[str, Any]]
    n_features: Callable[[dict[str, Any]], int]
    grammar_hash: Callable[[dict[str, Any]], str]
    feature_names: Callable[[dict[str, Any]], list[str]]
    truth_region: Callable[[np.ndarray, np.ndarray, list[int], int], tuple[np.ndarray, np.ndarray]]
    features: Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class StatsAPI:
    score: Callable[..., dict[str, np.ndarray]]
    permutation_test: Callable[..., dict[str, Any]]
    threshold: Callable[..., float]


def load_default_core() -> CoreAPI:
    from core import cohorts
    from core.grammar import feature_names, features, grammar, grammar_hash, n_features
    from core.truth import truth_region

    return CoreAPI(
        load_regions=cohorts.load_regions,
        load_patients=cohorts.load_patients,
        load_cells=cohorts.load_cells,
        grammar=grammar,
        n_features=n_features,
        grammar_hash=grammar_hash,
        feature_names=feature_names,
        truth_region=truth_region,
        features=features,
    )


def load_default_stats() -> StatsAPI:
    from stats.reference import permutation_test, score, threshold

    return StatsAPI(score=score, permutation_test=permutation_test, threshold=threshold)


def _aggregate_patient_features(
    *,
    regions: Any,
    patients: Any,
    cells: dict[str, Any],
    g: dict[str, Any],
    core_api: CoreAPI,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool truth_region sums per patient in sorted patient_id index order."""
    region_index = {rid: idx for idx, rid in enumerate(cells["region_ids"])}
    patient_index = {pid: idx for idx, pid in enumerate(patients["patient_id"])}
    ptr = cells["region_ptr"]
    xy_all = cells["xy_um"]
    ty_all = cells["cell_type"]
    focal = list(g["focal_types"])
    k_types = len(g["type_names"])

    first_s, first_m = core_api.truth_region(xy_all[:0], ty_all[:0], focal, k_types)
    p_count = len(patients)
    s_pat = np.zeros((p_count, *first_s.shape), dtype=np.float64)
    m_pat = np.zeros((p_count, *first_m.shape), dtype=np.int64)

    for row in regions.itertuples(index=False):
        ri = region_index[row.region_id]
        pi = patient_index[row.patient_id]
        sl = slice(int(ptr[ri]), int(ptr[ri + 1]))
        s_reg, m_reg = core_api.truth_region(xy_all[sl], ty_all[sl], focal, k_types)
        s_pat[pi] += s_reg
        m_pat[pi] += m_reg

    return core_api.features(s_pat, m_pat), m_pat


def leave_one_patient_out(
    *,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    strata: np.ndarray,
    patient_ids: list[str],
    observed_z: np.ndarray,
    top_indices: list[int],
    score_fn: Callable[..., dict[str, np.ndarray]],
) -> dict[int, dict[str, Any]]:
    """Measure leave-one-patient-out z stability and rank-1 retention for selected features."""
    n_patients = len(patient_ids)
    overall_top = int(np.argmax(np.abs(observed_z))) if len(observed_z) else -1
    z_traces: dict[int, list[float]] = {idx: [] for idx in top_indices}
    rank1_matches = 0

    for drop_i in range(n_patients):
        keep = np.ones(n_patients, dtype=bool)
        keep[drop_i] = False
        z_sub = score_fn(X[keep], time[keep], event[keep], strata[keep])["z"]
        if len(z_sub) and int(np.argmax(np.abs(z_sub))) == overall_top:
            rank1_matches += 1
        for idx in top_indices:
            z_traces[idx].append(float(z_sub[idx]))

    summary: dict[int, dict[str, Any]] = {}
    for idx in top_indices:
        vals = np.asarray(z_traces[idx], dtype=np.float64)
        obs = float(observed_z[idx])
        deltas = np.abs(vals - obs)
        worst_i = int(np.argmax(deltas)) if len(deltas) else 0
        sign_flips = int(np.sum((vals * obs) < 0)) if obs != 0.0 else 0
        summary[idx] = {
            "n_patients": n_patients,
            "min_z": float(np.min(vals)),
            "max_z": float(np.max(vals)),
            "max_abs_z_delta": float(np.max(deltas)),
            "sign_flips": sign_flips,
            "rank1_retained_fraction": float(rank1_matches / n_patients) if idx == overall_top else None,
            "most_influential_patient_index": worst_i,
            "most_influential_patient_id": patient_ids[worst_i],
        }
    return summary


def _run_permutation_audit(
    *,
    cohort: str,
    identity: str,
    n_resamples: int,
    seed: int,
    top_k: int,
    data_dir: Path | None,
    core_api: CoreAPI,
    stats_api: StatsAPI,
) -> dict[str, Any]:
    kwargs = {"data_dir": data_dir} if data_dir is not None else {}
    regions = core_api.load_regions(cohort, identity=identity, **kwargs) if cohort == "upmc" else core_api.load_regions(cohort, **kwargs)
    patients = core_api.load_patients(regions)
    cells = core_api.load_cells(cohort, **kwargs)
    g = core_api.grammar(cells["type_names"])
    fnames = core_api.feature_names(g)
    x_all, m_pat = _aggregate_patient_features(
        regions=regions, patients=patients, cells=cells, g=g, core_api=core_api
    )

    time_all = patients["time_days"].to_numpy(dtype=np.float64)
    event_all = patients["event"].to_numpy(dtype=np.float64)
    strata_all = patients["stratum"].to_numpy()
    pids_all = [str(p) for p in patients["patient_id"]]

    valid_mask = np.isfinite(time_all) & (time_all >= 0) & np.isin(event_all, [0.0, 1.0])
    excluded = [pid for pid, ok in zip(pids_all, valid_mask) if not ok]

    x_eval = x_all[valid_mask]
    time_eval = time_all[valid_mask]
    event_eval = event_all[valid_mask]
    strata_eval = strata_all[valid_mask]
    m_eval = m_pat[valid_mask]
    pids_eval = [pid for pid, ok in zip(pids_all, valid_mask) if ok]

    finite_cols = np.isfinite(x_eval).all(axis=0)
    x_tested = x_eval[:, finite_cols]
    tested_names = [name for name, ok in zip(fnames, finite_cols) if ok]

    dropped_focal = []
    for ti, ft in enumerate(g["focal_types"]):
        n_zero = int(np.sum(m_eval[:, ti] == 0))
        if n_zero > 0:
            dropped_focal.append(f"{g['type_names'][ft]} ({n_zero}/{len(m_eval)} patients lack focal cells)")
    dropped_reason = (
        "Excluded features for focal types absent in some evaluable patients: " + "; ".join(dropped_focal)
        if dropped_focal or not finite_cols.all()
        else None
    )

    perm = stats_api.permutation_test(
        x_tested, time_eval, event_eval, strata_eval, n_resamples=n_resamples, seed=seed
    )
    z_obs = perm["z"]
    adj_p = perm["adjusted_p"]
    order = sorted(range(len(tested_names)), key=lambda idx: (-abs(float(z_obs[idx])), idx))[:top_k]
    lopo = leave_one_patient_out(
        X=x_tested,
        time=time_eval,
        event=event_eval,
        strata=strata_eval,
        patient_ids=pids_eval,
        observed_z=z_obs,
        top_indices=order,
        score_fn=stats_api.score,
    )
    top_hits = [
        {
            "rank": rank,
            "feature": tested_names[idx],
            "index": int(idx),
            "z": float(z_obs[idx]),
            "adjusted_p": float(adj_p[idx]),
            "lopo": lopo[idx],
        }
        for rank, idx in enumerate(order, start=1)
    ]

    leakage = compute_leakage_audit(regions)
    top_hit_stable = bool(
        top_hits
        and top_hits[0]["lopo"]["sign_flips"] == 0
        and (top_hits[0]["lopo"]["rank1_retained_fraction"] or 0.0) >= 0.8
    )
    maxt_p = float(perm["p"])
    omnibus_p = float(perm["omnibus"]["p"])

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "cohort": cohort,
        "exploratory": True,
        "already_seen": True,
        "already_seen_disclosure": ALREADY_SEEN_DISCLOSURE,
        "clinical_claims": False,
        "inference_scope": (
            "descriptive_with_tuple_sensitivity" if cohort == "upmc" else "patient_level_exploratory"
        ),
        "grammar_hash": core_api.grammar_hash(g),
        "n_regions": int(len(regions)),
        "n_patients_total": int(len(patients)),
        "n_patients_evaluable": int(len(pids_eval)),
        "n_events": int(np.sum(event_eval)),
        "excluded_patients": excluded,
        "n_features_total": int(core_api.n_features(g)),
        "n_features_tested": int(x_tested.shape[1]),
        "dropped_features_reason": dropped_reason,
        "coverslips": sorted(regions["coverslip"].astype(str).unique().tolist()),
        "patients_spanning_multiple_coverslips": leakage["patients_spanning_multiple_coverslips"],
        "strata_note": (
            "Modal coverslip per patient (UPMC patients span multiple coverslips; descriptive only with tuple+case sensitivity)."
            if cohort == "upmc"
            else "Patient-level coverslip strata (Charville case numbers identify patients)."
        ),
        "leakage_audit": leakage,
        "maxT": {
            "statistic": float(perm["maxT"]),
            "p": maxt_p,
            "threshold_alpha_05": float(stats_api.threshold({"alpha": 0.05}, perm["null_z"])),
            "n_resamples": int(n_resamples),
            "top_hits": top_hits,
        },
        "omnibus": {
            "statistics": {k: float(v) for k, v in perm["omnibus"]["statistics"].items()},
            "component_p": {k: float(v) for k, v in perm["omnibus"]["component_p"].items()},
            "p": omnibus_p,
        },
        "sensitivity": None,
        "atlas_placement": {
            "cohort": cohort,
            "n_evaluable": int(len(pids_eval)),
            "n_events": int(np.sum(event_eval)),
            "regions_per_patient_median": float(np.median(patients["n_regions"].to_numpy())),
            "atlas_overlay_ready": True,
        },
        "prepared_outcome": choose_prepared_outcome(
            maxt_p=maxt_p, omnibus_p=omnibus_p, top_hit_stable=top_hit_stable
        ),
    }


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
    """Run the already-seen real-data audit for one cohort and optionally save JSON."""
    core = core_api or load_default_core()
    stats = stats_api or load_default_stats()
    result = _run_permutation_audit(
        cohort=cohort,
        identity="tuple",
        n_resamples=n_resamples,
        seed=seed,
        top_k=top_k,
        data_dir=data_dir,
        core_api=core,
        stats_api=stats,
    )
    if cohort == "upmc":
        tc = _run_permutation_audit(
            cohort="upmc",
            identity="tuple+case",
            n_resamples=n_resamples,
            seed=seed,
            top_k=1,
            data_dir=data_dir,
            core_api=core,
            stats_api=stats,
        )
        result["sensitivity"] = {
            "kind": "tuple+case",
            "n_patients": tc["n_patients_evaluable"],
            "n_events": tc["n_events"],
            "maxT": tc["maxT"]["statistic"],
            "maxT_p": tc["maxT"]["p"],
            "omnibus_p": tc["omnibus"]["p"],
            "top_hit_feature": tc["maxT"]["top_hits"][0]["feature"] if tc["maxT"]["top_hits"] else None,
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
        "| Cohort (already-seen, exploratory) | Evaluable P (Events) | Features Tested | maxT (p) | Omnibus p | Leakage (mean/max) | Top Hit (LOPO flips) | Prepared Outcome |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in results:
        hit = r["maxT"]["top_hits"][0] if r["maxT"]["top_hits"] else None
        hit_s = f"`{hit['feature']}` (z={hit['z']:.2f}, flips={hit['lopo']['sign_flips']})" if hit else "none"
        leak = r["leakage_audit"]
        lines.append(
            f"| {r['cohort']} | {r['n_patients_evaluable']} ({r['n_events']}) | "
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
