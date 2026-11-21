from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit.leakage import compute_leakage_audit
from audit.runner import CoreAPI, StatsAPI, leave_one_patient_out, run_cohort_audit
from audit.schema import (
    AUDIT_SCHEMA_VERSION,
    PREPARED_OUTCOMES,
    choose_prepared_outcome,
    validate_audit_result,
)


def _sample_record(maxt_p: float = 0.42, omnibus_p: float = 0.004, stable: bool = False) -> dict:
    outcome = choose_prepared_outcome(maxt_p=maxt_p, omnibus_p=omnibus_p, top_hit_stable=stable)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "cohort": "upmc",
        "exploratory": True,
        "already_seen": True,
        "already_seen_disclosure": (
            "Both cohorts' outcomes have been examined by reviewers already; "
            "all real-data results are already-seen and exploratory, not confirmatory."
        ),
        "clinical_claims": False,
        "inference_scope": "descriptive_with_tuple_sensitivity",
        "grammar_hash": "a" * 64,
        "n_regions": 308,
        "n_patients_total": 79,
        "n_patients_evaluable": 79,
        "n_events": 27,
        "excluded_patients": [],
        "n_features_total": 612,
        "n_features_tested": 612,
        "dropped_features_reason": None,
        "coverslips": ["c001", "c002"],
        "patients_spanning_multiple_coverslips": 74,
        "strata_note": "Modal coverslip per patient; UPMC is descriptive with tuple+case sensitivity.",
        "leakage_audit": {
            "total_patients": 79,
            "patients_spanning_multiple_coverslips": 74,
            "mean_leakage_fraction": 0.9,
            "min_leakage_fraction": 0.8,
            "max_leakage_fraction": 1.0,
            "per_coverslip": [
                {"coverslip": "c001", "n_patients": 40, "shared_patients": 36, "leakage_fraction": 0.9},
            ],
        },
        "maxT": {
            "statistic": 3.1,
            "p": maxt_p,
            "threshold_alpha_05": 3.85,
            "n_resamples": 999,
            "top_hits": [
                {
                    "rank": 1,
                    "feature": "Tumor|CD8|25um|enrichment",
                    "index": 0,
                    "z": 3.1,
                    "adjusted_p": maxt_p,
                    "lopo": {
                        "n_patients": 79,
                        "min_z": 2.9,
                        "max_z": 3.3,
                        "max_abs_z_delta": 0.2,
                        "sign_flips": 0,
                        "rank1_retained_fraction": 1.0,
                        "most_influential_patient_index": 3,
                    },
                }
            ],
        },
        "omnibus": {
            "statistics": {"maxT": 3.1, "sum_z2": 820.0, "HC": 4.2},
            "component_p": {"maxT": maxt_p, "sum_z2": 0.004, "HC": 0.01},
            "p": omnibus_p,
        },
        "sensitivity": {
            "kind": "tuple+case",
            "n_patients": 81,
            "n_events": 27,
            "maxT": 3.05,
            "maxT_p": 0.45,
            "omnibus_p": 0.006,
            "top_hit_feature": "Tumor|CD8|25um|enrichment",
        },
        "atlas_placement": {
            "cohort": "upmc",
            "n_evaluable": 79,
            "n_events": 27,
            "regions_per_patient_median": 4.0,
            "atlas_overlay_ready": True,
        },
        "prepared_outcome": outcome,
    }


def test_prepared_outcome_selection_covers_all_three_outcomes() -> None:
    o1 = choose_prepared_outcome(maxt_p=0.40, omnibus_p=0.004, top_hit_stable=False)
    o2 = choose_prepared_outcome(maxt_p=0.02, omnibus_p=0.02, top_hit_stable=True)
    o3 = choose_prepared_outcome(maxt_p=0.50, omnibus_p=0.40, top_hit_stable=False)
    assert o1["key"] == "omnibus_only_diffuse"
    assert o2["key"] == "real_hit_stable"
    assert o3["key"] == "atlas_design_choices"
    assert set(PREPARED_OUTCOMES) == {"atlas_design_choices", "omnibus_only_diffuse", "real_hit_stable"}


def test_schema_validation_accepts_valid_record_and_rejects_missing_or_clinical_claims() -> None:
    rec = _sample_record()
    validate_audit_result(rec)

    bad_clinical = {**rec, "clinical_claims": True}
    with pytest.raises(ValueError, match="clinical_claims"):
        validate_audit_result(bad_clinical)

    bad_seen = {**rec, "already_seen": False}
    with pytest.raises(ValueError, match="already_seen"):
        validate_audit_result(bad_seen)

    bad_exploratory = {**rec, "exploratory": False}
    with pytest.raises(ValueError, match="exploratory"):
        validate_audit_result(bad_exploratory)


def test_leakage_audit_detects_cross_coverslip_patient_overlap() -> None:
    regions = pd.DataFrame([
        {"region_id": "r1", "coverslip": "c001", "patient_id": "P1"},
        {"region_id": "r2", "coverslip": "c002", "patient_id": "P1"},
        {"region_id": "r3", "coverslip": "c001", "patient_id": "P2"},
        {"region_id": "r4", "coverslip": "c002", "patient_id": "P3"},
        {"region_id": "r5", "coverslip": "c003", "patient_id": "P4"},
    ])
    out = compute_leakage_audit(regions)
    assert out["total_patients"] == 4
    assert out["patients_spanning_multiple_coverslips"] == 1
    assert out["min_leakage_fraction"] == pytest.approx(0.0)
    assert out["max_leakage_fraction"] == pytest.approx(0.5)
    by_cov = {row["coverslip"]: row for row in out["per_coverslip"]}
    assert by_cov["c001"] == {"coverslip": "c001", "n_patients": 2, "shared_patients": 1, "leakage_fraction": pytest.approx(0.5)}
    assert by_cov["c002"] == {"coverslip": "c002", "n_patients": 2, "shared_patients": 1, "leakage_fraction": pytest.approx(0.5)}
    assert by_cov["c003"] == {"coverslip": "c003", "n_patients": 1, "shared_patients": 0, "leakage_fraction": pytest.approx(0.0)}


def _fake_stats_api() -> StatsAPI:
    def score(X, time, event, strata):
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)
        event = np.asarray(event, dtype=np.float64)
        strata = np.asarray(strata)
        if strata.dtype.kind not in "biufUS":
            raise ValueError("strata must contain finite numeric or string labels")
        if not np.isfinite(X).all() or not np.isfinite(time).all():
            raise ValueError("non-finite")
        r = np.zeros_like(time)
        for s in np.unique(strata):
            ix = np.flatnonzero(strata == s)
            r[ix] = (event[ix] - event[ix].mean()) + 0.01 * (time[ix].mean() - time[ix])
            r[ix] -= r[ix].mean()
        Xc = X - X.mean(axis=0, keepdims=True)
        U = Xc.T @ r
        V = np.sum(Xc * Xc, axis=0) * np.dot(r, r) / max(1, len(r) - 1)
        z = np.divide(U, np.sqrt(V), out=np.zeros_like(U), where=V > 0)
        return {"U": U, "V": V, "z": z}

    def permutation_test(X, time, event, strata, *, n_resamples=99, seed=0):
        obs = score(X, time, event, strata)
        rng = np.random.default_rng(seed)
        null_z = rng.normal(0.0, 0.7, size=(n_resamples, X.shape[1]))
        max_obs = float(np.max(np.abs(obs["z"])))
        null_max = np.max(np.abs(null_z), axis=1)
        p = float((1 + np.sum(null_max >= max_obs)) / (n_resamples + 1))
        adj = np.array([(1 + np.sum(null_max >= abs(zi))) / (n_resamples + 1) for zi in obs["z"]])
        return {
            **obs,
            "maxT": max_obs,
            "p": p,
            "adjusted_p": adj,
            "omnibus": {
                "statistics": {"maxT": max_obs, "sum_z2": float(np.sum(obs["z"] ** 2)), "HC": 1.5},
                "component_p": {"maxT": p, "sum_z2": 0.02, "HC": 0.03},
                "p": 0.02,
            },
            "null_z": null_z,
            "exact": False,
        }

    def threshold(design, null_studies):
        del design
        return float(np.quantile(np.max(np.abs(null_studies), axis=1), 0.95))

    return StatsAPI(score=score, permutation_test=permutation_test, threshold=threshold)


def test_leave_one_patient_out_captures_outlier_influence() -> None:
    stats = _fake_stats_api()
    rng = np.random.default_rng(7)
    X = rng.normal(0.0, 0.2, size=(12, 3))
    time = np.linspace(10.0, 120.0, 12)
    event = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
    strata = np.array(["c001"] * 6 + ["c002"] * 6)
    # Patient 0 drives feature 0
    X[0, 0] = 5.0
    obs = stats.score(X, time, event, strata)["z"]
    lopo = leave_one_patient_out(
        X=X,
        time=time,
        event=event,
        strata=strata,
        patient_ids=[f"P{i}" for i in range(12)],
        observed_z=obs,
        top_indices=[0, 1],
        score_fn=stats.score,
    )
    assert lopo[0]["most_influential_patient_id"] == "P0"
    assert lopo[0]["max_abs_z_delta"] > 0.1


def test_run_cohort_audit_handles_missing_endpoints_nan_features_and_writes_json(tmp_path: Path) -> None:
    reg_df = pd.DataFrame([
        {"region_id": "r0", "coverslip": "c001", "case": "P0", "patient_id": "P0", "time_days": 50.0, "event": 1.0},
        {"region_id": "r1", "coverslip": "c001", "case": "P1", "patient_id": "P1", "time_days": 80.0, "event": 0.0},
        {"region_id": "r2", "coverslip": "c001", "case": "P2", "patient_id": "P2", "time_days": 60.0, "event": 1.0},
        {"region_id": "r3", "coverslip": "c002", "case": "P3", "patient_id": "P3", "time_days": 90.0, "event": 0.0},
        {"region_id": "r4", "coverslip": "c002", "case": "P4", "patient_id": "P4", "time_days": 40.0, "event": 1.0},
        {"region_id": "r5", "coverslip": "c002", "case": "P5", "patient_id": "P5", "time_days": np.nan, "event": np.nan},
    ])

    def load_regions(cohort, data_dir=None, identity="tuple"):
        del cohort, data_dir, identity
        return reg_df

    def load_patients(regions):
        g = regions.groupby("patient_id", sort=True)
        pts = g[["time_days", "event"]].first().reset_index()
        pts["stratum"] = g["coverslip"].first().to_numpy()
        pts["n_coverslips"] = g["coverslip"].nunique().to_numpy()
        pts["n_regions"] = g.size().to_numpy()
        return pts

    def load_cells(cohort, region_ids=None, data_dir=None):
        del cohort, region_ids, data_dir
        # 2 cells per region -> 12 cells total; Tumor B missing in P1 so its features are NaN
        xy = np.tile([[0.0, 0.0], [10.0, 0.0]], (6, 1))
        # Types: 0=Tumor A, 1=Tumor B, 2=CD8
        ty = np.array([0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2], dtype=np.int16)
        ty[3] = 1  # Tumor B present only in r1? Keep r1 with only Tumor A & CD8; put Tumor B in r0
        ty[1] = 1
        return {
            "xy_um": xy,
            "cell_type": ty,
            "region_ptr": np.arange(0, 14, 2, dtype=np.int64),
            "region_ids": list(reg_df["region_id"]),
            "type_names": ["Tumor A", "Tumor B", "CD8"],
        }

    def grammar(type_names):
        return {"type_names": type_names, "focal_types": [0, 1], "radii_um": [25.0], "stats": ["fraction"]}

    def n_features(g):
        return len(g["focal_types"]) * len(g["type_names"]) * len(g["radii_um"]) * len(g["stats"])

    def grammar_hash(g):
        del g
        return "b" * 64

    def feature_names(g):
        tn = g["type_names"]
        return [f"{tn[t]}|{tn[j]}|25um|fraction" for t in g["focal_types"] for j in range(len(tn))]

    def truth_region(xy, ty, focal, K):
        del xy
        S = np.zeros((len(focal), K, 1, 1))
        m = np.zeros(len(focal), dtype=np.int64)
        for ti, ft in enumerate(focal):
            cnt = int((ty == ft).sum())
            m[ti] = cnt
            if cnt > 0:
                S[ti, :, 0, 0] = np.arange(1, K + 1, dtype=np.float64) * cnt
        return S, m

    def features(S, m):
        with np.errstate(invalid="ignore", divide="ignore"):
            X = S / m[:, :, None, None, None]
        return X.reshape(len(S), -1)

    core_api = CoreAPI(
        load_regions=load_regions,
        load_patients=load_patients,
        load_cells=load_cells,
        grammar=grammar,
        n_features=n_features,
        grammar_hash=grammar_hash,
        feature_names=feature_names,
        truth_region=truth_region,
        features=features,
    )
    out = run_cohort_audit(
        "charville",
        out_dir=tmp_path,
        n_resamples=39,
        seed=0,
        top_k=2,
        core_api=core_api,
        stats_api=_fake_stats_api(),
    )
    validate_audit_result(out)
    assert out["n_patients_total"] == 6
    assert out["n_patients_evaluable"] == 5
    assert out["excluded_patients"] == ["P5"]
    assert out["n_features_total"] == 6
    assert out["n_features_tested"] == 3
    assert "Tumor B" in (out["dropped_features_reason"] or "")
    saved = json.loads((tmp_path / "charville.json").read_text())
    assert saved["cohort"] == "charville"
    assert saved["already_seen"] is True
    assert saved["clinical_claims"] is False
    assert all("grammar_index" in h for h in saved["maxT"]["top_hits"])
