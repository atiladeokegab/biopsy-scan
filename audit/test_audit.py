from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit.leakage import compute_leakage_audit, leakage_from_cells
from audit.runner import (
    CoreAPI,
    StatsAPI,
    conditional_permutation_cutoff,
    format_summary_table,
    leave_one_patient_out,
    run_cohort_audit,
)
from audit.schema import (
    AUDIT_SCHEMA_VERSION,
    CONDITIONAL_CUTOFF_SCOPE,
    PREPARED_OUTCOMES,
    choose_prepared_outcome,
    validate_audit_result,
)


def _sample_record(
    *, cohort: str = "charville", maxt_p: float = 0.42, omnibus_p: float = 0.004, stable: bool = False, n_events: int = 38
) -> dict:
    outcome = choose_prepared_outcome(
        cohort=cohort, maxt_p=maxt_p, omnibus_p=omnibus_p, top_hit_stable=stable, n_events=n_events
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "cohort": cohort,
        "exploratory": True,
        "already_seen": True,
        "already_seen_disclosure": (
            "Both cohorts' outcomes have been examined by reviewers already; "
            "all real-data results are already-seen and exploratory, not confirmatory."
        ),
        "clinical_claims": False,
        "inference_scope": "patient_level_exploratory" if cohort == "charville" else "descriptive_verified_cases_and_tuple_bounds",
        "grammar_hash": "a" * 64,
        "n_regions": 286,
        "n_patients": 159,
        "n_events": n_events,
        "excluded_summary": "Excluded cases: 24135, 24136, 24180.",
        "n_features_total": 630,
        "n_features_tested": 630,
        "nan_policy": "complete_case_per_feature",
        "strata_note": "Coverslip strata c001-c004 mapped to 0-3.",
        "leakage_audit": {
            "total_patients": 159,
            "patients_spanning_multiple_coverslips": 0,
            "mean_leakage_fraction": 0.0,
            "min_leakage_fraction": 0.0,
            "max_leakage_fraction": 0.0,
            "per_coverslip": [
                {"coverslip": "c001", "n_patients": 40, "shared_patients": 0, "leakage_fraction": 0.0},
            ],
        },
        "maxT": {
            "statistic": 3.1,
            "p": maxt_p,
            "conditional_cutoff_alpha_05": 3.65,
            "cutoff_scope": "conditional_within_stratum_permutation_not_reusable_design_calibration",
            "n_resamples": 999,
            "top_hits": [
                {
                    "rank": 1,
                    "feature": "Tumor 3|Blood vessel|100um|enrichment",
                    "index": 0,
                    "grammar_index": 0,
                    "z": 3.1,
                    "adjusted_p": maxt_p,
                    "lopo": {
                        "n_patients": 159,
                        "min_z": 2.9,
                        "max_z": 3.3,
                        "max_abs_z_delta": 0.2,
                        "sign_flips": 0,
                        "rank1_retained_fraction": 1.0,
                        "most_influential_patient_index": 3,
                        "most_influential_patient_name": "22380",
                    },
                }
            ],
        },
        "omnibus": {
            "statistics": {"maxT": 3.1, "sum_z2": 820.0, "HC": 4.2},
            "component_p": {"maxT": maxt_p, "sum_z2": 0.004, "HC": 0.01},
            "p": omnibus_p,
        },
        "sensitivity": None,
        "atlas_placement": {
            "cohort": cohort,
            "n_patients": 159,
            "n_events": n_events,
            "regions_per_patient_median": 2.0,
            "atlas_overlay_ready": True,
        },
        "prepared_outcome": outcome,
    }


def test_prepared_outcome_selection_covers_all_three_outcomes_and_keeps_upmc_descriptive() -> None:
    o1 = choose_prepared_outcome(cohort="charville", maxt_p=0.40, omnibus_p=0.004, top_hit_stable=False, n_events=38)
    o1_low_events = choose_prepared_outcome(
        cohort="charville", maxt_p=0.40, omnibus_p=0.004, top_hit_stable=False, n_events=3
    )
    o2 = choose_prepared_outcome(cohort="charville", maxt_p=0.02, omnibus_p=0.02, top_hit_stable=True, n_events=38)
    o3 = choose_prepared_outcome(cohort="charville", maxt_p=0.50, omnibus_p=0.40, top_hit_stable=False, n_events=38)
    upmc = choose_prepared_outcome(cohort="upmc", maxt_p=0.01, omnibus_p=0.01, top_hit_stable=True, n_events=10)
    assert o1["key"] == "omnibus_only_diffuse"
    assert o1_low_events["key"] == "omnibus_only_diffuse"
    stmt = o1["statement"].lower()
    assert "exploratory conditional association" in stmt
    assert "diffuse" not in stmt
    assert "signal is real" not in stmt
    assert o2["key"] == "real_hit_stable"
    assert o3["key"] == "atlas_design_choices"
    assert upmc["key"] == "atlas_design_choices"
    assert set(PREPARED_OUTCOMES) == {"atlas_design_choices", "omnibus_only_diffuse", "real_hit_stable"}


def test_schema_validation_accepts_valid_record_and_rejects_missing_or_clinical_claims() -> None:
    rec = _sample_record()
    validate_audit_result(rec)

    with pytest.raises(ValueError, match="clinical_claims"):
        validate_audit_result({**rec, "clinical_claims": True})

    with pytest.raises(ValueError, match="already_seen"):
        validate_audit_result({**rec, "already_seen": False})

    with pytest.raises(ValueError, match="exploratory"):
        validate_audit_result({**rec, "exploratory": False})

    with pytest.raises(ValueError, match="threshold_alpha_05"):
        validate_audit_result({**rec, "maxT": {**rec["maxT"], "threshold_alpha_05": 3.5}})

    with pytest.raises(ValueError, match="conditional_cutoff_alpha_05"):
        bad_maxt = dict(rec["maxT"])
        bad_maxt.pop("conditional_cutoff_alpha_05")
        validate_audit_result({**rec, "maxT": bad_maxt})

    with pytest.raises(ValueError, match="cutoff_scope"):
        validate_audit_result({**rec, "maxT": {**rec["maxT"], "cutoff_scope": "unconditional_reusable_design_calibration"}})

    for bad_cutoff in ("bad cutoff", -1.0, True, float("inf")):
        with pytest.raises(ValueError, match="conditional_cutoff_alpha_05"):
            validate_audit_result({**rec, "maxT": {**rec["maxT"], "conditional_cutoff_alpha_05": bad_cutoff}})

    validate_audit_result({**rec, "maxT": {**rec["maxT"], "conditional_cutoff_alpha_05": None, "cutoff_scope": CONDITIONAL_CUTOFF_SCOPE}})

    assert conditional_permutation_cutoff(np.ones((10, 3))) is None
    with pytest.raises(ValueError, match="finite"):
        conditional_permutation_cutoff(np.array([[1.0, np.nan], [0.5, 0.2]]))


def test_leakage_audit_detects_cross_coverslip_patient_overlap_and_works_on_contract_cells() -> None:
    cells = {
        "region_names": [
            "UPMC_c001_v001_r001_reg001",
            "UPMC_c002_v001_r001_reg002",
            "UPMC_c001_v001_r001_reg003",
            "UPMC_c002_v001_r001_reg004",
            "UPMC_c003_v001_r001_reg005",
        ],
        "patient_names": ["P1", "P2", "P3", "P4"],
        "patient_of_region": np.array([0, 0, 1, 2, 3], dtype=np.int32),
    }
    out = leakage_from_cells(cells)
    assert out["total_patients"] == 4
    assert out["patients_spanning_multiple_coverslips"] == 1
    assert out["min_leakage_fraction"] == pytest.approx(0.0)
    assert out["max_leakage_fraction"] == pytest.approx(0.5)
    by_cov = {row["coverslip"]: row for row in out["per_coverslip"]}
    assert by_cov["c001"] == {"coverslip": "c001", "n_patients": 2, "shared_patients": 1, "leakage_fraction": pytest.approx(0.5)}
    assert by_cov["c002"] == {"coverslip": "c002", "n_patients": 2, "shared_patients": 1, "leakage_fraction": pytest.approx(0.5)}
    assert by_cov["c003"] == {"coverslip": "c003", "n_patients": 1, "shared_patients": 0, "leakage_fraction": pytest.approx(0.0)}
    assert compute_leakage_audit(pd.DataFrame())["total_patients"] == 0


def _fake_stats_api(*, support_nan: bool = True) -> StatsAPI:
    def score(X, time, event, strata):
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64)
        event = np.asarray(event, dtype=np.float64)
        strata = np.asarray(strata)
        if strata.dtype.kind not in "biufUS":
            raise ValueError("strata must contain finite numeric or string labels")
        if not support_nan and not np.isfinite(X).all():
            raise ValueError("non-finite X")
        r = np.zeros_like(time)
        for s in np.unique(strata):
            ix = np.flatnonzero(strata == s)
            r[ix] = (event[ix] - event[ix].mean()) + 0.01 * (time[ix].mean() - time[ix])
            r[ix] -= r[ix].mean()
        z = np.zeros(X.shape[1], dtype=np.float64)
        U = np.zeros(X.shape[1], dtype=np.float64)
        V = np.zeros(X.shape[1], dtype=np.float64)
        for c in range(X.shape[1]):
            ok = np.isfinite(X[:, c])
            if ok.sum() < 2:
                continue
            xc = X[ok, c] - X[ok, c].mean()
            rc = r[ok] - r[ok].mean()
            U[c] = float(xc @ rc)
            V[c] = float((xc @ xc) * (rc @ rc) / max(1, len(rc) - 1))
            z[c] = U[c] / np.sqrt(V[c]) if V[c] > 0 else 0.0
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

    return StatsAPI(
        score=score,
        permutation_test=permutation_test,
        threshold=threshold,
        maxT=lambda z: float(np.max(np.abs(z))),
    )


def test_leave_one_patient_out_captures_outlier_influence() -> None:
    stats = _fake_stats_api()
    rng = np.random.default_rng(7)
    X = rng.normal(0.0, 0.2, size=(12, 3))
    time = np.linspace(10.0, 120.0, 12)
    event = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
    strata = np.array([0] * 6 + [1] * 6, dtype=np.int32)
    X[0, 0] = 5.0
    obs = stats.score(X, time, event, strata)["z"]
    lopo = leave_one_patient_out(
        X=X,
        time=time,
        event=event,
        strata=strata,
        patient_names=[f"P{i}" for i in range(12)],
        observed_z=obs,
        top_indices=[0, 1],
        score_fn=stats.score,
    )
    assert lopo[0]["most_influential_patient_name"] == "P0"
    assert lopo[0]["max_abs_z_delta"] > 0.1


def _synthetic_core() -> CoreAPI:
    def load(cohort, data_dir=None):
        del data_dir
        return {
            "cell_type": np.array([0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 2], dtype=np.int8),
            "x": np.zeros(11, dtype=np.float32),
            "y": np.zeros(11, dtype=np.float32),
            "region_id": np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], dtype=np.int32),
            "patient_of_region": np.array([0, 1, 2, 3, 4], dtype=np.int32),
            "time": np.array([50.0, 80.0, 60.0, 90.0, 40.0], dtype=np.float64),
            "event": np.array([True, False, True, False, True], dtype=bool),
            "strata": np.array([0, 0, 0, 1, 1], dtype=np.int32),
            "type_names": ["Tumor A", "Tumor B", "CD8"],
            "region_names": [
                f"{cohort.upper()}_c001_v001_r001_reg001",
                f"{cohort.upper()}_c001_v001_r001_reg002",
                f"{cohort.upper()}_c001_v001_r001_reg003",
                f"{cohort.upper()}_c002_v001_r001_reg004",
                f"{cohort.upper()}_c002_v001_r001_reg005",
            ],
            "patient_names": ["P0", "P1", "P2", "P3", "P4"],
        }

    def truth(cells):
        del cells
        # Tumor B is present only for P4; P0-P3 stay NaN (no zero fill)
        return np.array([
            [0.2, 1.1, 0.3, np.nan, np.nan, np.nan],
            [0.4, 1.3, 0.2, np.nan, np.nan, np.nan],
            [0.1, 0.9, 0.5, np.nan, np.nan, np.nan],
            [0.5, 1.5, 0.1, np.nan, np.nan, np.nan],
            [0.3, 1.0, 0.4, 0.6, 1.2, 0.3],
        ], dtype=np.float64)

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

    return CoreAPI(
        load=load,
        truth=truth,
        grammar=grammar,
        n_features=n_features,
        grammar_hash=grammar_hash,
        feature_names=feature_names,
    )


def test_run_cohort_audit_consumes_contract_load_and_truth_preserves_nan_and_formats_table(tmp_path: Path) -> None:
    core_api = _synthetic_core()
    out_complete = run_cohort_audit(
        "charville",
        out_dir=tmp_path,
        n_resamples=39,
        seed=0,
        top_k=2,
        core_api=core_api,
        stats_api=_fake_stats_api(support_nan=True),
    )
    validate_audit_result(out_complete)
    assert out_complete["n_patients"] == 5
    assert out_complete["n_events"] == 3
    assert out_complete["n_features_total"] == 6
    assert out_complete["n_features_tested"] == 6
    assert "complete_case_per_feature" in out_complete["nan_policy"]

    # Fallback when stats rejects NaN
    out_fallback = run_cohort_audit(
        "charville",
        out_dir=tmp_path / "fb",
        n_resamples=39,
        seed=0,
        top_k=2,
        core_api=core_api,
        stats_api=_fake_stats_api(support_nan=False),
    )
    assert out_fallback["n_features_tested"] == 3
    assert out_fallback["nan_policy"] == "fallback_dropped_nonfinite_cols"
    assert all("grammar_index" in h for h in out_fallback["maxT"]["top_hits"])
    saved = json.loads((tmp_path / "charville.json").read_text())
    assert saved["already_seen"] is True
    assert saved["clinical_claims"] is False
    assert "already-seen, exploratory" in format_summary_table([out_complete])
