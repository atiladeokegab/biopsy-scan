from __future__ import annotations

import pytest

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
