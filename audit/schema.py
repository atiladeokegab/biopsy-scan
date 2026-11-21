"""Output schema, prepared outcome selector, and exploratory/already-seen guards for the real-data audit."""
from __future__ import annotations

from typing import Any

AUDIT_SCHEMA_VERSION = 1

ALREADY_SEEN_DISCLOSURE = (
    "Both cohorts' outcomes have been examined by reviewers already; "
    "all real-data results are already-seen and exploratory, not confirmatory."
)

PREPARED_OUTCOMES: dict[str, dict[str, str]] = {
    "atlas_design_choices": {
        "key": "atlas_design_choices",
        "label": "The atlas shows design choices matter",
        "statement": (
            "Exploratory already-seen cohort search did not isolate a stable single-pattern hit; "
            "the cohort-specific atlas quantifies how regions per patient, field of view, and cell-typing "
            "mis-calls govern study power."
        ),
    },
    "omnibus_only_diffuse": {
        "key": "omnibus_only_diffuse",
        "label": "The real-data search finds nothing under maxT but the omnibus is significant",
        "statement": (
            "The signal is real but diffuse (already-seen exploratory omnibus p <= 0.05 while single-step "
            "maxT > 0.05); no single pattern is detectable at this sample size, and the cohort atlas marks "
            "the study size where one would be."
        ),
    },
    "real_hit_stable": {
        "key": "real_hit_stable",
        "label": "A real hit survives with stable influence",
        "statement": (
            "A ranked hypothesis (already-seen and exploratory; no clinical claim) survives single-step "
            "maxT with stable leave-one-patient-out influence, and the cohort atlas reports how often a "
            "study of this size identifies it."
        ),
    },
}

_REQUIRED_KEYS = {
    "schema_version",
    "cohort",
    "exploratory",
    "already_seen",
    "already_seen_disclosure",
    "clinical_claims",
    "inference_scope",
    "grammar_hash",
    "n_regions",
    "n_patients",
    "n_events",
    "excluded_summary",
    "n_features_total",
    "n_features_tested",
    "nan_policy",
    "strata_note",
    "leakage_audit",
    "maxT",
    "omnibus",
    "sensitivity",
    "atlas_placement",
    "prepared_outcome",
}


def choose_prepared_outcome(
    *,
    cohort: str,
    maxt_p: float,
    omnibus_p: float,
    top_hit_stable: bool,
    n_events: int = 38,
    alpha: float = 0.05,
) -> dict[str, str]:
    """Select one of the three prepared outcomes from IDEA.md based on observed test results and cohort policy.

    Charville is the only cohort with patient-level inference; UPMC (10 events on verified cases; tuple match
    provisional/descriptive only) never claims a patient-level real hit.
    """
    if cohort == "charville" and n_events >= 20:
        if maxt_p <= alpha and top_hit_stable:
            return dict(PREPARED_OUTCOMES["real_hit_stable"])
        if maxt_p > alpha and omnibus_p <= alpha:
            return dict(PREPARED_OUTCOMES["omnibus_only_diffuse"])
    return dict(PREPARED_OUTCOMES["atlas_design_choices"])


def validate_audit_result(record: dict[str, Any]) -> None:
    """Validate an audit result dict against the contract schema and scientific guardrails."""
    missing = _REQUIRED_KEYS - set(record)
    if missing:
        raise ValueError(f"audit result missing required keys: {sorted(missing)}")
    if record["schema_version"] != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"unexpected schema_version: {record['schema_version']!r}")
    if record["cohort"] not in {"upmc", "charville"}:
        raise ValueError(f"unknown cohort: {record['cohort']!r}")
    if record["exploratory"] is not True:
        raise ValueError("audit result must set exploratory=True")
    if record["already_seen"] is not True:
        raise ValueError("audit result must set already_seen=True")
    disclosure = str(record["already_seen_disclosure"]).lower()
    if "already-seen" not in disclosure or "exploratory" not in disclosure:
        raise ValueError("already_seen_disclosure must state both 'already-seen' and 'exploratory'")
    if record["clinical_claims"] is not False:
        raise ValueError("audit result must set clinical_claims=False (no clinical claims)")
    outcome = record["prepared_outcome"]
    if not isinstance(outcome, dict) or outcome.get("key") not in PREPARED_OUTCOMES:
        raise ValueError(f"prepared_outcome must be one of {sorted(PREPARED_OUTCOMES)}")
    maxt = record["maxT"]
    omnibus = record["omnibus"]
    if not isinstance(maxt, dict) or "statistic" not in maxt or "p" not in maxt or "top_hits" not in maxt:
        raise ValueError("maxT block must contain statistic, p, and top_hits")
    if not isinstance(omnibus, dict) or "statistics" not in omnibus or "p" not in omnibus:
        raise ValueError("omnibus block must contain statistics and p")
    leakage = record["leakage_audit"]
    if not isinstance(leakage, dict) or "per_coverslip" not in leakage:
        raise ValueError("leakage_audit block must contain per_coverslip")
