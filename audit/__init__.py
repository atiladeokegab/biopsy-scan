"""Real-data already-seen audit package (Issue #6)."""
from __future__ import annotations

from audit.leakage import compute_leakage_audit, leakage_from_cells
from audit.runner import format_summary_table, leave_one_patient_out, run_cohort_audit
from audit.schema import (
    ALREADY_SEEN_DISCLOSURE,
    AUDIT_SCHEMA_VERSION,
    PREPARED_OUTCOMES,
    choose_prepared_outcome,
    validate_audit_result,
)

__all__ = [
    "ALREADY_SEEN_DISCLOSURE",
    "AUDIT_SCHEMA_VERSION",
    "PREPARED_OUTCOMES",
    "choose_prepared_outcome",
    "compute_leakage_audit",
    "format_summary_table",
    "leakage_from_cells",
    "leave_one_patient_out",
    "run_cohort_audit",
    "validate_audit_result",
]
