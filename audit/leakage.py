"""Coverslip-vs-patient leakage audit (IDEA.md #3 and #17)."""
from __future__ import annotations

from typing import Any

import pandas as pd


def compute_leakage_audit(regions: pd.DataFrame) -> dict[str, Any]:
    """Measure how many patients on each coverslip also appear in other coverslips."""
    if regions.empty:
        return {
            "total_patients": 0,
            "patients_spanning_multiple_coverslips": 0,
            "mean_leakage_fraction": 0.0,
            "min_leakage_fraction": 0.0,
            "max_leakage_fraction": 0.0,
            "per_coverslip": [],
        }
    coverslips = sorted(regions["coverslip"].astype(str).unique())
    patient_coverslips = regions.groupby("patient_id")["coverslip"].nunique()
    spanning = int((patient_coverslips > 1).sum())

    by_cov = {
        c: set(regions.loc[regions["coverslip"].astype(str) == c, "patient_id"].astype(str))
        for c in coverslips
    }
    per_coverslip = []
    fractions = []
    for c in coverslips:
        here = by_cov[c]
        others = set().union(*(by_cov[k] for k in coverslips if k != c)) if len(coverslips) > 1 else set()
        shared = len(here & others)
        n_patients = len(here)
        frac = float(shared / n_patients) if n_patients else 0.0
        fractions.append(frac)
        per_coverslip.append({
            "coverslip": c,
            "n_patients": n_patients,
            "shared_patients": shared,
            "leakage_fraction": frac,
        })

    return {
        "total_patients": int(len(patient_coverslips)),
        "patients_spanning_multiple_coverslips": spanning,
        "mean_leakage_fraction": float(sum(fractions) / len(fractions)),
        "min_leakage_fraction": float(min(fractions)),
        "max_leakage_fraction": float(max(fractions)),
        "per_coverslip": per_coverslip,
    }
