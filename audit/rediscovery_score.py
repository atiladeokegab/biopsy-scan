"""A-16.2 (locked half): score the pre-registered R-16 candidate once on the locked UPMC patients.

The rule (feature and direction) is frozen in audit/rediscovery.py at RULE_SHA256 (#61). This module computes
that one feature on locked patients only and runs one pre-specified test:
    Cox proportional hazards, the single feature, strata = coverslip group, Breslow ties (statsmodels PHReg);
    H0: HR <= 1 against H1: HR > 1 (the rule's direction), one-sided Wald p, alpha 0.05.
It also gives Harrell's C-index with the feature as the risk score (higher = worse).
Every number is labelled exploratory: the outcomes were already seen in the literature and the split is internal.

The real run opens the locked outcomes once, only with --open-locked, and refuses to overwrite a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "upmc-rediscovery.json"
# The pre-registration hash from #61 (audit/test_rediscovery_definition.py). The scorer refuses any other rule.
PREREG_SHA256 = "a70424447055052d19610332a478fb46da68d87a9f2f30b350cc3c14a4bada1e"
ALPHA = 0.05
LABEL = "exploratory: outcomes already seen (Punovuori et al., Cell 2024); internal held-out split"


def rule_sha256(rule: dict) -> str:
    """Same canonical form as audit/rediscovery.py's RULE_SHA256."""
    return hashlib.sha256(json.dumps(rule, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def harrell_c(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C: over pairs where the earlier time is an event, the share where it had the higher risk.
    Risk ties count 1/2; pairs tied in time are not comparable. NaN when no pair is comparable."""
    num = den = 0.0
    for i in np.flatnonzero(event):
        later = time > time[i]
        den += later.sum()
        num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return num / den if den else float("nan")


def score_feature(x, time, event, strata, rule: dict) -> dict:
    """The one pre-registered test on one feature vector [P] (patients without a value are dropped)."""
    from statsmodels.duration.hazard_regression import PHReg

    if not rule.get("expressible", False):
        return {"status": "UNMEASURED", "reason": "the pre-registration says UPMC cannot express the candidate"}
    if not rule["direction"].startswith("higher is worse"):
        raise ValueError("this scorer implements only the pre-registered direction 'higher is worse'")
    x, time = np.asarray(x, float), np.asarray(time, float)
    event, strata = np.asarray(event, bool), np.asarray(strata)
    keep = np.isfinite(x)
    x, time, event, strata = x[keep], time[keep], event[keep], strata[keep]
    out = {"n_patients": int(keep.sum()), "n_dropped_no_focal_cells": int((~keep).sum()),
           "n_events": int(event.sum()), "label": LABEL}
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    if event.sum() < 2 or not sd > 0:
        return {**out, "status": "UNMEASURED",
                "reason": f"not estimable: {int(event.sum())} events, feature SD {sd:.3g} on the locked patients"}
    z_x = (x - x.mean()) / sd  # per locked-set SD: changes the HR's scale only, never z or p
    try:
        fit = PHReg(time, z_x[:, None], status=event.astype(float), strata=strata, ties="breslow").fit(disp=False)
        beta, se = float(fit.params[0]), float(fit.bse[0])
    except (np.linalg.LinAlgError, ValueError) as exc:
        return {**out, "status": "UNMEASURED", "reason": f"Cox fit failed: {exc}"}
    if not (np.isfinite(beta) and np.isfinite(se) and se > 0):
        return {**out, "status": "UNMEASURED", "reason": f"Cox fit did not converge (beta {beta}, se {se})"}
    z = beta / se
    p = float(norm.sf(z))
    return {**out, "status": "MEASURED", "feature_sd": float(sd),
            "hr_per_sd": float(np.exp(beta)), "hr_per_sd_ci95": [float(np.exp(beta - 1.959964 * se)),
                                                                  float(np.exp(beta + 1.959964 * se))],
            "log_hr_per_sd": beta, "se": se, "wald_z": z, "p_one_sided": p, "alpha": ALPHA,
            "locked_test_passes": bool(p < ALPHA),  # the rank half of "rediscovered" is on discovery (R-15)
            "c_index": harrell_c(x, time, event)}


def score_locked(cells: dict, rule: dict, sha: str) -> dict:
    """Compute the rule's feature from a contract cells dict (locked patients only) and score it."""
    from core.grammar import feature_names, grammar
    from core.truth import truth

    if rule_sha256(rule) != sha:
        raise ValueError("rule does not match its recorded sha256: the pre-registration was edited")
    names = feature_names(grammar(cells["type_names"]))
    if rule["feature"] not in names:
        return {"rule_sha256": sha, "feature": rule["feature"], "status": "UNMEASURED",
                "reason": "the feature is not a column of this cohort's grammar"}
    x = truth(cells)[:, names.index(rule["feature"])]
    return {"rule_sha256": sha, "feature": rule["feature"], "test": __doc__.split("\n\n")[1].strip(),
            **score_feature(x, cells["time"], cells["event"], cells["strata"], rule)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--open-locked", action="store_true", help="open the locked outcomes (once, on the lead's word)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if not a.open_locked:
        print("refusing: the locked patients open once, on the lead's word; pass --open-locked", file=sys.stderr)
        return 2
    if a.out.exists():
        print(f"refusing: {a.out} exists; the locked set has already been opened", file=sys.stderr)
        return 2
    from audit.rediscovery import RULE, RULE_SHA256
    from core.cohorts import load

    if RULE_SHA256 != PREREG_SHA256:
        print("refusing: audit/rediscovery.py RULE differs from the pre-registration", file=sys.stderr)
        return 1
    result = score_locked(load("upmc", split="locked"), RULE, RULE_SHA256)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"cohort": "upmc", "split": "locked", **result}, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
