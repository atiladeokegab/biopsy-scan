"""A-16.2 (locked half): score the pre-registered R-16 candidate once on the locked UPMC patients.

The rule (feature and direction) is frozen in audit/rediscovery.py at RULE_SHA256 (#61). This module computes
that one feature on locked patients only and runs one pre-specified test:
    Cox proportional hazards, the single feature, strata = coverslip group, Breslow ties (statsmodels PHReg);
    H0: HR <= 1 against H1: HR > 1 (the rule's direction), one-sided Wald p, alpha 0.05.
It also gives Harrell's C-index with the feature as the risk score (higher = worse).
Every number is labelled exploratory: the outcomes were already seen in the literature and the split is internal.

Before the test, Preflight's own power is computed without outcomes: the same test, repeated on the locked
patients' own feature values and strata, with planted times at HR EFFECT_HR_PER_SD per SD and exactly the
design's event count. Below MIN_POWER the result is UNMEASURED (underpowered), whatever the p-value.

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
EFFECT_HR_PER_SD = 2.0  # the effect the power atlas plants (scripts/export_grid.py effect_hr_per_sd)
MIN_POWER = 0.5
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


def _cox(z_x, time, event, strata):
    """(log HR, se) of the stratified Breslow Cox fit on one standardised feature."""
    from statsmodels.duration.hazard_regression import PHReg

    fit = PHReg(time, z_x[:, None], status=event.astype(float), strata=strata, ties="breslow").fit(disp=False)
    return float(fit.params[0]), float(fit.bse[0])


def power(x, strata, n_events: int, hr_per_sd: float = EFFECT_HR_PER_SD, sims: int = 400, seed: int = 0) -> float:
    """Share of planted-effect studies the one test detects, from feature values and strata only (no outcomes).

    Each study draws exponential times with log-hazard log(hr_per_sd) * z(x) and censors every patient after the
    n_events-th death (type II censoring), so the event count matches the design. A failed fit counts as a miss.
    """
    x, strata = np.asarray(x, float), np.asarray(strata)
    keep = np.isfinite(x)
    x, strata = x[keep], strata[keep]
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    if not 2 <= n_events <= len(x) or not sd > 0:
        return 0.0
    z_x = (x - x.mean()) / sd
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        t = rng.exponential(1 / np.exp(np.log(hr_per_sd) * z_x))
        cut = np.sort(t)[n_events - 1]
        try:
            beta, se = _cox(z_x, np.minimum(t, cut), t <= cut, strata)
        except (np.linalg.LinAlgError, ValueError):
            continue
        hits += bool(np.isfinite(beta) and se > 0 and norm.sf(beta / se) < ALPHA)
    return hits / sims


def score_feature(x, time, event, strata, rule: dict) -> dict:
    """The one pre-registered test on one feature vector [P] (patients without a value are dropped)."""
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
        beta, se = _cox(z_x, time, event, strata)
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


def score_locked(cells: dict, rule: dict, sha: str, n_events: int) -> dict:
    """Compute the rule's feature from a contract cells dict (locked patients only), check power, then score it.

    n_events is the design's locked event count, given before the outcomes are read.
    """
    from core.grammar import feature_names, grammar
    from core.truth import truth

    if rule_sha256(rule) != sha:
        raise ValueError("rule does not match its recorded sha256: the pre-registration was edited")
    names = feature_names(grammar(cells["type_names"]))
    if rule["feature"] not in names:
        return {"rule_sha256": sha, "feature": rule["feature"], "status": "UNMEASURED",
                "reason": "the feature is not a column of this cohort's grammar"}
    x = truth(cells)[:, names.index(rule["feature"])]
    pw = power(x, cells["strata"], n_events)  # before any outcome is read
    head = {"rule_sha256": sha, "feature": rule["feature"], "test": __doc__.split("\n\n")[1].strip(),
            "power": {"value": pw, "effect_hr_per_sd": EFFECT_HR_PER_SD, "design_events": n_events,
                      "alpha": ALPHA, "min_power": MIN_POWER, "label": "SIMULATED, no outcomes read"}}
    res = score_feature(x, cells["time"], cells["event"], cells["strata"], rule)
    if pw < MIN_POWER and res.get("status") == "MEASURED":
        res = {**res, "status": "UNMEASURED",
               "reason": f"underpowered: Preflight power {pw:.0%} < {MIN_POWER:.0%} at HR {EFFECT_HR_PER_SD:g} per SD "
                         f"with {n_events} events; the test numbers below cannot confirm or refute the rule"}
    return {**head, **res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--open-locked", action="store_true", help="open the locked outcomes (once, on the lead's word)")
    ap.add_argument("--events", type=int, help="the locked event count from the split's design, for the power check")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if not a.open_locked:
        print("refusing: the locked patients open once, on the lead's word; pass --open-locked", file=sys.stderr)
        return 2
    if a.events is None:
        print("refusing: pass --events, the design's locked event count, so power is fixed before outcomes", file=sys.stderr)
        return 2
    if a.out.exists():
        print(f"refusing: {a.out} exists; the locked set has already been opened", file=sys.stderr)
        return 2
    from audit.rediscovery import RULE, RULE_SHA256
    from core.cohorts import load

    if RULE_SHA256 != PREREG_SHA256:
        print("refusing: audit/rediscovery.py RULE differs from the pre-registration", file=sys.stderr)
        return 1
    result = score_locked(load("upmc", split="locked"), RULE, RULE_SHA256, a.events)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"cohort": "upmc", "split": "locked", **result}, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
