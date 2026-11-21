"""R-16 pre-registration: the Helsinki candidate 'pEMT tumour cells bordered by fibroblasts' in UPMC terms.

Committed before any search ran and before any outcome was opened. It was written from UPMC cell
types and marker names only (raw_data/*.cell_types.csv and *.expression.csv headers); no label or
outcome file was read. Editing RULE changes RULE_SHA256, and audit/test_rediscovery_definition.py fails.

Source: Punovuori et al., Cell 187:7267 (2024). pEMT epithelium was defined by Slug and Vimentin
(with K14), and the poor-survival stroma was CAF-enriched. Survival was about 21% at five years for
pEMT with CAF-enriched stroma, and about 90% for pEMT with immune-enriched stroma.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

# Every UPMC cell type on disk (17), sorted as core.cohorts.load_cells sorts them.
UPMC_TYPES = sorted(["APC", "B cell", "CD4 T cell", "CD8 T cell", "Granulocyte", "Lymph vessel", "Macrophage",
                     "Naive immune cell", "Stromal / Fibroblast", "Tumor", "Tumor (CD15+)", "Tumor (CD20+)",
                     "Tumor (CD21+)", "Tumor (Ki67+)", "Tumor (Podo+)", "Unassigned", "Vessel"])
# Every UPMC CODEX marker on disk (expression.csv columns other than CELL_ID). R-17 uses M = 22; the plan's ~40 was an estimate.
UPMC_MARKERS = ["CD11b", "CD14", "CD15", "CD163", "CD20", "CD21", "CD31", "CD34", "CD3e", "CD4", "CD45", "CD45RA",
                "CD45RO", "CD68", "CD8", "CollagenIV", "HLA-DR", "Ki67", "PanCK", "Podoplanin", "Vimentin", "aSMA"]
M = len(UPMC_MARKERS)

RULE = {
    "id": "R-16 pEMT tumour bordered by fibroblasts (UPMC)",
    "expressible": True,
    # The one primary candidate is a column of the v1 grammar, so its rank among all candidates is well defined.
    "feature": "Tumor (Podo+)|Stromal / Fibroblast|25um|fraction",
    "focal": "Tumor (Podo+)",
    "neighbour": "Stromal / Fibroblast",
    "radius_um": 25,
    "statistic": "fraction",
    "direction": "higher is worse: hazard ratio > 1; a top rank with HR < 1 counts as not rediscovered",
    "pEMT_proxy": "Podoplanin-positive tumour cells (UPMC type 'Tumor (Podo+)'). Podoplanin is a pEMT-programme "
                  "gene in HNSCC. Slug is not in the UPMC panel. Vimentin is, but the grammar has no "
                  "marker-gated focal type, so Vimentin+ tumour is not a candidate column",
    "fibroblast_proxy": "UPMC type 'Stromal / Fibroblast'. aSMA is a marker, not a type, so it stays out of the primary candidate",
    "deviations": ["The plan's 26 um becomes 25 um, the nearest grammar radius",
                   "The plan's '>30% fibroblasts' threshold becomes the continuous fraction the screen tests"],
    "outcomes_opened": False,
}
RULE_SHA256 = hashlib.sha256(json.dumps(RULE, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---- A-16.2, locked half: the scorer. Nothing below changes RULE or RULE_SHA256. ----
SCORER_DOC = """A-16.2 (locked half): score the pre-registered R-16 candidate once on the locked UPMC patients.

The rule (feature and direction) is frozen in audit/rediscovery.py at RULE_SHA256 (#61). The scorer computes
that one feature on locked patients only and runs one pre-specified test:
    Cox proportional hazards, the single feature, strata = coverslip group, Breslow ties (statsmodels PHReg);
    H0: HR <= 1 against H1: HR > 1 (the rule's direction), one-sided Wald p, alpha 0.05.
It also gives Harrell's C-index with the feature as the risk score (higher = worse).
Every number is labelled exploratory: the outcomes were already seen in the literature and the split is internal.

Before the test, Preflight's own power is computed without outcomes: the same test, repeated on the locked
patients' own feature values and strata, with planted times at HR EFFECT_HR_PER_SD per SD and exactly the
design's event count. Below MIN_POWER the result is UNMEASURED (underpowered), whatever the p-value.

The real run (python -m audit.rediscovery) opens the locked outcomes once, only with --open-locked and --events,
and refuses to overwrite a result.
"""
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "upmc-rediscovery.json"
# The pre-registration hash (audit/test_rediscovery_definition.py). The scorer refuses any other rule.
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
    head = {"rule_sha256": sha, "feature": rule["feature"], "test": SCORER_DOC.split("\n\n")[1].strip(),
            "power": {"value": pw, "effect_hr_per_sd": EFFECT_HR_PER_SD, "design_events": n_events,
                      "alpha": ALPHA, "min_power": MIN_POWER, "label": "SIMULATED, no outcomes read"}}
    res = score_feature(x, cells["time"], cells["event"], cells["strata"], rule)
    if pw < MIN_POWER and res.get("status") == "MEASURED":
        res = {**res, "status": "UNMEASURED",
               "reason": f"underpowered: Preflight power {pw:.0%} < {MIN_POWER:.0%} at HR {EFFECT_HR_PER_SD:g} per SD "
                         f"with {n_events} events; the test numbers below cannot confirm or refute the rule"}
    return {**head, **res}


def discovery_rank(z, names: list[str], rule: dict) -> dict:
    """The rule's rank by |z| among every candidate (1 = strongest; ties share the better rank).
    The rule's direction needs z > 0: a top rank with HR < 1 is not a rediscovery (RULE["direction"])."""
    z = np.asarray(z, float)
    j = names.index(rule["feature"])
    return {"rank": int(1 + np.sum(np.abs(z) > abs(z[j]))), "of": len(names), "z": float(z[j]),
            "direction_ok": bool(z[j] > 0), "label": f"MEASURED on discovery patients only; {LABEL}"}


def rank_on_discovery(cells: dict, rule: dict, clinical=None) -> dict:
    """Run the R-15 screen (stats.screen, no resampling) over every v1 grammar column and rank the rule's."""
    from core.grammar import feature_names, grammar
    from core.truth import truth

    try:
        from stats.screen import screen  # R-15; never reimplemented here
    except ImportError:
        return {"status": "UNMEASURED", "reason": "the R-15 screen (stats.screen) is not on this branch"}
    names = feature_names(grammar(cells["type_names"]))
    s = screen(truth(cells), cells["time"], np.asarray(cells["event"], float), clinical, cells["strata"],
               n_perm=0, names=names)
    return {"status": "MEASURED", "screen": s["metadata"]["statistic"], "clinical": s["metadata"]["clinical"]["mode"],
            "patients": s["metadata"]["patients"], **discovery_rank(s["z"], names, rule)}


def _hpv(cells: dict, data_dir) -> np.ndarray:
    """Clinical C [P, 1] for UPMC: hpvstatus per patient from the labels (R-15), NaN when 'NA'."""
    import pandas as pd

    lab = pd.read_csv(Path(data_dir) / "upmc_labels.csv", dtype=str, keep_default_na=False)
    by_region = dict(zip(lab.region_id, lab.hpvstatus))
    C = np.full((len(cells["time"]), 1), np.nan)
    for r, p in zip(cells["region_names"], cells["patient_of_region"]):
        C[p, 0] = np.nan if by_region[r] == "NA" else float(by_region[r])
    return C


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=SCORER_DOC.splitlines()[0])
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
    from core.cohorts import DATA_DIR, load

    if RULE_SHA256 != PREREG_SHA256:
        print("refusing: audit/rediscovery.py RULE differs from the pre-registration", file=sys.stderr)
        return 1
    disc = load("upmc")  # the discovery side (the loader's default); the search is frozen before the locked set opens
    result = {"cohort": "upmc", "rule_sha256": RULE_SHA256, "label": LABEL,
              "discovery": rank_on_discovery(disc, RULE, _hpv(disc, DATA_DIR)),
              "locked": score_locked(load("upmc", split="locked"), RULE, RULE_SHA256, a.events)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
