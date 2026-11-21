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
from datetime import datetime, timezone
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


# ---- A-16.2: the scorer. Nothing below changes RULE or RULE_SHA256. ----
SCORER_DOC = """A-16.2: rank the frozen R-16 candidate on discovery, then score it once on the locked UPMC patients.

The rule (feature and direction) is frozen in audit/rediscovery.py at RULE_SHA256 (#61). The scorer computes
that one feature on locked patients only and runs one pre-specified test:
    Cox proportional hazards, the single feature, strata = coverslip group, Breslow ties (statsmodels PHReg);
    H0: HR <= 1 against H1: HR > 1 (the rule's direction), one-sided Wald p, alpha 0.05.
It also gives Harrell's C-index with the feature as the risk score (higher = worse).
Every number is labelled exploratory: the outcomes were already seen in the literature and the split is internal.

Before the test, Preflight's own power is simulated without outcomes (see power()). It is conditional model
power, not achieved power. Below MIN_POWER the result is UNMEASURED (underpowered), whatever the p-value.

The real run (python -m audit.rediscovery) checks the R-13 split API and manifest before any cohort read, ranks
on discovery with the R-15 screen and refuses to go on unless that is MEASURED, writes a durable opening record
tied to the rule and the locked-set hash, and only then reads the locked patients. It never opens them twice.
"""
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "upmc-rediscovery.json"
RESULTS = ROOT / "results"
# The pre-registration hash (audit/test_rediscovery_definition.py). The scorer refuses any other rule.
PREREG_SHA256 = "a70424447055052d19610332a478fb46da68d87a9f2f30b350cc3c14a4bada1e"
# The approved locked design: #62's manifest (cfa8a41) read with strict case identity (the loader default), per
# the lead's ruling: 16 locked case patients, 4 events (MEASURED in #62's description).
APPROVED_SPLIT = {"manifest_sha256": "05b38abae377177a097f2db0cd42188178c275a338aa44ec79c1be8ecd3969e3",
                  "identity": "case", "locked_patients": 16, "locked_events": 4}
ALPHA = 0.05
EFFECT_HR_PER_SD = 2.0  # the effect the power atlas plants (scripts/export_grid.py effect_hr_per_sd)
MIN_POWER = 0.5
POWER_SIMS, POWER_SEED = 400, 20261121
LABEL = "exploratory: outcomes already seen (Punovuori et al., Cell 2024); internal held-out split"


def rule_sha256(rule: dict) -> str:
    """Same canonical form as audit/rediscovery.py's RULE_SHA256."""
    return hashlib.sha256(json.dumps(rule, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def harrell_c(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C over comparable pairs: i had an event and j was still at risk after it (later time, or the same
    time but censored). Event-event time ties are not comparable; risk ties count 1/2. NaN when none is comparable."""
    num = den = 0.0
    for i in np.flatnonzero(event):
        at_risk = (time > time[i]) | ((time == time[i]) & ~event)
        den += at_risk.sum()
        num += (risk[i] > risk[at_risk]).sum() + 0.5 * (risk[i] == risk[at_risk]).sum()
    return num / den if den else float("nan")


class FitFailed(ValueError):
    """The Cox fit did not converge or gave a non-finite estimate."""


def _cox(z_x, time, event, strata):
    """(log HR, se) of the stratified Breslow Cox fit on one standardised feature; FitFailed unless it converged."""
    import warnings

    from statsmodels.duration.hazard_regression import PHReg
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", RuntimeWarning)  # overflow in exp() during the fit
            fit = PHReg(time, z_x[:, None], status=event.astype(float), strata=strata, ties="breslow").fit(disp=False)
    except (ConvergenceWarning, RuntimeWarning, np.linalg.LinAlgError, ValueError) as exc:
        raise FitFailed(f"{type(exc).__name__}: {exc}") from None
    beta, se = float(fit.params[0]), float(fit.bse[0])
    if not (getattr(fit, "mle_retvals", None) or {}).get("converged", True) or not (np.isfinite(beta) and np.isfinite(se) and se > 0):
        raise FitFailed(f"not converged (beta {beta}, se {se})")
    return beta, se


def power(x, strata, n_events: int, hr_per_sd: float = EFFECT_HR_PER_SD, sims: int = POWER_SIMS,
          seed: int = POWER_SEED) -> dict:
    """Conditional model power of the one test, from feature values and strata only (no outcomes).

    Model: every design patient, with or without the feature, gets an exponential time with log-hazard
    log(hr_per_sd) * z(x) (z = 0, the average hazard, when the feature is missing) and one common baseline across
    strata; all are censored after the n_events-th death (type II), so the design's event count is planted in the
    whole split. The analysis then drops patients without the feature, as the real test does, so the analysable
    events vary. A failed fit, or fewer than 2 analysable events, counts as a miss. Not achieved power under the
    real follow-up.
    """
    x, strata = np.asarray(x, float), np.asarray(strata)
    keep = np.isfinite(x)
    out = {"effect_hr_per_sd": hr_per_sd, "design_events": n_events, "alpha": ALPHA, "min_power": MIN_POWER,
           "sims": sims, "seed": seed, "n_design": len(x), "n_with_feature": int(keep.sum()),
           "n_missing_feature": int((~keep).sum()),
           "event_model": "design events planted over all design patients (missing feature at average hazard), "
                          "type II censoring, then patients without the feature dropped as in the real test",
           "label": "SIMULATED conditional model power, no outcomes read; not achieved power"}
    sd = x[keep].std(ddof=1) if keep.sum() > 1 else 0.0
    if not 2 <= n_events <= len(x) or not sd > 0:
        return {**out, "value": 0.0, "mean_analysable_events": 0.0}
    z_all = np.where(keep, (x - np.nanmean(x)) / sd, 0.0)
    rng = np.random.default_rng(seed)
    hits, kept_events = 0, 0
    for _ in range(sims):
        t = rng.exponential(1 / np.exp(np.log(hr_per_sd) * z_all))
        cut = np.sort(t)[n_events - 1]
        ev = (t <= cut)[keep]
        kept_events += int(ev.sum())
        if ev.sum() < 2:
            continue
        try:
            beta, se = _cox(z_all[keep], np.minimum(t, cut)[keep], ev, strata[keep])
        except FitFailed:
            continue
        hits += bool(norm.sf(beta / se) < ALPHA)
    return {**out, "value": hits / sims, "mean_analysable_events": kept_events / sims}


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
    except FitFailed as exc:
        return {**out, "status": "UNMEASURED", "reason": f"Cox fit failed: {exc}"}
    with np.errstate(over="ignore"):
        hr = np.exp([beta, beta - 1.959964 * se, beta + 1.959964 * se])
    if not np.isfinite(hr).all():
        return {**out, "status": "UNMEASURED", "reason": f"HR or its CI overflows (beta {beta:.3g}, se {se:.3g})"}
    z = beta / se
    p = float(norm.sf(z))
    c = harrell_c(x, time, event)
    return {**out, "status": "MEASURED", "feature_sd": float(sd),
            "hr_per_sd": float(hr[0]), "hr_per_sd_ci95": [float(hr[1]), float(hr[2])],
            "log_hr_per_sd": beta, "se": se, "wald_z": z, "p_one_sided": p, "alpha": ALPHA,
            "locked_test_passes": bool(p < ALPHA),  # the rank half of "rediscovered" is on discovery (R-15)
            "c_index": c if np.isfinite(c) else None,
            **({} if np.isfinite(c) else {"c_index_reason": "no comparable pair"})}


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
    head = {"rule_sha256": sha, "feature": rule["feature"], "test": SCORER_DOC.split("\n\n")[1].strip(), "power": pw}
    res = score_feature(x, cells["time"], cells["event"], cells["strata"], rule)
    if pw["value"] < MIN_POWER and res.get("status") == "MEASURED":
        res = {**res, "status": "UNMEASURED",
               "reason": f"underpowered: Preflight conditional power {pw['value']:.0%} < {MIN_POWER:.0%} at HR "
                         f"{EFFECT_HR_PER_SD:g} per SD with {n_events} design events; the test numbers below "
                         "cannot confirm or refute the rule"}
    return {**head, **res}


def discovery_rank(z, names: list[str], rule: dict, V=None, n=None) -> dict:
    """The rule's rank by |z| among the informative candidates (finite z, V > 0; 1 = strongest; ties share the
    better rank). The rule's direction needs z > 0: a top rank with HR < 1 is not a rediscovery (RULE["direction"]).
    An uninformative rule column is UNMEASURED, never ranked."""
    z = np.asarray(z, float)
    V = np.ones_like(z) if V is None else np.asarray(V, float)
    ok = np.isfinite(z) & np.isfinite(V) & (V > 0)
    j = names.index(rule["feature"])
    base = {"of_all": len(names), "uninformative": int((~ok).sum()),
            "n_rule": None if n is None else int(np.asarray(n)[j])}
    if not ok[j]:
        return {**base, "status": "UNMEASURED",
                "reason": f"the rule's column is uninformative on discovery (z {z[j]}, V {V[j]})"}
    return {**base, "status": "MEASURED", "rank": int(1 + np.sum(np.abs(z[ok]) > abs(z[j]))), "of": int(ok.sum()),
            "z": float(z[j]), "direction_ok": bool(z[j] > 0),
            "label": f"MEASURED on discovery patients only; {LABEL}"}


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
    return {"screen": s["metadata"]["statistic"], "clinical": s["metadata"]["clinical"]["mode"],
            "patients": s["metadata"]["patients"], **discovery_rank(s["z"], names, rule, s["V"], s["n"])}


def _hpv(cells: dict, data_dir) -> np.ndarray:
    """Clinical C [P, 1] for UPMC: hpvstatus per patient from the labels (R-15), NaN when 'NA'.
    Raises on a label other than 0, 1 or NA, and on a patient whose regions disagree."""
    import pandas as pd

    lab = pd.read_csv(Path(data_dir) / "upmc_labels.csv", usecols=["region_id", "hpvstatus"], dtype=str,
                      keep_default_na=False)
    bad = sorted(set(lab.hpvstatus) - {"0", "1", "NA"})
    if bad:
        raise ValueError(f"hpvstatus must be 0, 1 or NA; found {bad}")
    by_region = dict(zip(lab.region_id, lab.hpvstatus))
    C = np.full((len(cells["time"]), 1), np.nan)
    for r, p in zip(cells["region_names"], cells["patient_of_region"]):
        if by_region[r] == "NA":
            continue
        v = float(by_region[r])
        if np.isfinite(C[p, 0]) and C[p, 0] != v:
            raise ValueError(f"patient {p}: regions disagree on hpvstatus (region {r})")
        C[p, 0] = v
    return C


def _code_state() -> tuple[str, bool]:
    """(HEAD sha, whether tracked files differ from it): the locked set opens only from a committed, reviewed head."""
    import subprocess

    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout  # noqa: E731
    return git("rev-parse", "HEAD").strip(), bool(git("status", "--porcelain", "--untracked-files=no").strip())


def _split_preflight() -> str:
    """Check the R-13 split API and its manifest before any cohort read; returns the manifest sha256.
    Raises unless load() takes split=, the manifest and labels verify, and the manifest is the approved one."""
    import inspect

    from core import cohorts

    if "split" not in inspect.signature(cohorts.load).parameters or not {"discovery", "locked"} <= set(
            getattr(cohorts, "SPLITS", ())):
        raise RuntimeError("core.cohorts.load has no discovery/locked split (R-13 not on this branch)")
    if getattr(cohorts, "LOCK_FILE", None) is None:
        raise RuntimeError("core.cohorts.LOCK_FILE is unset: no locked split to honour")
    cohorts.locked_regions("upmc")  # raises if the manifest was edited or the labels differ from the locked ones
    sha = json.loads(Path(cohorts.LOCK_FILE).read_text())["sha256"]
    if sha != APPROVED_SPLIT["manifest_sha256"]:
        raise RuntimeError(f"locked manifest {sha[:12]} is not the approved split {APPROVED_SPLIT['manifest_sha256'][:12]}")
    return sha


def opening_record(rule_sha: str, manifest_sha: str) -> Path:
    """The durable once-only marker for this rule on this locked set, whatever --out says."""
    return RESULTS / f"upmc-locked-opened.{rule_sha[:16]}.{manifest_sha[:16]}.json"


def _refuse(msg: str) -> int:
    print(f"refusing: {msg}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=SCORER_DOC.splitlines()[0])
    ap.add_argument("--open-locked", action="store_true", help="open the locked outcomes (once, on the lead's word)")
    ap.add_argument("--events", type=int, help="the approved locked design's event count, for the power check")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if not a.open_locked:
        return _refuse("the locked patients open once, on the lead's word; pass --open-locked")
    if a.events != APPROVED_SPLIT["locked_events"]:
        return _refuse(f"--events must be the approved design's {APPROVED_SPLIT['locked_events']} locked events "
                       f"({APPROVED_SPLIT['identity']} identity), got {a.events}")
    if a.out.exists():
        return _refuse(f"{a.out} exists")
    if RULE_SHA256 != PREREG_SHA256:
        return _refuse("audit/rediscovery.py RULE differs from the pre-registration")
    sha, dirty = _code_state()
    if dirty:
        return _refuse(f"tracked files differ from {sha[:7]}; run only from the reviewed head")
    try:
        manifest_sha = _split_preflight()
    except (RuntimeError, ValueError, OSError) as exc:
        return _refuse(f"split preflight failed: {exc}")
    marker = opening_record(RULE_SHA256, manifest_sha)
    if marker.exists():
        return _refuse(f"{marker} exists: this rule has already opened this locked set")
    from core.cohorts import DATA_DIR, load

    disc_cells = load("upmc", split="discovery")
    discovery = rank_on_discovery(disc_cells, RULE, _hpv(disc_cells, DATA_DIR))
    if discovery.get("status") != "MEASURED":
        return _refuse(f"the discovery rank is not ready ({discovery.get('reason')}); the locked set stays closed")
    record = {"code_sha": sha, "rule_sha256": RULE_SHA256, "manifest_sha256": manifest_sha,
              "design": APPROVED_SPLIT, "opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "discovery": discovery}
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "x") as f:  # atomic create: a second opening fails here, before any locked read
        f.write(json.dumps(record, indent=1, allow_nan=False) + "\n")
    locked = score_locked(load("upmc", split="locked"), RULE, RULE_SHA256, a.events)
    result = {"cohort": "upmc", "code_sha": sha, "rule_sha256": RULE_SHA256, "manifest_sha256": manifest_sha,
              "opening_record": marker.name, "label": LABEL, "discovery": discovery, "locked": locked}
    text = json.dumps(result, indent=1, allow_nan=False) + "\n"
    with open(a.out, "x") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
