"""Implementation A, the CPU baseline (R-14): one full statsmodels PHReg Cox fit per candidate column.

For each candidate feature j: a stratified Cox model (Breslow ties) on [C, X_j] over the patients with a
finite X_j and complete C, giving X_j's coefficient, SE and Wald p. The fits run in a
concurrent.futures.ProcessPoolExecutor. This is "the standard way": a full fit per candidate, so its
outputs differ from R-15's score screen and a ratio of the two is a workflow comparison, never an
equal-work speed-up.

Run (after make data): uv run python -m bench.impl_a --output bench/results/impl-a-<cpu>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time as clock
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.duration.hazard_regression import PHReg

_SHARED: dict = {}


def fit_one(time, event, strata, C, x) -> tuple[float, float, float, int]:
    """(coef, se, Wald p, n) for x in a stratified Breslow Cox fit on [C, x]; NaN when x cannot be fitted."""
    ok = np.isfinite(x) & np.isfinite(C).all(1)
    n = int(ok.sum())
    if n < 2 or np.ptp(x[ok]) == 0 or event[ok].sum() == 0:
        return np.nan, np.nan, np.nan, n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence and overflow warnings: the result is reported as it is
        try:
            r = PHReg(time[ok], np.column_stack([C[ok], x[ok]]), status=event[ok], strata=strata[ok],
                      ties="breslow").fit()
        except (np.linalg.LinAlgError, ValueError):
            return np.nan, np.nan, np.nan, n
    return float(r.params[-1]), float(r.bse[-1]), float(r.pvalues[-1]), n


def _init(time, event, strata, C, X):
    _SHARED.update(time=time, event=event, strata=strata, C=C, X=X)


def _fit_column(j):
    s = _SHARED
    return fit_one(s["time"], s["event"], s["strata"], s["C"], s["X"][:, j])


def fit_all(time, event, strata, C, X, workers: int) -> dict:
    """Every candidate column of X fitted in a ProcessPoolExecutor of `workers` processes, in column order."""
    data = (np.asarray(time, float), np.asarray(event, float), np.asarray(strata), np.asarray(C, float).reshape(len(time), -1),
            np.asarray(X, float))
    J = data[4].shape[1]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=data) as ex:
        rows = list(ex.map(_fit_column, range(J), chunksize=max(1, J // (8 * workers))))
    coef, se, p, n = (np.array(c) for c in zip(*rows)) if rows else (np.zeros(0),) * 4
    return {"coef": coef, "se": se, "p": p, "n": n.astype(int)}


def fingerprint(a) -> str:
    a = np.ascontiguousarray(a, np.float64)
    return hashlib.sha256(repr(a.shape).encode() + a.tobytes()).hexdigest()


def fingerprints(patient_ids, time, event, strata, C, X) -> dict:
    rows = hashlib.sha256(json.dumps(list(map(str, patient_ids))).encode()
                          + fingerprint(np.column_stack([time, event, strata])).encode()).hexdigest()
    return {"patient_rows": rows, "X": fingerprint(X), "C": fingerprint(C)}


def compare(a: dict, b: dict) -> dict:
    """Wall-time ratio of two runs on identical inputs; refuses runs whose three fingerprints differ."""
    if a["fingerprints"] != b["fingerprints"]:
        raise ValueError("runs differ in patient rows, X or C: not comparable")
    return {"label": "workflow comparison", "ratio": a["wall_s"] / b["wall_s"],
            "note": "different outputs (full Cox fits vs a score screen): not an equal-work speed-up"}


def clinical(cohort: str, patient_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """C per patient: UPMC [HPV status] from the labels' hpvstatus; Charville none (zero-covariate mode)."""
    if cohort != "upmc":
        return np.zeros((len(patient_ids), 0)), []
    from core import cohorts

    reg = cohorts.load_regions(cohort)
    lab = pd.read_csv(cohorts.DATA_DIR / "upmc_labels.csv", dtype=str, keep_default_na=False)
    hpv = reg.merge(lab[["region_id", "hpvstatus"]], on="region_id", validate="one_to_one")
    hpv = pd.to_numeric(hpv.hpvstatus, errors="coerce").groupby(hpv.patient_id)
    if (hpv.nunique() > 1).any():
        raise ValueError("a patient has conflicting HPV status across regions")
    return hpv.first().reindex(patient_ids).to_numpy(float)[:, None], ["hpv"]


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines() if Path("/proc/cpuinfo").exists() else []:
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _num(v):
    return float(v) if np.isfinite(v) else None  # strict JSON: an unfittable candidate is null


def measure(cohort, patient_ids, time, event, strata, C, c_names, X, names, cores: int) -> dict:
    """Time fit_all with 1 worker and with `cores` workers on one cohort's arrays; results must be identical."""
    args = (np.asarray(time, float), np.asarray(event, float), np.asarray(strata), np.asarray(C, float), np.asarray(X, float))
    out = {"cohort": cohort, "patients": len(time), "events": int(np.sum(event)), "candidates": X.shape[1],
           "clinical": list(c_names), "fingerprints": fingerprints(patient_ids, *args), "timings": []}
    res = None
    for w in sorted({1, cores}):
        before = os.getloadavg()
        t0 = clock.perf_counter()
        r = fit_all(*args, workers=w)
        wall = clock.perf_counter() - t0
        if res is not None:  # the A-14.1 property on real data: the worker count never changes a result
            for k in r:
                np.testing.assert_array_equal(r[k], res[k])
        res = r
        out["timings"].append({"workers": w, "wall_s": wall, "label": "MEASURED",
                               "load_avg_1m_before": before[0], "load_avg_1m_after": os.getloadavg()[0]})
    out["wall_s"] = out["timings"][-1]["wall_s"]
    out["pool_speedup_vs_1_worker"] = out["timings"][0]["wall_s"] / out["wall_s"]
    out["fitted"] = int(np.isfinite(res["p"]).sum())
    out["outputs"] = [{"feature": f, "coef": _num(a), "se": _num(b), "wald_p": _num(d), "n": int(e)}
                      for f, a, b, d, e in zip(names, res["coef"], res["se"], res["p"], res["n"])]
    return out


def run_cohort(cohort: str, cores: int) -> dict:
    from core import cohorts
    from core.grammar import feature_names, grammar
    from core.truth import truth

    c = cohorts.load(cohort)
    X = truth(c)
    C, c_names = clinical(cohort, c["patient_names"])
    names = feature_names(grammar(c["type_names"]))
    return measure(cohort, c["patient_names"], c["time"], c["event"], c["strata"], C, c_names, X, names, cores)


def report(cohorts: list[dict], cores: int) -> dict:
    return {
        "what": "R-14 Implementation A: one statsmodels PHReg Cox fit per candidate (clinical C plus the candidate), "
                "stratified by coverslip group, Breslow ties, complete cases per candidate",
        "outputs_timed": "coefficient, SE and Wald p-value per candidate",
        "comparison": "workflow: quote against R-15's score screen only as a workflow comparison, never as the "
                      "equal-work speed-up, and only for runs with identical fingerprints",
        "timing": "wall time of fit_all (pool start, all fits, results back); excludes loading and the feature build",
        "cpu_model": cpu_model(), "cores_used": cores,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "statsmodels": statsmodels.__version__},
        "cohorts": cohorts,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", nargs="+", default=["charville", "upmc"])
    ap.add_argument("--cores", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    rep = report([run_cohort(c, a.cores) for c in a.cohort], a.cores)
    Path(a.output).write_text(json.dumps(rep, indent=1, allow_nan=False) + "\n")
    for c in rep["cohorts"]:
        print(f"{c['cohort']}: {c['patients']} patients, {c['candidates']} candidates ({c['fitted']} fitted), "
              + ", ".join(f"{t['workers']} workers {t['wall_s']:.2f} s" for t in c["timings"]))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
