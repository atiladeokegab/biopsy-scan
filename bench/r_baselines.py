"""Two public baselines on the Charville v2 screen: R survival (8 cores) and permGPU (A100), against ours.

Workload (results/charville-v2.json, #47): 159 patients, 38 events, 3,780 candidate patterns (11,880 NaN
cells, complete case per feature), 4 coverslip strata, 9,999 within-stratum outcome shuffles, statistic
z = U / sqrt(V): U the stratified Cox partial-likelihood score at beta = 0 (Breslow ties), V its exact
within-stratum permutation variance; single-step maxT.

Arms (all given the same X; ours and R also the same shuffle index matrix P, drawn in the reference's order):
  ours      Preflight's product form X'[P_1 r .. P_B r] (stats.permutation_maxT), extended to NaN columns: one
            GEMM per missingness mask, risk sets recomputed per shuffle on that mask's complete cases. NumPy.
  r         bench/r/survival_perm.R: survival::coxph null fits (strata, Breslow) for the residuals, the score
            and V vectorised with crossprod/rowsum, shuffles split over parallel::mclapply(mc.cores = 8).
  permgpu   permGPU 0.15 (Shterev et al. 2010, CRAN), test = "cox", B = 9,999. It has no strata and draws its
            own shuffles, so it runs the closest workload it can express: the same X, unstratified.
Parity: ours and r against the committed audit (maxT 3.957181854, p 0.0981) and each other (same P, so the
same p). permGPU against a NumPy transcription of its own kernel (see permgpu_stat); the ours-vs-permGPU ratio is a
different-workload comparison (no strata, its statistic, float32), and its parity FAILED in the measured run
(r-baselines-20261122T133118Z: worst column 3.5e-3 relative, against the 1e-3 tolerance set before the run).
"""
from __future__ import annotations

import numpy as np

from stats import reference

AUDIT = {"maxT": 3.9571818541499213, "p": 0.0981, "n_resamples": 9999, "seed": 0,
         "top": "Tumor 5|Macrophage|100um|PD1", "source": "results/charville-v2.json"}
Z_TOL = 1e-9  # ours vs r and vs the audit: |z| and null maxT, absolute (float64 both sides)
PERMGPU_TOL = 1e-3  # permGPU's single-precision statistic vs its float64 transcription, relative


def draws(strata, n_resamples: int, seed: int) -> np.ndarray:
    """P[B, n]: shuffle b gives patient i the outcome of patient P[b, i]; stats.reference's draw order."""
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    rng = np.random.default_rng(seed)
    P = np.empty((n_resamples, len(strata)), np.int32)
    for b in range(n_resamples):
        for ix in groups:
            P[b, ix] = rng.permutation(ix)
    return P


def ours(X, time, event, strata, P, block: int = 1024) -> dict:
    """Observed z and every shuffle's maxT, as reference.permutation_test, with the shuffles batched."""
    X, time, event, _ = reference._inputs(X, time, event, strata)
    strata = np.asarray(strata)
    masks, inverse = np.unique(np.isfinite(X).T, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    z = np.zeros(X.shape[1])
    null = np.zeros(len(P))
    for g, keep in enumerate(masks):
        cols, rows = np.flatnonzero(inverse == g), np.flatnonzero(keep)
        if not len(rows):
            continue
        s = strata[rows]
        sub = [np.flatnonzero(s == v) for v in np.unique(s)]
        r = reference._residuals(time[rows], event[rows], sub)
        obs, Xc = reference._score(X[np.ix_(rows, cols)], r, sub)
        z[cols] = obs["z"]
        live = [ix for ix in sub if len(ix) >= 2]
        Sxx = np.array([np.sum(Xc[ix] ** 2, axis=0) / (len(ix) - 1) for ix in live])  # [S, C]
        for start in range(0, len(P), block):
            Pb = P[start:start + block]
            if keep.all():  # risk sets unchanged: shuffling outcome pairs shuffles the residuals
                R = r[Pb]
            else:
                R = np.stack([reference._residuals(time[p][rows], event[p][rows], sub) for p in Pb])
            V = np.stack([np.sum(R[:, ix] ** 2, axis=1) for ix in live], axis=1) @ Sxx  # [b, C]
            U = R @ Xc
            Z = np.divide(U, np.sqrt(V), out=np.zeros_like(U), where=V > 0)
            null[start:start + len(Pb)] = np.maximum(null[start:start + len(Pb)], np.abs(Z).max(axis=1))
    return {"z": z, "null_maxT": null, **summary(z, null)}


def write_inputs(d, X, time, event, strata, P) -> None:
    """The files bench/r/*.R read: column-major X (n x C), and P as n x B with column b = shuffle b."""
    n, C = X.shape
    (d / "dims.txt").write_text(f"{n} {C} {len(P)}\n")
    np.asarray(X, np.float64).ravel(order="F").tofile(d / "X.f64")
    np.asarray(time, np.float64).tofile(d / "time.f64")
    np.asarray(event, np.float64).tofile(d / "event.f64")
    np.asarray(strata, np.int32).tofile(d / "strata.i32")
    np.asarray(P, np.int32).tofile(d / "P.i32")  # C order [B, n] is Fortran order [n, B]


def summary(z, null) -> dict:
    """maxT, its p with the observed row counted (reference.permutation_test's rule) and the top column."""
    observed = float(np.max(np.abs(z)))
    tol = 16 * np.finfo(np.float64).eps * max(1.0, observed)
    return {"maxT": observed, "p": float((1 + np.sum(np.asarray(null) >= observed - tol)) / (len(null) + 1)),
            "top": int(np.argmax(np.abs(z)))}


def permgpu_stat(X, time, event) -> np.ndarray:
    """permGPU's test = "cox" statistic in float64, transcribed from SUstat_new (stand = 1): no strata,
    sum over deaths with finite x of (x_i - risk-set mean) squared over the summed risk-set variance. Its
    risk-set size counts patients whose x is NaN (Den), while the sums skip them: kept as written."""
    X = np.asarray(X, np.float64)
    time, event = np.asarray(time, np.float64), np.asarray(event)
    stat = np.zeros(X.shape[1])
    se = np.zeros(X.shape[1])
    finite = np.isfinite(X)
    Xz = np.where(finite, X, 0.0)
    for i in np.flatnonzero(event == 1):
        at = time >= time[i]
        den = at.sum()
        nom, nom2 = Xz[at].sum(0), (Xz[at] ** 2).sum(0)
        ok = finite[i]
        stat += np.where(ok, Xz[i] - nom / den, 0.0)
        se += np.where(ok, nom2 / den - nom * nom / den / den, 0.0)
    return np.divide(stat * stat, se, out=np.zeros_like(stat), where=se > 0)


def permgpu_label(parity: dict) -> dict:
    """The label every ours-vs-permGPU ratio carries: what it compares and whether its parity held."""
    return {"comparison": "different workload: permGPU has no strata and runs its own statistic (Cox score chi2, "
                          "information variance, float32); ours runs the unstratified permutation-variance z",
            "parity": "PASSED" if parity["within_tol"] else
                      f"FAILED: worst column {parity['max_rel_diff_all']:.1e} relative vs the {parity['tol']:g} "
                      "tolerance set before the run"}


def spread(values) -> dict:
    v = [float(x) for x in values]
    return {"repeats_s": v, "median_s": float(np.median(v)), "min_s": min(v), "max_s": max(v),
            "spread_pct": 100 * (max(v) - min(v)) / float(np.median(v))}


def ratio(base: dict, fast: dict) -> dict:
    """Speed-up as median / median, with the range from the slowest-vs-fastest pairings of the repeats."""
    return {"median": base["median_s"] / fast["median_s"],
            "range": [base["min_s"] / fast["max_s"], base["max_s"] / fast["min_s"]]}
