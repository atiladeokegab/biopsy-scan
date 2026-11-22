"""Why the external baseline agrees with Preflight on 14 of 50 top-1 picks, and agreement on a matched config.

Same studies, seeds and visible cells as bench/external_baseline.py (#132). Per study it scores one patient matrix
under several configurations and compares each with Preflight's reference (our grammar X, stratified Cox score z):
  matched        squidpy.gr.spatial_neighbors's radius graph (per region, no self edges) for the neighbour counts,
                 then the grammar's statistic written out by hand on that graph (per-cell fraction, enrichment =
                 fraction / region share of j, focal-cell mean, regions pooled), scored with lifelines' own Efron
                 gradient and Hessian at beta = 0 summed over strata: the stratified Cox score z.
  matched_wald   the matched X with lifelines' default per-feature CoxPHFitter Wald z (strata as before).
  pooled_frac    our X with the fraction slots replaced by squidpy's pooled edge share (nhood_enrichment count share).
  squidpy_*      squidpy's nhood_enrichment features as #132 ran them (pooled share + 1,000-permutation z), scored
                 with our score test; and the two halves swapped one at a time into our X.
Top-1 is argmax |z| over each method's own finite features (as external_baseline.agreement).
"""
from __future__ import annotations

import warnings

import numpy as np

from bench import external_baseline as eb

_S = eb._S


def graph_counts(looks: list, K: int) -> list[np.ndarray]:
    """Per radius, (n, K) neighbour-type counts from squidpy's radius graph; cells in looks' order."""
    import anndata as ad
    import pandas as pd
    import squidpy as sq
    from core.grammar import RADII_UM
    xy = np.concatenate([x for _, x, _ in looks])
    ty = np.concatenate([t for _, _, t in looks])
    onehot = np.eye(K)[ty]
    obs = pd.DataFrame({"region": pd.Categorical(np.concatenate([[str(i)] * len(t) for i, (_, _, t) in enumerate(looks)]))},
                       index=[str(i) for i in range(len(ty))])
    a = ad.AnnData(X=np.zeros((len(ty), 1), np.float32), obs=obs, obsm={"spatial": xy})
    out = []
    for rad in RADII_UM:
        sq.gr.spatial_neighbors(a, coord_type="generic", radius=float(rad), library_key="region", delaunay=False,
                                set_diag=False)
        A = a.obsp["spatial_connectivities"]
        out.append(np.asarray((A != 0).astype(np.float64) @ onehot))
    return out


def graph_features(looks: list, focal: list[int], K: int) -> tuple[np.ndarray, np.ndarray]:
    """One patient's grammar row from squidpy's graph, and the same row with squidpy's pooled edge share as fraction.

    Written independently of core.truth.region_sums: fraction_i = n_ij / n_i (0 if n_i = 0), enrichment_i =
    fraction_i / p_j with p_j the share of j among the region's visible cells (0 if p_j = 0), patient feature =
    mean over the patient's focal cells of type t (NaN if none). Pooled fraction: sum_i n_ij / sum_i n_i over those cells.
    """
    from core.grammar import RADII_UM
    T, R = len(focal), len(RADII_UM)
    out = np.full((T, K, R, 2), np.nan)
    pooled = out.copy()
    ty = np.concatenate([t for _, _, t in looks])
    if not len(ty):
        return out.reshape(-1), pooled.reshape(-1)
    region = np.concatenate([[i] * len(t) for i, (_, _, t) in enumerate(looks)]).astype(int)
    share = np.stack([np.bincount(t, minlength=K) / max(len(t), 1) for _, _, t in looks])[region]  # (n, K)
    for r, C in enumerate(graph_counts(looks, K)):
        tot = C.sum(1, keepdims=True)
        frac = np.divide(C, tot, out=np.zeros_like(C), where=tot > 0)
        enr = np.divide(frac, share, out=np.zeros_like(frac), where=share > 0)
        for ti, t in enumerate(focal):
            sel = ty == t
            if not sel.any():
                continue
            s = C[sel].sum(0)
            out[ti, :, r, 0], out[ti, :, r, 1] = frac[sel].mean(0), enr[sel].mean(0)
            pooled[ti, :, r, 0], pooled[ti, :, r, 1] = (s / s.sum() if s.sum() > 0 else 0.0), out[ti, :, r, 1]
    return out.reshape(-1), pooled.reshape(-1)


def lifelines_score(X: np.ndarray, time: np.ndarray, event: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Stratified Cox score z = U / sqrt(I) at beta = 0 per column, from lifelines' Efron gradient and Hessian.

    lifelines exposes no score test for CoxPHFitter; this calls the Efron routine of the model CoxPHFitter fits by
    default (SemiParametricPHFitter._get_efron_values_single) per stratum (data sorted on T, as it requires), the routine its Newton-Raphson fit starts from. Rows with NaN dropped
    per column, as lifelines_screen. Efron equals Breslow when no event times tie.
    """
    import pandas as pd
    from lifelines.fitters.coxph_fitter import SemiParametricPHFitter
    f = SemiParametricPHFitter()
    z = np.full(X.shape[1], np.nan)
    order = np.argsort(time, kind="stable")
    X, time, event, strata = X[order], time[order], event[order], np.asarray(strata)[order]
    for c in range(X.shape[1]):
        ok = np.isfinite(X[:, c])
        U = I = 0.0
        for g in np.unique(strata[ok]):
            m = ok & (strata == g)
            x = X[m, c] - X[m, c].mean()  # the score is shift-invariant within a stratum; centring is lifelines' too
            h, grad, _ = f._get_efron_values_single(pd.DataFrame(x[:, None]), pd.Series(time[m]), pd.Series(event[m]),
                                                    pd.Series(np.ones(m.sum())), None, np.zeros(1))
            U, I = U + grad[0], I - h[0, 0]
        if I > 0:
            z[c] = U / np.sqrt(I)
    return z


def compare(X: np.ndarray, z: np.ndarray, Xo: np.ndarray, zo: np.ndarray) -> dict:
    """external_baseline.agreement plus the largest feature difference and the signed z rank correlation."""
    from scipy.stats import spearmanr
    a = eb.agreement(X, Xo, z, zo)
    ok = np.isfinite(z) & np.isfinite(zo)
    both = np.isfinite(X) & np.isfinite(Xo)
    return {**a, "x_max_abs_diff": float(np.abs(X - Xo)[both].max()) if both.any() else None,
            "z_max_abs_diff": float(np.abs(z - zo)[ok].max()) if ok.any() else None,
            "z_spearman": float(spearmanr(z[ok], zo[ok])[0]) if ok.sum() > 2 else None}


def gap(zo: np.ndarray) -> float:
    """|z| of our top feature minus our second: how close the top-1 race was."""
    a = np.sort(np.abs(zo[np.isfinite(zo)]))[::-1]
    return float(a[0] - a[1])


def one_study(job: tuple[str, int, bool, bool]) -> dict:
    stream, i, wald, orig = job
    from bench import design_batch as db
    from stats import screen_batch
    s = _S["studies"][stream][i]
    names, focal, K = _S["names"], _S["focal"], _S["K"]
    score = lambda X: screen_batch(X, s["time"][None], s["event"][None], strata=s["strata"], backend="numpy")["z"][0]
    sc = eb.study_cells(_S["cells"], db.design(), s["image_seed"])
    Xo = eb.our_features(sc, focal, K)
    zo = score(Xo)
    Xg, Xp = (np.stack(a) for a in zip(*(graph_features(looks, focal, K) for looks in sc)))
    Xpf = Xo.copy()
    Xpf[:, 0::2] = Xp[:, 0::2]  # squidpy's pooled share in the fraction slots, our enrichment
    rec = {"stream": stream, "i": i, "x_sha_ours": db.sha(Xo), "ours_top": int(np.nanargmax(np.abs(zo))),
           "ours_gap": gap(zo),
           "configs": {"matched": compare(Xg, lifelines_score(Xg, s["time"], s["event"], s["strata"]), Xo, zo),
                       "ours_x_lifelines_score": compare(Xo, lifelines_score(Xo, s["time"], s["event"], s["strata"]), Xo, zo),
                       "pooled_frac": compare(Xpf, score(Xpf), Xo, zo)}}
    if wald:
        rec["configs"]["matched_wald"] = compare(Xg, eb.lifelines_screen(Xg, s["time"], s["event"], s["strata"])["z"], Xo, zo)
    if orig:
        Xs = np.stack([eb.squidpy_features(looks, names, focal, seed=int(s["image_seed"]) % 2**32) for looks in sc])
        Xe = Xo.copy()
        Xe[:, 1::2] = Xs[:, 1::2]
        rec["configs"].update({"squidpy_x_our_score": compare(Xs, score(Xs), Xo, zo),
                               "squidpy_z_enrichment_only": compare(Xe, score(Xe), Xo, zo)})
        rec["squidpy_nan"] = int(np.isnan(Xs).sum() - np.isnan(Xo).sum())
    return rec


def run(cells: dict, plan: dict, wald: int, orig: int, workers: int = 2) -> dict:
    """Every planned study; Wald on the first `wald` and squidpy's original features on the first `orig` per stream."""
    import multiprocessing as mp
    info = eb.prepare(cells, plan)
    jobs = [(st, i, i < wald, i < orig) for st, n in plan.items() for i in range(n)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if workers == 1:
            recs = [one_study(j) for j in jobs]
        else:
            with mp.get_context("fork").Pool(workers) as pool:
                recs = pool.map(one_study, jobs, chunksize=1)
    return {"records": recs, **info}


def summarize(recs: list) -> dict:
    """Per config and stream: top-1 matches, median top-10 overlap, median z Spearman, worst feature difference."""
    out = {}
    names = sorted({c for r in recs for c in r["configs"]})
    for c in names:
        out[c] = {}
        for st in ("null", "alternative", "all"):
            rs = [r["configs"][c] for r in recs if c in r["configs"] and st in ("all", r["stream"])]
            if not rs:
                continue
            med = lambda k: float(np.median([x[k] for x in rs if x[k] is not None])) if any(x[k] is not None for x in rs) else None
            out[c][st] = {"top1_same": {"k": sum(bool(x["top1_same"]) for x in rs), "of": len(rs)},
                          "top10_overlap_median": med("top10_overlap"), "z_spearman_median": med("z_spearman"),
                          "z_spearman_abs_median": med("z_spearman_abs"),
                          "fraction_spearman_median": med("fraction_spearman"),
                          "enrichment_spearman_median": med("enrichment_spearman"),
                          "x_max_abs_diff": max((x["x_max_abs_diff"] or 0.0) for x in rs),
                          "z_max_abs_diff": max((x["z_max_abs_diff"] or 0.0) for x in rs)}
    return out


def main(studies: int, wald: int, orig: int, workers: int) -> None:
    """Run locally and write bench/results/external-agreement-<utc>.json (tracked tree must be committed)."""
    import json
    import logging
    import os
    import platform
    import subprocess
    from datetime import datetime, timezone
    from importlib.metadata import version
    from pathlib import Path
    from time import perf_counter

    from core.cohorts import load
    logging.disable(logging.WARNING)
    repo = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit tracked changes first: source_commit must be the code that runs")
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")),
               platform.processor())
    start = datetime.now(timezone.utc)
    load0 = os.getloadavg()
    t = perf_counter()
    out = run(load("charville"), {"null": studies, "alternative": studies}, wald, orig, workers)
    wall = perf_counter() - t
    result = {
        "source_commit": git("rev-parse", "HEAD"),
        "labels": {"agreement": "MEASURED (same re-imaged cells as #132, deterministic: one repeat, spread 0 by "
                                "construction; squidpy's permutation z uses #132's per-study seed)",
                   "outcomes": "SIMULATED (atlas/simulation.py outcomes, as #113/#132)",
                   "wall_s": "MEASURED on a shared, loaded VM; not a speed figure"},
        "method": __doc__.strip(),
        "design": {"cohort": "charville", "studies_per_stream": studies, "wald_studies_per_stream": wald,
                   "squidpy_original_studies_per_stream": orig, "streams": {"null": 202, "alternative": 303},
                   "baseline_result": "bench/results/external-baseline-20261122T043823Z.json"},
        "hardware": {"cpu": cpu, "cores": os.cpu_count(), "workers": workers, "load_average_start": list(load0),
                     "load_average_end": list(os.getloadavg()), "platform": platform.platform()},
        "versions": {p: version(p) for p in ("squidpy", "lifelines", "pandas", "anndata", "numpy", "scipy")},
        "repeats": 1, "started_utc": start.isoformat(), "wall_s": wall,
        "plant_feature": out["plant_feature"], "summary": summarize(out["records"]), "records": out["records"],
    }
    path = repo / "bench" / "results" / f"external-agreement-{start:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps(result["summary"], indent=1))
    print(f"Saved {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=25, help="studies per stream")
    ap.add_argument("--wald", type=int, default=0, help="first N per stream also get lifelines' Wald fits")
    ap.add_argument("--orig", type=int, default=0, help="first N per stream also get squidpy's nhood_enrichment")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    main(a.studies, a.wald, a.orig, a.workers)
