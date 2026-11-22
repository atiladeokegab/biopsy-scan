"""External baseline: the design point done the way a lab would with off-the-shelf tools (squidpy + lifelines).

Per study, from the same design, seeds and draws as #113 (bench/design_batch.py):
  cells     the visible, mis-called cells our re-imager sees (core.reimage_ref.draw, crop_mask, miscall), so both
            pipelines get the same cells; study_cells() is checked bit for bit against core.reimage_ref.reimage.
  squidpy   per drawn patient, one AnnData of its regions; per radius (25, 50, 100 um)
            sq.gr.spatial_neighbors(coord_type="generic", radius=r, library_key="region") and
            sq.gr.nhood_enrichment (default 1,000 permutations). It fills the same 630 slots as our grammar,
            (tumour type t, neighbour type j, radius, statistic): "fraction" = squidpy's edge count share
            count[t, j] / sum_j count[t, :], "enrichment" = its z-score. NaN where the patient has no visible t.
  lifelines CoxPHFitter per feature, strata = coverslip group, rows with NaN dropped; Bonferroni over all 630:
            the study is called when min p x 630 <= 0.05.
Timed per study inside one single-thread pool worker: draw, mis-call, AnnData, squidpy and lifelines. Our own X and
score z on the same cells are computed after the timer, for agreement only. Seconds per study = pool wall / N,
the basis of the fused path's 0.0419 s (#120: compute_wall_s / 3,000 studies, 8 workers).
"""
from __future__ import annotations

import warnings
from time import perf_counter

import numpy as np

from bench import design_batch as db

ALPHA = 0.05
N_PERMS = 1000  # squidpy's default for nhood_enrichment
FUSED_S_PER_STUDY = {"value": 0.0419, "source": "bench/results/modal-design-20261121T205536Z.json",
                     "basis": "fused compute_wall_s 125.73 / 3,000 studies, 8 single-thread workers (#120)"}

_S: dict = {}


def study_cells(cells: dict, design: dict, seed) -> list[list[tuple[int, np.ndarray, np.ndarray]]]:
    """Per drawn patient, per look: (region, visible xy, visible observed types), in reimage()'s draw order."""
    from core.reimage_ref import confusion_matrix, crop_mask, draw, miscall
    from core.truth import layout
    L = layout(cells)
    d = draw(cells, design, seed)
    M = confusion_matrix(L["K"], float(design.get("eps") or 0.0))
    out = []
    for row in d["looks"]:
        looks = []
        for r, crop, u in row:
            xy, ty = L["xy"](r), L["types"](r)
            vis = crop_mask(xy, crop)
            looks.append((r, xy[vis], miscall(ty, M, u)[vis]))
        out.append(looks)
    return out


def our_features(sc: list, focal: list[int], K: int) -> np.ndarray:
    """Our grammar's X from study_cells() (the truth layer on the visible cells); equals reimage() bit for bit."""
    from core.grammar import features
    from core.truth import R, region_sums, truth_counts
    S = np.zeros((len(sc), len(focal), K, R, 2))
    m = np.zeros((len(sc), len(focal)), np.int64)
    for i, looks in enumerate(sc):
        for _, xy, ty in looks:
            s_, m_ = region_sums(truth_counts(xy, ty, K), ty, np.ones(len(ty), bool), focal, K)
            S[i] += s_
            m[i] += m_
    return features(S, m)


def squidpy_features(looks: list, names: list[str], focal: list[int], seed: int) -> np.ndarray:
    """One patient's 630 features from squidpy, in the grammar's order ((t * K + j) * R + r) * 2 + s."""
    import anndata as ad
    import pandas as pd
    import squidpy as sq
    from core.grammar import RADII_UM
    K = len(names)
    xy = np.concatenate([x for _, x, _ in looks])
    ty = np.concatenate([t for _, _, t in looks])
    out = np.full((len(focal), K, len(RADII_UM), 2), np.nan)
    if not len(ty):
        return out.reshape(-1)
    obs = pd.DataFrame({"type": pd.Categorical(np.asarray(names)[ty], categories=names),
                        "region": pd.Categorical(np.concatenate([[str(r)] * len(t) for r, _, t in looks]))},
                       index=[str(i) for i in range(len(ty))])
    a = ad.AnnData(X=np.zeros((len(ty), 1), np.float32), obs=obs, obsm={"spatial": xy})
    present = np.bincount(ty, minlength=K) > 0
    for r, rad in enumerate(RADII_UM):
        sq.gr.spatial_neighbors(a, coord_type="generic", radius=float(rad), library_key="region", delaunay=False)
        z, count = sq.gr.nhood_enrichment(a, cluster_key="type", copy=True, n_perms=N_PERMS, seed=seed, n_jobs=1,
                                          show_progress_bar=False)
        for ti, t in enumerate(focal):
            if not present[t]:
                continue
            tot = count[t].sum()
            out[ti, :, r, 0] = count[t] / tot if tot > 0 else 0.0
            out[ti, :, r, 1] = z[t]
    return out.reshape(-1)


def lifelines_screen(X: np.ndarray, time: np.ndarray, event: np.ndarray, strata: np.ndarray) -> dict:
    """One stratified CoxPHFitter per column; z and p per feature (NaN where the fit fails or is undefined)."""
    import pandas as pd
    from lifelines import CoxPHFitter
    C = X.shape[1]
    z, p = np.full(C, np.nan), np.full(C, np.nan)
    failed = 0
    base = pd.DataFrame({"T": time, "E": event, "g": strata})
    for c in range(C):
        df = base.assign(x=X[:, c]).dropna()
        if len(df) < 2 or df["E"].sum() == 0 or np.ptp(df["x"].to_numpy()) == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row = CoxPHFitter().fit(df, "T", "E", strata=["g"]).summary.loc["x"]
            z[c], p[c] = float(row["z"]), float(row["p"])
        except Exception:  # lifelines' ConvergenceError and singular fits: a lab records them and moves on
            failed += 1
    return {"z": z, "p": p, "failed": failed, "fitted": int(np.isfinite(p).sum())}


def bonferroni(p: np.ndarray, m: int) -> dict:
    """The study's call: min p x m <= ALPHA over all m candidates (untestable ones count against the lab)."""
    finite = np.isfinite(p)
    if not finite.any():
        return {"min_p_adj": None, "top": None, "call": False}
    top = int(np.nanargmin(p))
    adj = min(1.0, float(p[top]) * m)
    return {"min_p_adj": adj, "top": top, "call": adj <= ALPHA}


def agreement(Xs: np.ndarray, Xo: np.ndarray, zs: np.ndarray, zo: np.ndarray) -> dict:
    """How far squidpy's features and lifelines' hits agree with ours on the same cells (not forced to)."""
    from scipy.stats import pearsonr, spearmanr
    out = {}
    for s, name in ((0, "fraction"), (1, "enrichment")):
        a, b = Xs.reshape(len(Xs), -1, 2)[..., s].ravel(), Xo.reshape(len(Xo), -1, 2)[..., s].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        out[f"{name}_pearson"] = float(pearsonr(a[ok], b[ok])[0]) if ok.sum() > 2 else None
        out[f"{name}_spearman"] = float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 2 else None
    nan_s, nan_o = np.isnan(Xs), np.isnan(Xo)
    out["nan_only_squidpy"], out["nan_only_ours"] = int((nan_s & ~nan_o).sum()), int((nan_o & ~nan_s).sum())
    ok = np.isfinite(zs) & np.isfinite(zo)
    out["z_spearman_abs"] = float(spearmanr(np.abs(zs[ok]), np.abs(zo[ok]))[0]) if ok.sum() > 2 else None
    top_s = [int(c) for c in np.argsort(-np.where(np.isfinite(zs), np.abs(zs), -1))[:10]]
    top_o = [int(c) for c in np.argsort(-np.where(np.isfinite(zo), np.abs(zo), -1))[:10]]
    out.update(top1_same=top_s[0] == top_o[0], top10_overlap=len(set(top_s) & set(top_o)),
               top_lifelines=top_s[0], top_ours=top_o[0])
    return out


def prepare(cells: dict, plan: dict) -> dict:
    """Outcomes for every planned study (as design_batch.prepare, which also gives the planted feature)."""
    info = db.prepare("lab", cells, plan)  # "lab": outcomes only, no CSR adjacency
    from core.truth import layout
    L = layout(cells)
    _S.update(names=list(cells["type_names"]), focal=L["focal"], K=L["K"], studies=db._S["studies"],
              cells=cells, plant=info["plant_feature"])
    return info


def one_study(job: tuple[str, int]) -> dict:
    stream, i = job
    s = _S["studies"][stream][i]
    names, focal, K = _S["names"], _S["focal"], _S["K"]
    t0 = perf_counter()
    sc = study_cells(_S["cells"], db.design(), s["image_seed"])
    t1 = perf_counter()
    Xs = np.stack([squidpy_features(looks, names, focal, seed=int(s["image_seed"]) % 2**32) for looks in sc])
    t2 = perf_counter()
    lf = lifelines_screen(Xs, s["time"], s["event"], s["strata"])
    call = bonferroni(lf["p"], Xs.shape[1])
    t3 = perf_counter()
    Xo = our_features(sc, focal, K)  # after the timer: agreement only
    from stats import screen_batch
    zo = screen_batch(Xo, s["time"][None], s["event"][None], strata=s["strata"], backend="numpy")["z"][0]
    return {"stream": stream, "i": i, "x_sha_ours": db.sha(Xo), "study_s": t3 - t0, "cells_s": t1 - t0,
            "squidpy_s": t2 - t1, "lifelines_s": t3 - t2, "fitted": lf["fitted"], "failed": lf["failed"],
            **{f"lab_{k}": v for k, v in call.items()}, "ours_top": int(np.nanargmax(np.abs(zo))),
            "ours_maxT": float(np.nanmax(np.abs(zo))), "agreement": agreement(Xs, Xo, lf["z"], zo)}


def run(cells: dict, plan: dict, workers: int = 8) -> dict:
    """The plan's studies over a fork pool of single-thread workers; pool wall is the timed compute."""
    import multiprocessing as mp
    info = prepare(cells, plan)
    first = next((st, 0) for st, n in plan.items() if n)
    t = perf_counter()
    one_study(first)  # numba JIT in squidpy and lifelines' imports, before the fork; recomputed in the pool
    warm_s = perf_counter() - t
    jobs = [(st, i) for st, n in plan.items() for i in range(n)]
    t = perf_counter()
    with mp.get_context("fork").Pool(workers) as pool:
        recs = pool.map(one_study, jobs, chunksize=1)
    wall = perf_counter() - t
    return {"records": recs, "pool_wall_s": wall, "warm_s": warm_s, "workers": workers, **info}


def summarize(out: dict) -> dict:
    recs, plant = out["records"], out["plant_feature"]
    n = len(recs)
    per = out["pool_wall_s"] / n
    by = {st: [r for r in recs if r["stream"] == st] for st in ("null", "alternative")}
    ag = [r["agreement"] for r in recs]
    med = lambda k: float(np.nanmedian([a[k] for a in ag if a[k] is not None])) if ag else None
    return {
        "studies": n, "seconds_per_study": per, "pool_wall_s": out["pool_wall_s"], "workers": out["workers"],
        "worker_study_s": {"median": float(np.median([r["study_s"] for r in recs])),
                           "min": float(min(r["study_s"] for r in recs)), "max": float(max(r["study_s"] for r in recs))},
        "stage_share": {k: float(sum(r[f"{k}_s"] for r in recs) / sum(r["study_s"] for r in recs))
                        for k in ("cells", "squidpy", "lifelines")},
        "ratio_vs_fused": per / FUSED_S_PER_STUDY["value"], "fused": FUSED_S_PER_STUDY,
        "calls": {st: db.rate([r["lab_call"] for r in rs]) for st, rs in by.items() if rs},
        "planted_feature": plant,
        "top_is_planted": {"lifelines": sum(r["lab_top"] == plant for r in by["alternative"]),
                           "ours": sum(r["ours_top"] == plant for r in by["alternative"]),
                           "of": len(by["alternative"])},
        "fits": {"fitted": sum(r["fitted"] for r in recs), "failed": sum(r["failed"] for r in recs)},
        "agreement_median": {k: med(k) for k in ("fraction_pearson", "fraction_spearman", "enrichment_pearson",
                                                   "enrichment_spearman", "z_spearman_abs", "top10_overlap")},
        "top1_same": sum(a["top1_same"] for a in ag), "nan_only_squidpy": sum(a["nan_only_squidpy"] for a in ag),
        "nan_only_ours": sum(a["nan_only_ours"] for a in ag),
    }
