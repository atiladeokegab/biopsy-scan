"""Translate Step 1 (A-17.1): exhaustive 1- and 2-marker panel search for the frozen R-16 biomarker."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import block_diag, csr_matrix
from scipy.spatial import cKDTree

from ._guard import resolve_r16_rule, validate_discovery_cohort
from ._stats import (
    fit_stratified_cox,
    fold_assignment_vector,
    harrell_c_index,
    paired_retained_c_index,
    stable_hash,
    stratified_kfolds,
)


def enumerate_panels(marker_names: list[str] | tuple[str, ...]) -> list[tuple[int, ...]]:
    """Enumerate every 1- and 2-marker panel: M + M*(M-1)/2 panels in canonical index order."""
    m = len(marker_names)
    panels: list[tuple[int, ...]] = [(i,) for i in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            panels.append((i, j))
    return panels


def _region_adjacencies(
    cohort_data: dict[str, Any], radius_um: float
) -> dict[str, Any]:
    """Precompute a block-diagonal CSR neighbour matrix across all regions (self excluded)."""
    xy = np.stack([np.asarray(cohort_data["x"], dtype=np.float64), np.asarray(cohort_data["y"], dtype=np.float64)], 1)
    region_id = np.asarray(cohort_data["region_id"], dtype=np.int64)
    pat_of_reg = np.asarray(cohort_data["patient_of_region"], dtype=np.int64)
    n_pat = len(cohort_data["time"])

    reg_order = np.argsort(region_id, kind="mergesort")
    reg_sorted = region_id[reg_order]
    uniq_reg, starts, counts = np.unique(reg_sorted, return_index=True, return_counts=True)

    r_sq = float(radius_um) ** 2
    adj_blocks: list[csr_matrix] = []
    reg_cells_ptr = [0]
    reg_sizes = []
    for _reg, s, c in zip(uniq_reg.tolist(), starts.tolist(), counts.tolist()):
        cells_r = reg_order[s : s + c]
        xy_r = xy[cells_r]
        n_r = len(cells_r)
        if n_r == 0:
            continue
        tree = cKDTree(xy_r)
        pairs = tree.query_pairs(float(radius_um), output_type="ndarray")
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            d2 = ((xy_r[a] - xy_r[b]) ** 2).sum(axis=1)
            keep = d2 <= r_sq
            a, b = a[keep], b[keep]
            rows = np.concatenate([a, b])
            cols = np.concatenate([b, a])
            adj = csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n_r, n_r))
        else:
            adj = csr_matrix((n_r, n_r), dtype=np.float64)
        adj_blocks.append(adj)
        reg_cells_ptr.append(reg_cells_ptr[-1] + n_r)
        reg_sizes.append(n_r)

    adj_all = block_diag(adj_blocks, format="csr") if adj_blocks else csr_matrix((0, 0), dtype=np.float64)
    deg_all = np.asarray(adj_all.sum(axis=1)).ravel()
    ord_reg = region_id[reg_order]
    _, reg_inv, reg_counts_arr = np.unique(ord_reg, return_inverse=True, return_counts=True)
    pat_ord = pat_of_reg[ord_reg]
    return {
        "n_pat": n_pat,
        "reg_order": reg_order,
        "adj_all": adj_all,
        "deg_all": deg_all,
        "pat_ord": pat_ord,
        "reg_inv": reg_inv,
        "reg_counts": reg_counts_arr.astype(np.float64),
        "n_regs": len(reg_counts_arr),
        "radius_um": float(radius_um),
    }


def _r16_patient_feature(
    geom: dict[str, Any],
    is_focal: np.ndarray,
    is_nbr: np.ndarray,
    *,
    statistic: str,
) -> np.ndarray:
    """Compute exact patient-level R-16 spatial biomarker vectorized across all regions."""
    n_pat = geom["n_pat"]
    if len(geom["reg_order"]) == 0:
        return np.full(n_pat, np.nan, dtype=np.float64)

    ord_idx = geom["reg_order"]
    foc_ord = is_focal[ord_idx]
    if not foc_ord.any():
        return np.full(n_pat, np.nan, dtype=np.float64)

    nbr_ord = is_nbr[ord_idx].astype(np.float64)
    nbr_cnt = geom["adj_all"].dot(nbr_ord)
    deg = geom["deg_all"]
    foc_pos = np.flatnonzero(foc_ord)
    frac = np.divide(nbr_cnt[foc_pos], deg[foc_pos], out=np.zeros(len(foc_pos), dtype=np.float64), where=deg[foc_pos] > 0.0)
    if statistic == "enrichment":
        reg_nbr = np.bincount(geom["reg_inv"], weights=nbr_ord, minlength=geom["n_regs"])
        p_nbr_reg = reg_nbr / np.maximum(geom["reg_counts"], 1.0)
        p_nbr_foc = p_nbr_reg[geom["reg_inv"][foc_pos]]
        vals = np.divide(frac, p_nbr_foc, out=np.zeros_like(frac), where=p_nbr_foc > 0.0)
    else:
        vals = frac

    pat_foc = geom["pat_ord"][foc_pos]
    sum_pat = np.bincount(pat_foc, weights=vals, minlength=n_pat)
    cnt_pat = np.bincount(pat_foc, minlength=n_pat)
    out = np.full(n_pat, np.nan, dtype=np.float64)
    has = cnt_pat > 0
    out[has] = sum_pat[has] / cnt_pat[has]
    return out


def _fit_additive_cell_ll(
    markers: np.ndarray,
    cell_type: np.ndarray,
    tr_cell_mask: np.ndarray,
    focal_code: int,
    nbr_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit additive per-marker Gaussian Naive Bayes parameters on training-fold cells ONLY.

    Classes: 0 = other, 1 = R-16 focal type, 2 = R-16 neighbour type.
    Because class log-likelihoods decompose cleanly per marker, any panel P subset sums LL[:, P, :].
    """
    m = markers.shape[1]
    y_all = np.zeros(len(cell_type), dtype=np.int64)
    y_all[cell_type == focal_code] = 1
    y_all[cell_type == nbr_code] = 2

    x_tr = markers[tr_cell_mask]
    y_tr = y_all[tr_cell_mask]
    active_classes = np.array([c for c in (0, 1, 2) if (y_tr == c).any()], dtype=np.int64)
    k_cls = len(active_classes)
    if k_cls == 0 or not len(x_tr):
        raise ValueError("training fold contains no cells")

    global_var = np.maximum(np.var(x_tr, axis=0), 1e-6)
    mu = np.zeros((k_cls, m), dtype=np.float64)
    var = np.ones((k_cls, m), dtype=np.float64)
    log_prior = np.zeros(k_cls, dtype=np.float64)

    n_tr = float(len(y_tr))
    for idx_c, cls in enumerate(active_classes.tolist()):
        mask_c = y_tr == cls
        n_c = int(mask_c.sum())
        log_prior[idx_c] = np.log(n_c / n_tr)
        xc = x_tr[mask_c]
        mu[idx_c] = xc.mean(axis=0)
        v_c = xc.var(axis=0) if n_c > 1 else global_var
        var[idx_c] = np.maximum(v_c, 1e-3 * global_var)

    # LL shape: (n_cells, m, k_cls)
    diff = markers[:, :, None] - mu.T[None, :, :]
    ll = -0.5 * (np.log(2.0 * np.pi * var.T[None, :, :]) + (diff * diff) / var.T[None, :, :])
    params = {
        "active_classes": active_classes.tolist(),
        "log_prior": log_prior.tolist(),
        "mu": mu.tolist(),
        "var": var.tolist(),
    }
    return active_classes, log_prior, ll, params


def _predict_states_for_panel(
    active_classes: np.ndarray,
    log_prior: np.ndarray,
    ll: np.ndarray,
    panel: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    scores = log_prior[None, :] + ll[:, panel, :].sum(axis=1)
    pred_cls = active_classes[np.argmax(scores, axis=1)]
    return pred_cls == 1, pred_cls == 2


def fit_full_r16_oof(
    cohort_data: dict[str, Any],
    rule: dict[str, Any],
    folds: list[tuple[np.ndarray, np.ndarray]],
    geom: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Fit full M-marker R-16 Cox model on each training fold and return OOF risk."""
    if geom is None:
        geom = _region_adjacencies(cohort_data, float(rule["radius_um"]))
    markers = np.asarray(cohort_data["markers"], dtype=np.float64)
    cell_type = np.asarray(cohort_data["cell_type"], dtype=np.int64)
    region_id = np.asarray(cohort_data["region_id"], dtype=np.int64)
    pat_of_cell = np.asarray(cohort_data["patient_of_region"], dtype=np.int64)[region_id]
    time = np.asarray(cohort_data["time"], dtype=np.float64)
    event = np.asarray(cohort_data["event"], dtype=bool)
    strata = np.asarray(cohort_data["strata"], dtype=np.int64)

    all_markers = tuple(range(markers.shape[1]))
    risk_full_oof = np.full(len(time), np.nan, dtype=np.float64)
    fold_models = []
    for f_id, (tr_idx, te_idx) in enumerate(folds):
        tr_cells = np.isin(pat_of_cell, tr_idx)
        ac, lp, ll, cls_params = _fit_additive_cell_ll(
            markers, cell_type, tr_cells, int(rule["focal_code"]), int(rule["nbr_code"])
        )
        is_foc, is_nbr = _predict_states_for_panel(ac, lp, ll, all_markers)
        x_r16 = _r16_patient_feature(geom, is_foc, is_nbr, statistic=str(rule["statistic"]))
        mdl = fit_stratified_cox(
            x_r16[tr_idx, None],
            time[tr_idx],
            event[tr_idx],
            strata[tr_idx],
            columns=[str(rule["feature"])],
            l2=0.05,
        )
        risk_full_oof[te_idx] = mdl.predict_risk(x_r16[te_idx, None])
        fold_models.append({"fold": f_id, "cell_classifier": cls_params, "cox": mdl.to_dict()})
    return risk_full_oof, fold_models, geom


def run_panel_cv(
    cohort_data: dict[str, Any],
    *,
    split: str = "discovery",
    n_outer: int = 5,
    n_inner: int = 3,
    n_boot: int = 400,
    seed: int = 0,
    r16_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Step 1 exhaustive 1- and 2-marker panel search for the frozen R-16 biomarker (A-17.1)."""
    validate_discovery_cohort(cohort_data, split=split)
    marker_names = list(cohort_data["marker_names"])
    m = len(marker_names)
    panels = enumerate_panels(marker_names)
    assert len(panels) == m + m * (m - 1) // 2

    rule = resolve_r16_rule(cohort_data, r16_rule)
    if not rule["expressible"]:
        reason = str(rule.get("reason") or "R-16 candidate reported unsupported by cohort")
        return {
            "status": "UNMEASURED",
            "reason": reason,
            "r16_rule_sha256": rule["rule_sha256"],
            "r16_feature": rule.get("feature"),
            "m": m,
            "n_panels": len(panels),
            "n_patients": int(len(cohort_data["time"])),
            "n_events": int(np.asarray(cohort_data["event"], dtype=bool).sum()),
            "panels": [
                {"markers": [marker_names[i] for i in p], "size": len(p), "status": "UNMEASURED", "reason": reason}
                for p in panels
            ],
            "nested_selected_1marker": {"status": "UNMEASURED", "reason": reason},
            "nested_selected_2marker": {"status": "UNMEASURED", "reason": reason},
            "panel_model_hash": stable_hash({"status": "UNMEASURED", "reason": reason, "r16_rule_sha256": rule["rule_sha256"], "m": m}),
        }

    markers = np.asarray(cohort_data["markers"], dtype=np.float64)
    cell_type = np.asarray(cohort_data["cell_type"], dtype=np.int64)
    region_id = np.asarray(cohort_data["region_id"], dtype=np.int64)
    pat_of_cell = np.asarray(cohort_data["patient_of_region"], dtype=np.int64)[region_id]
    time = np.asarray(cohort_data["time"], dtype=np.float64)
    event = np.asarray(cohort_data["event"], dtype=bool)
    strata = np.asarray(cohort_data["strata"], dtype=np.int64)
    n_pat = len(time)

    outer_folds = stratified_kfolds(event, strata, n_splits=n_outer, seed=seed)
    outer_fold_vec = fold_assignment_vector(n_pat, outer_folds)
    geom = _region_adjacencies(cohort_data, float(rule["radius_um"]))
    risk_full_oof, full_models, geom = fit_full_r16_oof(cohort_data, rule, outer_folds, geom=geom)
    c_full_oof = harrell_c_index(time, event, risk_full_oof, strata=strata, folds=outer_fold_vec)

    oof_risk_by_panel = np.full((len(panels), n_pat), np.nan, dtype=np.float64)
    outer_selections: list[dict[str, Any]] = []
    panel_hashes_per_fold: list[dict[str, Any]] = []
    stat_name = str(rule["statistic"])
    feat_name = str(rule["feature"])

    for f_id, (tr_idx, te_idx) in enumerate(outer_folds):
        tr_cells = np.isin(pat_of_cell, tr_idx)
        ac, lp, ll, cls_params = _fit_additive_cell_ll(
            markers, cell_type, tr_cells, int(rule["focal_code"]), int(rule["nbr_code"])
        )

        # Build panel features inside outer fold using ONLY training-fold cell classifier.
        inner_folds = stratified_kfolds(
            event[tr_idx], strata[tr_idx], n_splits=min(n_inner, len(tr_idx)), seed=seed + 100 + f_id
        )
        inner_fold_vec = fold_assignment_vector(len(tr_idx), inner_folds)
        inner_ll_cache = []
        for in_tr_rel, in_te_rel in inner_folds:
            in_tr = tr_idx[in_tr_rel]
            in_tr_cells = np.isin(pat_of_cell, in_tr)
            in_ac, in_lp, in_ll, _ = _fit_additive_cell_ll(
                markers, cell_type, in_tr_cells, int(rule["focal_code"]), int(rule["nbr_code"])
            )
            inner_ll_cache.append((in_tr, tr_idx[in_te_rel], in_te_rel, in_ac, in_lp, in_ll))

        inner_scores = np.full(len(panels), np.nan, dtype=np.float64)
        for p_idx, panel in enumerate(panels):
            in_oof = np.full(len(tr_idx), np.nan, dtype=np.float64)
            for in_tr, in_te, in_te_rel, in_ac, in_lp, in_ll in inner_ll_cache:
                i_foc, i_nbr = _predict_states_for_panel(in_ac, in_lp, in_ll, panel)
                x_in = _r16_patient_feature(geom, i_foc, i_nbr, statistic=stat_name)
                mdl_in = fit_stratified_cox(
                    x_in[in_tr, None], time[in_tr], event[in_tr], strata[in_tr], columns=[feat_name], l2=0.05
                )
                in_oof[in_te_rel] = mdl_in.predict_risk(x_in[in_te, None])
            inner_scores[p_idx] = harrell_c_index(
                time[tr_idx], event[tr_idx], in_oof, strata=strata[tr_idx], folds=inner_fold_vec
            )

        one_idx = [i for i, p in enumerate(panels) if len(p) == 1]
        two_idx = [i for i, p in enumerate(panels) if len(p) == 2]
        best1 = max(one_idx, key=lambda idx: -np.inf if not np.isfinite(inner_scores[idx]) else inner_scores[idx])
        best2 = max(two_idx, key=lambda idx: -np.inf if not np.isfinite(inner_scores[idx]) else inner_scores[idx]) if two_idx else best1

        fold_best_models = {}
        for p_idx, panel in enumerate(panels):
            o_foc, o_nbr = _predict_states_for_panel(ac, lp, ll, panel)
            x_p = _r16_patient_feature(geom, o_foc, o_nbr, statistic=stat_name)
            mdl = fit_stratified_cox(
                x_p[tr_idx, None], time[tr_idx], event[tr_idx], strata[tr_idx], columns=[feat_name], l2=0.05
            )
            oof_risk_by_panel[p_idx, te_idx] = mdl.predict_risk(x_p[te_idx, None])
            if p_idx in (best1, best2):
                fold_best_models[str(p_idx)] = mdl.to_dict()

        outer_selections.append({
            "fold": f_id,
            "best_1marker_idx": int(best1),
            "best_1marker": [marker_names[i] for i in panels[best1]],
            "best_2marker_idx": int(best2),
            "best_2marker": [marker_names[i] for i in panels[best2]],
        })
        panel_hashes_per_fold.append({
            "fold": f_id,
            "cell_classifier": cls_params,
            "selected_models": fold_best_models,
        })

    nested1_oof = np.full(n_pat, np.nan, dtype=np.float64)
    nested2_oof = np.full(n_pat, np.nan, dtype=np.float64)
    for (_tr_idx, te_idx), sel in zip(outer_folds, outer_selections):
        nested1_oof[te_idx] = oof_risk_by_panel[sel["best_1marker_idx"], te_idx]
        nested2_oof[te_idx] = oof_risk_by_panel[sel["best_2marker_idx"], te_idx]

    panel_records = []
    for p_idx, panel in enumerate(panels):
        ev = paired_retained_c_index(
            time, event, oof_risk_by_panel[p_idx], risk_full_oof,
            strata=strata, folds=outer_fold_vec, n_boot=n_boot, seed=seed + 1000 + p_idx,
        )
        panel_records.append({
            "panel_index": p_idx,
            "markers": [marker_names[i] for i in panel],
            "size": len(panel),
            **ev,
        })

    nested1_res = paired_retained_c_index(
        time, event, nested1_oof, risk_full_oof, strata=strata, folds=outer_fold_vec, n_boot=n_boot, seed=seed + 50000
    )
    nested2_res = paired_retained_c_index(
        time, event, nested2_oof, risk_full_oof, strata=strata, folds=outer_fold_vec, n_boot=n_boot, seed=seed + 50001
    )

    model_hash = stable_hash({
        "r16_rule_sha256": rule["rule_sha256"],
        "marker_names": marker_names,
        "full_models": full_models,
        "outer_selections": outer_selections,
        "selected_models": panel_hashes_per_fold,
    })

    measured = [r for r in panel_records if r["status"] == "MEASURED"]
    top_panel = max(measured, key=lambda r: r["c_panel"]) if measured else panel_records[0]

    return {
        "status": "MEASURED" if np.isfinite(c_full_oof) and c_full_oof > 0.5 else "UNMEASURED",
        "reason": None if (np.isfinite(c_full_oof) and c_full_oof > 0.5) else (
            "C_full is not estimable" if not np.isfinite(c_full_oof) else f"C_full <= 0.5 ({c_full_oof:.4f})"
        ),
        "r16_rule_sha256": rule["rule_sha256"],
        "r16_feature": feat_name,
        "m": m,
        "n_panels": len(panels),
        "n_patients": n_pat,
        "n_events": int(event.sum()),
        "c_full_oof": None if not np.isfinite(c_full_oof) else float(c_full_oof),
        "panel_model_hash": model_hash,
        "outer_selections": outer_selections,
        "nested_selected_1marker": nested1_res,
        "nested_selected_2marker": nested2_res,
        "top_panel": top_panel,
        "panels": panel_records,
        "_folds": outer_folds,
        "_fold_vec": outer_fold_vec,
        "_risk_full_oof": risk_full_oof,
    }
