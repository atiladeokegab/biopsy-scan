"""Translate Step 2 (A-17.2): IHC format simulation of the frozen R-16 biomarker."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ._guard import resolve_r16_rule, validate_discovery_cohort
from ._stats import (
    fit_stratified_cox,
    fold_assignment_vector,
    harrell_c_index,
    paired_retained_c_index,
    paired_retained_c_index_draws,
    stable_hash,
    stratified_kfolds,
)
from .panels import fit_full_r16_oof


class ArcsinhGmmGate:
    """Two-component 1D Gaussian mixture positivity gate on arcsinh-transformed intensities."""

    def __init__(
        self,
        cofactor: float,
        weights: tuple[float, float],
        means: tuple[float, float],
        variances: tuple[float, float],
        constant_call: bool | None = None,
    ) -> None:
        self.cofactor = float(cofactor)
        self.weights = (float(weights[0]), float(weights[1]))
        self.means = (float(means[0]), float(means[1]))
        self.variances = (float(variances[0]), float(variances[1]))
        self.constant_call = constant_call

    def predict_positive(self, x: np.ndarray) -> np.ndarray:
        """Return boolean positivity calls from fitted training parameters only."""
        vals = np.asarray(x, dtype=np.float64)
        if self.constant_call is not None:
            return np.full(len(vals), bool(self.constant_call), dtype=bool)
        y = np.arcsinh(vals / self.cofactor)
        log_p0 = math.log(max(self.weights[0], 1e-12)) - 0.5 * (
            math.log(2.0 * math.pi * self.variances[0]) + ((y - self.means[0]) ** 2) / self.variances[0]
        )
        log_p1 = math.log(max(self.weights[1], 1e-12)) - 0.5 * (
            math.log(2.0 * math.pi * self.variances[1]) + ((y - self.means[1]) ** 2) / self.variances[1]
        )
        return log_p1 >= log_p0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cofactor": self.cofactor,
            "weights": list(self.weights),
            "means": list(self.means),
            "variances": list(self.variances),
            "constant_call": self.constant_call,
        }


def fit_arcsinh_gmm(x_train: np.ndarray, *, max_iter: int = 50, tol: float = 1e-6) -> ArcsinhGmmGate:
    """Fit arcsinh cofactor and 2-component 1D GMM on training cells ONLY."""
    vals = np.asarray(x_train, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 4:
        return ArcsinhGmmGate(1.0, (0.5, 0.5), (0.0, 0.0), (1.0, 1.0), constant_call=False)

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    scale = mad if mad > 1e-8 else (iqr if iqr > 1e-8 else float(np.std(vals)))
    cofactor = max(scale, 0.25)

    y = np.arcsinh(vals / cofactor)
    var_y = float(np.var(y))
    if var_y < 1e-10:
        return ArcsinhGmmGate(cofactor, (0.5, 0.5), (float(y[0]), float(y[0])), (1e-6, 1e-6), constant_call=False)

    mu = np.array([float(np.percentile(y, 25)), float(np.percentile(y, 75))], dtype=np.float64)
    if mu[1] - mu[0] < 1e-6:
        mu = np.array([float(np.mean(y)) - 0.5 * math.sqrt(var_y), float(np.mean(y)) + 0.5 * math.sqrt(var_y)])
    var = np.array([0.5 * var_y, 0.5 * var_y], dtype=np.float64)
    pi = np.array([0.5, 0.5], dtype=np.float64)
    min_var = max(1e-6, 1e-4 * var_y)

    prev_ll = -np.inf
    for _ in range(max_iter):
        lp0 = np.log(pi[0]) - 0.5 * (np.log(2.0 * math.pi * var[0]) + ((y - mu[0]) ** 2) / var[0])
        lp1 = np.log(pi[1]) - 0.5 * (np.log(2.0 * math.pi * var[1]) + ((y - mu[1]) ** 2) / var[1])
        mmax = np.maximum(lp0, lp1)
        lse = mmax + np.log(np.exp(lp0 - mmax) + np.exp(lp1 - mmax))
        ll = float(lse.sum())
        gamma1 = np.exp(lp1 - lse)
        gamma0 = 1.0 - gamma1
        n0 = float(gamma0.sum())
        n1 = float(gamma1.sum())
        if n0 < 1.0 or n1 < 1.0:
            break
        pi[0] = n0 / len(y)
        pi[1] = n1 / len(y)
        mu[0] = float(np.dot(gamma0, y) / n0)
        mu[1] = float(np.dot(gamma1, y) / n1)
        var[0] = max(float(np.dot(gamma0, (y - mu[0]) ** 2) / n0), min_var)
        var[1] = max(float(np.dot(gamma1, (y - mu[1]) ** 2) / n1), min_var)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    if mu[0] > mu[1]:
        mu = mu[::-1]
        var = var[::-1]
        pi = pi[::-1]
    return ArcsinhGmmGate(cofactor, (pi[0], pi[1]), (mu[0], mu[1]), (var[0], var[1]))


def _infer_target_compartments(
    type_names: list[str], cell_type: np.ndarray, rule: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Derive morphological compartments (tumour vs non-tumour) matching R-16 focal/neighbour roles."""
    tumour_codes = [i for i, name in enumerate(type_names) if "tumor" in name.lower() or "tumour" in name.lower()]
    if not tumour_codes:
        ones = np.ones(len(cell_type), dtype=bool)
        return ones, ones
    is_tum = np.isin(cell_type, tumour_codes)
    foc_is_tum = ("tumor" in str(rule["focal"]).lower()) or ("tumour" in str(rule["focal"]).lower())
    nbr_is_tum = ("tumor" in str(rule["neighbour"]).lower()) or ("tumour" in str(rule["neighbour"]).lower())
    return (is_tum if foc_is_tum else ~is_tum), (is_tum if nbr_is_tum else ~is_tum)


def _pos_enrichment_margin(pos: np.ndarray, target: np.ndarray) -> float:
    """Training-fold binary sensitivity minus false-positive rate against a reference cell type."""
    if not target.any():
        return 0.0
    tpr = float(pos[target].mean())
    fpr = float(pos[~target].mean()) if (~target).any() else 0.0
    return tpr - fpr


def _split_regions(cohort_data: dict[str, Any]) -> list[tuple[int, int, np.ndarray]]:
    reg_id = np.asarray(cohort_data["region_id"], dtype=np.int64)
    pat_of_reg = np.asarray(cohort_data["patient_of_region"], dtype=np.int64)
    order = np.argsort(reg_id, kind="mergesort")
    sorted_r = reg_id[order]
    uniq_r, starts, counts = np.unique(sorted_r, return_index=True, return_counts=True)
    return [(int(r), int(pat_of_reg[r]), order[s : s + c]) for r, s, c in zip(uniq_r.tolist(), starts.tolist(), counts.tolist())]


def _r16_ihc_spatial_feature(
    reg_cells: list[tuple[int, int, np.ndarray]],
    xy_by_reg: list[np.ndarray],
    pos_foc_all: np.ndarray,
    pos_nbr_all: np.ndarray,
    n_pat: int,
    *,
    radius_um: float,
    statistic: str,
    misreg_um: float = 0.0,
    shift_angle_rad: float = 0.0,
    rot_rad: float = 0.0,
) -> np.ndarray:
    """Recompute the R-16 biomarker (single-slide at misreg=0, or displaced Section 2)."""
    sum_pat = np.zeros(n_pat, dtype=np.float64)
    cnt_pat = np.zeros(n_pat, dtype=np.int64)
    r_sq = float(radius_um) ** 2
    zero_shift = float(misreg_um) == 0.0 and float(rot_rad) == 0.0
    dx = float(misreg_um) * math.cos(shift_angle_rad)
    dy = float(misreg_um) * math.sin(shift_angle_rad)
    cos_r, sin_r = math.cos(rot_rad), math.sin(rot_rad)

    for (_r, p, cells_r), xy_r in zip(reg_cells, xy_by_reg):
        pf = pos_foc_all[cells_r]
        if not pf.any() or not len(xy_r):
            continue
        pn = pos_nbr_all[cells_r]
        foc_idx = np.flatnonzero(pf)
        xy_a = xy_r[foc_idx]
        if zero_shift:
            xy_b = xy_r
        else:
            center = xy_r.mean(axis=0, keepdims=True)
            rel = xy_r - center
            rot_rel = np.column_stack([cos_r * rel[:, 0] - sin_r * rel[:, 1], sin_r * rel[:, 0] + cos_r * rel[:, 1]])
            xy_b = center + rot_rel + np.array([[dx, dy]], dtype=np.float64)

        tree_b = cKDTree(xy_b)
        lists = tree_b.query_ball_point(xy_a, float(radius_um))
        p_nbr = float(pn.mean()) if len(pn) else 0.0
        for u, cand in enumerate(lists):
            if not len(cand):
                continue
            cand_arr = np.asarray(cand, dtype=np.int64)
            if zero_shift:
                cand_arr = cand_arr[cand_arr != foc_idx[u]]
            if not len(cand_arr):
                continue
            d2 = ((xy_b[cand_arr] - xy_a[u]) ** 2).sum(axis=1)
            cand_arr = cand_arr[d2 <= r_sq]
            if len(cand_arr):
                frac = float(pn[cand_arr].mean())
                val = (frac / p_nbr if p_nbr > 0.0 else 0.0) if statistic == "enrichment" else frac
                sum_pat[p] += val
        cnt_pat[p] += int(len(foc_idx))

    out = np.full(n_pat, np.nan, dtype=np.float64)
    has = cnt_pat > 0
    out[has] = sum_pat[has] / cnt_pat[has]
    return out


def _r16_tile_feature(
    reg_cells: list[tuple[int, int, np.ndarray]],
    xy_by_reg: list[np.ndarray],
    pos_foc_all: np.ndarray,
    pos_nbr_all: np.ndarray,
    n_pat: int,
    *,
    tile_um: float,
    statistic: str,
) -> np.ndarray:
    """Recompute mesoscopic tile approximation of R-16 at tile_um x tile_um resolution."""
    sum_pat = np.zeros(n_pat, dtype=np.float64)
    cnt_pat = np.zeros(n_pat, dtype=np.int64)
    inv_t = 1.0 / float(tile_um)

    for (_r, p, cells_r), xy_r in zip(reg_cells, xy_by_reg):
        if not len(xy_r):
            continue
        pf = pos_foc_all[cells_r]
        pn = pos_nbr_all[cells_r]
        if not pf.any():
            continue
        p_nbr_reg = float(pn.mean()) if len(pn) else 0.0
        origin = xy_r.min(axis=0)
        tx = np.floor((xy_r[:, 0] - origin[0]) * inv_t).astype(np.int64)
        ty = np.floor((xy_r[:, 1] - origin[1]) * inv_t).astype(np.int64)
        # Group cells by tile using linear keys
        keys = tx * 1_000_003 + ty
        _uk, inv_tile, counts = np.unique(keys, return_inverse=True, return_counts=True)
        n_pos_nbr_tile = np.bincount(inv_tile, weights=pn.astype(np.float64), minlength=len(counts))
        frac_nbr_tile = n_pos_nbr_tile / counts
        if statistic == "enrichment":
            tile_val = frac_nbr_tile / p_nbr_reg if p_nbr_reg > 0.0 else np.zeros_like(frac_nbr_tile)
        else:
            tile_val = frac_nbr_tile
        foc_tiles = inv_tile[pf]
        sum_pat[p] += float(tile_val[foc_tiles].sum())
        cnt_pat[p] += int(len(foc_tiles))

    out = np.full(n_pat, np.nan, dtype=np.float64)
    has = cnt_pat > 0
    out[has] = sum_pat[has] / cnt_pat[has]
    return out


def simulate_formats(
    cohort_data: dict[str, Any],
    *,
    split: str = "discovery",
    panel: tuple[int, int] | None = None,
    panel_by_fold: list[tuple[int, int]] | None = None,
    step1_result: dict[str, Any] | None = None,
    misreg_grid: tuple[float, ...] = (0.0, 25.0, 50.0, 75.0, 100.0),
    tile_grid: tuple[float, ...] = (100.0, 150.0, 200.0),
    n_draws: int = 8,
    n_outer: int = 5,
    n_boot: int = 400,
    seed: int = 0,
    r16_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Step 2 IHC format degradation benchmarks (A-17.2) for the frozen R-16 biomarker."""
    validate_discovery_cohort(cohort_data, split=split)
    rule = resolve_r16_rule(cohort_data, r16_rule)
    if not rule["expressible"]:
        reason = str(rule.get("reason") or "R-16 candidate reported unsupported by cohort")
        return {
            "status": "UNMEASURED",
            "reason": reason,
            "r16_rule_sha256": rule["rule_sha256"],
            "single_slide": {"status": "UNMEASURED", "reason": reason},
            "serial_section": {str(int(d)): {"misreg_um": float(d), "status": "UNMEASURED", "reason": reason} for d in misreg_grid},
            "tiles": {str(int(t)): {"tile_um": float(t), "status": "UNMEASURED", "reason": reason} for t in tile_grid},
        }

    time = np.asarray(cohort_data["time"], dtype=np.float64)
    event = np.asarray(cohort_data["event"], dtype=bool)
    strata = np.asarray(cohort_data["strata"], dtype=np.int64)
    markers = np.asarray(cohort_data["markers"], dtype=np.float64)
    marker_names = list(cohort_data["marker_names"])
    n_pat = len(time)

    if step1_result is not None and "_folds" in step1_result and "_risk_full_oof" in step1_result:
        outer_folds = step1_result["_folds"]
        outer_fold_vec = step1_result["_fold_vec"]
        risk_full_oof = step1_result["_risk_full_oof"]
        if panel_by_fold is None and panel is None and "outer_selections" in step1_result:
            name_to_idx = {name: i for i, name in enumerate(marker_names)}
            panel_by_fold = []
            for sel in step1_result["outer_selections"]:
                chosen = sel.get("best_2marker") or sel.get("best_1marker")
                if chosen and len(chosen) == 2:
                    panel_by_fold.append((name_to_idx[chosen[0]], name_to_idx[chosen[1]]))
                elif chosen and len(chosen) == 1:
                    panel_by_fold.append((name_to_idx[chosen[0]], name_to_idx[chosen[0]]))
                else:
                    panel_by_fold.append((0, min(1, markers.shape[1] - 1)))
    else:
        outer_folds = stratified_kfolds(event, strata, n_splits=n_outer, seed=seed)
        outer_fold_vec = fold_assignment_vector(n_pat, outer_folds)
        risk_full_oof, _, _ = fit_full_r16_oof(cohort_data, rule, outer_folds)

    if panel_by_fold is None:
        if panel is None:
            panel = (0, min(1, markers.shape[1] - 1))
        panel_by_fold = [(int(panel[0]), int(panel[1])) for _ in outer_folds]

    c_full_oof = harrell_c_index(time, event, risk_full_oof, strata=strata, folds=outer_fold_vec)
    if not np.isfinite(c_full_oof) or c_full_oof <= 0.5:
        reason = "C_full is not estimable" if not np.isfinite(c_full_oof) else f"C_full <= 0.5 ({c_full_oof:.4f})"
        return {
            "status": "UNMEASURED",
            "reason": reason,
            "r16_rule_sha256": rule["rule_sha256"],
            "c_full_oof": None if not np.isfinite(c_full_oof) else float(c_full_oof),
            "single_slide": {"status": "UNMEASURED", "reason": reason},
            "serial_section": {str(int(d)): {"misreg_um": float(d), "status": "UNMEASURED", "reason": reason} for d in misreg_grid},
            "tiles": {str(int(t)): {"tile_um": float(t), "status": "UNMEASURED", "reason": reason} for t in tile_grid},
        }

    reg_cells = _split_regions(cohort_data)
    xy_all = np.stack([np.asarray(cohort_data["x"], dtype=np.float64), np.asarray(cohort_data["y"], dtype=np.float64)], 1)
    cell_type = np.asarray(cohort_data["cell_type"], dtype=np.int64)
    foc_comp, nbr_comp = _infer_target_compartments(list(cohort_data["type_names"]), cell_type, rule)
    xy_only = [xy_all[cells_r] for _, _, cells_r in reg_cells]
    pat_of_cell = np.empty(len(xy_all), dtype=np.int64)
    for _r, p, cells_r in reg_cells:
        pat_of_cell[cells_r] = p

    radius_um = float(rule["radius_um"])
    stat_name = str(rule["statistic"])
    feat_name = str(rule["feature"])
    foc_code = int(rule["focal_code"])
    nbr_code = int(rule["nbr_code"])

    # Fit training-fold arcsinh GMM gates across ALL training cells AND derive
    # column-order-invariant stain-to-target roles strictly inside each training fold.
    fold_calls: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    for f_id, (tr_idx, _te_idx) in enumerate(outer_folds):
        ma, mb = panel_by_fold[f_id]
        tr_cell_mask = np.isin(pat_of_cell, tr_idx)
        gate_a = fit_arcsinh_gmm(markers[tr_cell_mask, ma])
        gate_b = gate_a if ma == mb else fit_arcsinh_gmm(markers[tr_cell_mask, mb])
        pos_a_tr = gate_a.predict_positive(markers[tr_cell_mask, ma])
        pos_b_tr = gate_b.predict_positive(markers[tr_cell_mask, mb])
        y_foc_tr = cell_type[tr_cell_mask] == foc_code
        y_nbr_tr = cell_type[tr_cell_mask] == nbr_code
        score_ab = _pos_enrichment_margin(pos_a_tr, y_foc_tr) + _pos_enrichment_margin(pos_b_tr, y_nbr_tr)
        score_ba = _pos_enrichment_margin(pos_b_tr, y_foc_tr) + _pos_enrichment_margin(pos_a_tr, y_nbr_tr)
        if score_ba > score_ab or (
            score_ba == score_ab and (marker_names[mb], marker_names[ma]) < (marker_names[ma], marker_names[mb])
        ):
            foc_m, gate_foc, nbr_m, gate_nbr = mb, gate_b, ma, gate_a
        else:
            foc_m, gate_foc, nbr_m, gate_nbr = ma, gate_a, mb, gate_b
        pos_foc = foc_comp & gate_foc.predict_positive(markers[:, foc_m])
        pos_nbr = nbr_comp & gate_nbr.predict_positive(markers[:, nbr_m])
        fold_calls.append((
            pos_foc,
            pos_nbr,
            {
                "fold": f_id,
                "panel": sorted([marker_names[ma], marker_names[mb]]),
                "focal_marker": marker_names[foc_m],
                "neighbour_marker": marker_names[nbr_m],
                "gate_focal": gate_foc.to_dict(),
                "gate_neighbour": gate_nbr.to_dict(),
            },
        ))

    def _eval_shift(delta_um: float) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        rng = np.random.default_rng(seed + 777 + int(round(delta_um * 10)))
        draws = 1 if float(delta_um) == 0.0 else max(int(n_draws), 1)
        draw_oofs: list[np.ndarray] = []
        models_meta: list[dict[str, Any]] = []
        for d in range(draws):
            ang = 0.0 if float(delta_um) == 0.0 else float(rng.uniform(0.0, 2.0 * math.pi))
            rot = 0.0 if float(delta_um) == 0.0 else float(rng.uniform(-0.035, 0.035))
            risk_d = np.full(n_pat, np.nan, dtype=np.float64)
            for f_id, (tr_idx, te_idx) in enumerate(outer_folds):
                pos_foc, pos_nbr, _gmeta = fold_calls[f_id]
                x_feat = _r16_ihc_spatial_feature(
                    reg_cells, xy_only, pos_foc, pos_nbr, n_pat,
                    radius_um=radius_um, statistic=stat_name,
                    misreg_um=float(delta_um), shift_angle_rad=ang, rot_rad=rot,
                )
                mdl = fit_stratified_cox(
                    x_feat[tr_idx, None], time[tr_idx], event[tr_idx], strata[tr_idx], columns=[feat_name], l2=0.05
                )
                risk_d[te_idx] = mdl.predict_risk(x_feat[te_idx, None])
                models_meta.append({"fold": f_id, "draw": d, "delta_um": float(delta_um), "model": mdl.to_dict()})
            draw_oofs.append(risk_d)
        return draw_oofs, models_meta

    single_oofs, single_models = _eval_shift(0.0)
    single_res = paired_retained_c_index(
        time, event, single_oofs[0], risk_full_oof, strata=strata, folds=outer_fold_vec, n_boot=n_boot, seed=seed + 60000
    )

    serial_results: dict[str, Any] = {}
    serial_models_meta: dict[str, Any] = {}
    for idx_d, delta in enumerate(misreg_grid):
        key = str(int(delta)) if float(delta).is_integer() else str(delta)
        if float(delta) == 0.0:
            serial_results[key] = {"misreg_um": float(delta), **single_res}
            serial_models_meta[key] = single_models
        else:
            oofs_d, mods_d = _eval_shift(float(delta))
            ev_d = paired_retained_c_index_draws(
                time, event, oofs_d, risk_full_oof, strata=strata, folds=outer_fold_vec, n_boot=n_boot, seed=seed + 60000 + idx_d
            )
            serial_results[key] = {"misreg_um": float(delta), **ev_d}
            serial_models_meta[key] = mods_d

    tile_results: dict[str, Any] = {}
    tile_models_meta: dict[str, Any] = {}
    for idx_t, tile_um in enumerate(tile_grid):
        tkey = str(int(tile_um)) if float(tile_um).is_integer() else str(tile_um)
        tile_oof = np.full(n_pat, np.nan, dtype=np.float64)
        tmods = []
        for f_id, (tr_idx, te_idx) in enumerate(outer_folds):
            pos_foc, pos_nbr, _gmeta = fold_calls[f_id]
            xt = _r16_tile_feature(
                reg_cells, xy_only, pos_foc, pos_nbr, n_pat, tile_um=float(tile_um), statistic=stat_name
            )
            mdl = fit_stratified_cox(
                xt[tr_idx, None], time[tr_idx], event[tr_idx], strata[tr_idx], columns=[feat_name], l2=0.05
            )
            tile_oof[te_idx] = mdl.predict_risk(xt[te_idx, None])
            tmods.append({"fold": f_id, "tile_um": float(tile_um), "model": mdl.to_dict()})
        ev_t = paired_retained_c_index(
            time, event, tile_oof, risk_full_oof, strata=strata, folds=outer_fold_vec, n_boot=n_boot, seed=seed + 61000 + idx_t
        )
        tile_results[tkey] = {"tile_um": float(tile_um), **ev_t}
        tile_models_meta[tkey] = tmods

    format_hash = stable_hash({
        "r16_rule_sha256": rule["rule_sha256"],
        "fold_gates": [g[2] for g in fold_calls],
        "single_models": single_models,
        "serial_models": serial_models_meta,
        "tile_models": tile_models_meta,
    })

    return {
        "status": "MEASURED",
        "r16_rule_sha256": rule["rule_sha256"],
        "r16_feature": feat_name,
        "c_full_oof": float(c_full_oof),
        "format_model_hash": format_hash,
        "fold_gates": [g[2] for g in fold_calls],
        "single_slide": single_res,
        "serial_section": serial_results,
        "tiles": tile_results,
    }
