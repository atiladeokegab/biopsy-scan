"""Survival scoring, exact log-sum-exp Breslow Cox fit, fold-invariant C-index, and validation."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from scipy.optimize import minimize


def stable_hash(obj: Any) -> str:
    """Deterministic SHA-256 of JSON/NumPy parameter structures."""

    def _encode(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return {"dtype": str(x.dtype), "shape": list(x.shape), "data": x.tolist()}
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, dict):
            return {str(k): _encode(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [_encode(v) for v in x]
        return x

    payload = json.dumps(_encode(obj), sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def harrell_c_index(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    folds: np.ndarray | None = None,
) -> float:
    """Harrell C-index over comparable pairs within matching (fold, stratum) groups.

    Restricting comparisons to shared (fold, stratum) makes C-index invariant to
    fold-specific and stratum-specific additive offsets in Cox linear predictors.
    """
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    r = np.asarray(risk, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(r)
    st = np.asarray(strata, dtype=np.int64)[valid] if strata is not None else None
    fl = np.asarray(folds, dtype=np.int64)[valid] if folds is not None else None
    t, e, r = t[valid], e[valid], r[valid]
    if len(t) < 2 or not e.any():
        return float("nan")

    evt_idx = np.flatnonzero(e)
    t_i = t[evt_idx, None]
    r_i = r[evt_idx, None]
    comp = (t_i < t[None, :]) | ((t_i == t[None, :]) & (~e[None, :]))
    if st is not None:
        comp &= st[evt_idx, None] == st[None, :]
    if fl is not None:
        comp &= fl[evt_idx, None] == fl[None, :]
    n_comp = int(comp.sum())
    if n_comp == 0:
        return float("nan")
    conc = ((r_i > r[None, :]) & comp).sum() + 0.5 * ((r_i == r[None, :]) & comp).sum()
    return float(conc / n_comp)


def stratified_kfolds(
    event: np.ndarray, strata: np.ndarray, *, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic patient-level K-fold split stratified by (stratum, event)."""
    n = len(event)
    if n < 2:
        raise ValueError("need at least 2 patients for cross-validation")
    k = min(int(n_splits), n)
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    keys = np.asarray(strata, dtype=np.int64) * 2 + np.asarray(event, dtype=np.int64)
    offset = 0
    for key in sorted(set(keys.tolist())):
        idx = np.flatnonzero(keys == key)
        rng.shuffle(idx)
        for pos, pat in enumerate(idx.tolist()):
            folds[(offset + pos) % k].append(pat)
        offset = (offset + len(idx)) % k
    all_idx = np.arange(n, dtype=np.int64)
    out = []
    for f in folds:
        test = np.array(sorted(f), dtype=np.int64)
        mask = np.ones(n, dtype=bool)
        mask[test] = False
        out.append((all_idx[mask], test))
    return out


def fold_assignment_vector(n_patients: int, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Return length-n array mapping each patient to its outer test fold index."""
    fvec = np.full(n_patients, -1, dtype=np.int64)
    for f_id, (_tr, te) in enumerate(folds):
        fvec[te] = f_id
    return fvec


class FittedCoxModel:
    """Frozen regularized stratified Cox model (training stats and weights only)."""

    def __init__(
        self,
        medians: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        beta: np.ndarray,
        columns: list[str] | tuple[str, ...],
        l2: float,
        *,
        converged: bool = True,
        status_message: str = "ok",
    ) -> None:
        self.medians = medians
        self.mean = mean
        self.std = std
        self.beta = beta
        self.columns = list(columns)
        self.l2 = float(l2)
        self.converged = bool(converged)
        self.status_message = str(status_message)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64).copy()
        if arr.ndim == 1:
            arr = arr[:, None]
        for j in range(arr.shape[1]):
            bad = ~np.isfinite(arr[:, j])
            if bad.any():
                arr[bad, j] = self.medians[j]
        return np.divide(
            arr - self.mean,
            self.std,
            out=np.zeros_like(arr),
            where=self.std > 0.0,
        )

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if not self.converged or not np.isfinite(self.beta).all():
            return np.full(len(X), np.nan, dtype=np.float64)
        return self._transform(X) @ self.beta

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "medians": self.medians.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "beta": self.beta.tolist(),
            "l2": self.l2,
            "converged": self.converged,
            "status_message": self.status_message,
        }


def cox_neg_loglik_and_grad(
    beta: np.ndarray,
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    inv_n: float,
    l2: float,
) -> tuple[float, np.ndarray]:
    """Exact stratified Breslow partial negative log-likelihood + L2 penalty & matching gradient."""
    nll = 0.0
    grad = np.zeros_like(beta)
    for Zs, grp_ends, d_k, s_k in groups:
        eta = Zs @ beta
        # Exact log-sum-exp cumulative risk denominator: zero clipping, exact gradient.
        log_cum = np.logaddexp.accumulate(eta)[grp_ends]
        nll -= float((s_k @ beta).sum() - np.dot(d_k, log_cum))
        mask = np.arange(len(eta))[:, None] <= grp_ends[None, :]
        diff = np.where(mask, eta[:, None] - log_cum[None, :], -np.inf)
        weights = np.exp(diff) * d_k[None, :]
        grad -= s_k.sum(axis=0) - weights.sum(axis=1) @ Zs
    nll = nll * inv_n + 0.5 * l2 * float(np.dot(beta, beta))
    grad = grad * inv_n + l2 * beta
    return nll, grad


def _fit_cox_1d_newton(
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    inv_n: float,
    l2: float,
    *,
    maxiter: int = 40,
    tol: float = 1e-10,
) -> tuple[float, bool, str]:
    """Exact strictly convex 1D stratified Breslow Cox solver via damped Newton-Raphson."""
    g_1d = [
        (Zs[:, 0], grp_ends, d_k, s_k[:, 0], np.arange(len(Zs))[:, None] <= grp_ends[None, :])
        for Zs, grp_ends, d_k, s_k in groups
    ]

    def _eval(b: float) -> tuple[float, float, float]:
        nll = 0.0
        grad = 0.0
        hess = 0.0
        for z, grp_ends, d_k, sk, mask in g_1d:
            eta = z * b
            log_cum = np.logaddexp.accumulate(eta)[grp_ends]
            nll -= float((sk * b).sum() - np.dot(d_k, log_cum))
            diff = np.where(mask, eta[:, None] - log_cum[None, :], -np.inf)
            p = np.exp(diff)  # (n_s, n_evt_grp), columns sum to 1
            ez = p.T @ z
            ez2 = p.T @ (z * z)
            grad -= float(sk.sum() - np.dot(d_k, ez))
            hess += float(np.dot(d_k, ez2 - ez * ez))
        nll = nll * inv_n + 0.5 * l2 * b * b
        grad = grad * inv_n + l2 * b
        hess = max(hess * inv_n + l2, 1e-12)
        return nll, grad, hess

    b = 0.0
    nll, grad, hess = _eval(b)
    for _ in range(maxiter):
        if abs(grad) <= tol:
            return b, True, "ok"
        step = -grad / hess
        step = float(np.clip(step, -10.0, 10.0))
        alpha = 1.0
        accepted = False
        while alpha >= 1e-6:
            cand = b + alpha * step
            c_nll, c_grad, c_hess = _eval(cand)
            if np.isfinite(c_nll) and c_nll <= nll + 1e-4 * alpha * grad * step:
                b, nll, grad, hess = cand, c_nll, c_grad, c_hess
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
    if np.isfinite(b) and abs(grad) < 1e-5:
        return b, True, "ok"
    return float("nan"), False, f"1D Newton Cox did not converge (|grad|={abs(grad):.3e})"


def fit_stratified_cox(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    strata: np.ndarray,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    l2: float = 0.05,
) -> FittedCoxModel:
    """Fit an L2-regularized stratified Breslow Cox model using training patients only."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    _n, d = arr.shape
    cols = list(columns) if columns is not None else [f"f{j}" for j in range(d)]

    medians = np.zeros(d, dtype=np.float64)
    imputed = arr.copy()
    for j in range(d):
        fin = imputed[:, j][np.isfinite(imputed[:, j])]
        med = float(np.median(fin)) if len(fin) else 0.0
        medians[j] = med
        imputed[~np.isfinite(imputed[:, j]), j] = med

    mean = imputed.mean(axis=0)
    std = imputed.std(axis=0)
    Z = np.divide(imputed - mean, std, out=np.zeros_like(imputed), where=std > 0.0)

    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    st = np.asarray(strata, dtype=np.int64)
    active = std > 0.0
    if not e.any() or not active.any():
        return FittedCoxModel(medians, mean, std, np.zeros(d, dtype=np.float64), cols, l2)

    Za = Z[:, active]
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for s in np.unique(st):
        m = st == s
        if not e[m].any():
            continue
        ts, es, Zs = t[m], e[m], Za[m]
        order = np.argsort(-ts, kind="mergesort")
        ts, es, Zs = ts[order], es[order], Zs[order]
        _ut, first_idx, counts = np.unique(-ts, return_index=True, return_counts=True)
        ends = first_idx + counts - 1
        evt_counts = np.add.reduceat(es.astype(np.int64), first_idx)
        nz = evt_counts > 0
        if not nz.any():
            continue
        evt_sums = np.add.reduceat(Zs * es[:, None], first_idx, axis=0)[nz]
        groups.append((Zs, ends[nz], evt_counts[nz].astype(np.float64), evt_sums))

    if not groups:
        return FittedCoxModel(medians, mean, std, np.zeros(d, dtype=np.float64), cols, l2)

    inv_n = 1.0 / max(int(e.sum()), 1)
    if Za.shape[1] == 1:
        b1, ok, msg = _fit_cox_1d_newton(groups, inv_n, l2)
        if not ok:
            return FittedCoxModel(
                medians, mean, std, np.full(d, np.nan, dtype=np.float64), cols, l2,
                converged=False, status_message=msg,
            )
        beta_full = np.zeros(d, dtype=np.float64)
        beta_full[active] = b1
        return FittedCoxModel(medians, mean, std, beta_full, cols, l2, converged=True, status_message="ok")

    res = minimize(
        lambda b: cox_neg_loglik_and_grad(b, groups, inv_n, l2),
        np.zeros(Za.shape[1], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7},
    )
    finite_ok = bool(np.isfinite(res.x).all() and np.isfinite(res.fun))
    grad_ok = bool(res.success or (finite_ok and np.max(np.abs(res.jac)) < 1e-4))
    if not (finite_ok and grad_ok):
        beta_bad = np.full(d, np.nan, dtype=np.float64)
        return FittedCoxModel(
            medians, mean, std, beta_bad, cols, l2,
            converged=False,
            status_message=f"L-BFGS-B did not converge: {getattr(res, 'message', 'unknown')}",
        )
    beta_full = np.zeros(d, dtype=np.float64)
    beta_full[active] = res.x
    return FittedCoxModel(medians, mean, std, beta_full, cols, l2, converged=True, status_message="ok")


def _batch_paired_c_indices(
    t: np.ndarray,
    e: np.ndarray,
    rp_list: list[np.ndarray],
    rf: np.ndarray,
    st: np.ndarray | None,
    fl: np.ndarray | None,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized paired bootstrap C-indices sharing comparability mask across risk vectors."""
    n = len(t)
    d = len(rp_list)
    if n_boot <= 0 or n < 2:
        return np.empty((0, d), dtype=np.float64), np.empty(0, dtype=np.float64)

    cp_out = np.full((n_boot, d), np.nan, dtype=np.float64)
    cf_out = np.full(n_boot, np.nan, dtype=np.float64)
    batch = max(1, min(64, 4_000_000 // max(n * n, 1)))
    rp_stack = np.stack(rp_list, axis=0)  # (d, n)

    for start in range(0, n_boot, batch):
        stop = min(start + batch, n_boot)
        b_len = stop - start
        idx = rng.integers(0, n, size=(b_len, n))
        tb = t[idx]  # (B, n)
        eb = e[idx]
        comp = eb[:, :, None] & ((tb[:, :, None] < tb[:, None, :]) | ((tb[:, :, None] == tb[:, None, :]) & (~eb[:, None, :])))
        if st is not None:
            stb = st[idx]
            comp &= stb[:, :, None] == stb[:, None, :]
        if fl is not None:
            flb = fl[idx]
            comp &= flb[:, :, None] == flb[:, None, :]
        n_comp = comp.sum(axis=(1, 2))
        has_comp = n_comp > 0

        rfb = rf[idx]
        c_full_b = ((rfb[:, :, None] > rfb[:, None, :]) & comp).sum(axis=(1, 2)) + 0.5 * ((rfb[:, :, None] == rfb[:, None, :]) & comp).sum(axis=(1, 2))
        cf_out[start:stop] = np.divide(c_full_b, n_comp, out=np.full(b_len, np.nan), where=has_comp)

        for j in range(d):
            rpb = rp_stack[j][idx]
            cp_b = ((rpb[:, :, None] > rpb[:, None, :]) & comp).sum(axis=(1, 2)) + 0.5 * ((rpb[:, :, None] == rpb[:, None, :]) & comp).sum(axis=(1, 2))
            cp_out[start:stop, j] = np.divide(cp_b, n_comp, out=np.full(b_len, np.nan), where=has_comp)

    return cp_out, cf_out


def paired_retained_c_index(
    time: np.ndarray,
    event: np.ndarray,
    risk_panel: np.ndarray,
    risk_full: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    folds: np.ndarray | None = None,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate within-(fold, stratum) C_panel, C_full, retained = (C_panel - 0.5)/(C_full - 0.5),
    on the joint finite-prediction mask without conditional filtering of denominator <= 0.5 draws.
    """
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    rp = np.asarray(risk_panel, dtype=np.float64)
    rf = np.asarray(risk_full, dtype=np.float64)
    st = np.asarray(strata, dtype=np.int64) if strata is not None else None
    fl = np.asarray(folds, dtype=np.int64) if folds is not None else None

    joint = np.isfinite(t) & np.isfinite(rp) & np.isfinite(rf)
    n_dropped = int((~joint).sum())
    t, e, rp, rf = t[joint], e[joint], rp[joint], rf[joint]
    if st is not None:
        st = st[joint]
    if fl is not None:
        fl = fl[joint]

    c_panel = harrell_c_index(t, e, rp, strata=st, folds=fl)
    c_full = harrell_c_index(t, e, rf, strata=st, folds=fl)

    if not np.isfinite(c_full) or c_full <= 0.5:
        reason = "C_full is not estimable" if not np.isfinite(c_full) else f"C_full <= 0.5 ({c_full:.4f})"
        return {
            "status": "UNMEASURED",
            "reason": reason,
            "c_panel": None if not np.isfinite(c_panel) else float(c_panel),
            "c_full": None if not np.isfinite(c_full) else float(c_full),
            "retained_c_index": None,
            "ci_status": "UNMEASURED",
            "ci_reason": reason,
            "ci_low": None,
            "ci_high": None,
            "c_panel_ci": None,
            "c_full_ci": None,
            "n_boot": int(n_boot),
            "n_boot_nonpositive": int(n_boot),
            "n_patients_eval": int(joint.sum()),
            "n_patients_dropped": n_dropped,
        }
    if not np.isfinite(c_panel):
        return {
            "status": "UNMEASURED",
            "reason": "C_panel is not estimable on outer test patients",
            "c_panel": None,
            "c_full": float(c_full),
            "retained_c_index": None,
            "ci_status": "UNMEASURED",
            "ci_reason": "C_panel is not estimable on outer test patients",
            "ci_low": None,
            "ci_high": None,
            "c_panel_ci": None,
            "c_full_ci": None,
            "n_boot": int(n_boot),
            "n_boot_nonpositive": int(n_boot),
            "n_patients_eval": int(joint.sum()),
            "n_patients_dropped": n_dropped,
        }

    point = (c_panel - 0.5) / (c_full - 0.5)
    rng = np.random.default_rng(seed)
    n = len(t)
    cp_arr, cf_arr = _batch_paired_c_indices(t, e, [rp], rf, st, fl, rng, int(n_boot))
    cp_vals = cp_arr[:, 0]
    cp_all = cp_vals[np.isfinite(cp_vals)]
    cf_all = cf_arr[np.isfinite(cf_arr)]
    good = np.isfinite(cp_vals) & np.isfinite(cf_arr) & (cf_arr > 0.5)
    n_nonpos = int((~good).sum())
    ret_all = (cp_vals[good] - 0.5) / (cf_arr[good] - 0.5)

    c_panel_ci = (
        [float(np.quantile(cp_all, alpha / 2.0)), float(np.quantile(cp_all, 1.0 - alpha / 2.0))]
        if len(cp_all) >= 2
        else None
    )
    c_full_ci = (
        [float(np.quantile(cf_all, alpha / 2.0)), float(np.quantile(cf_all, 1.0 - alpha / 2.0))]
        if len(cf_all) >= 2
        else None
    )

    # Unconditional percentile CI is reported only when zero draws cross C_full <= 0.5.
    if int(n_boot) > 0 and n_nonpos == 0 and len(ret_all) == int(n_boot):
        lo, hi = np.quantile(ret_all, [alpha / 2.0, 1.0 - alpha / 2.0])
        ci_status = "MEASURED"
        ci_reason = None
        ci_low, ci_high = float(lo), float(hi)
    else:
        ci_status = "UNMEASURED"
        ci_reason = f"{n_nonpos}/{int(n_boot)} paired bootstrap draws had C_full <= 0.5 or non-finite C"
        ci_low, ci_high = None, None

    return {
        "status": "MEASURED",
        "c_panel": float(c_panel),
        "c_full": float(c_full),
        "retained_c_index": float(point),
        "ci_status": ci_status,
        "ci_reason": ci_reason,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "c_panel_ci": c_panel_ci,
        "c_full_ci": c_full_ci,
        "n_boot": int(n_boot),
        "n_boot_nonpositive": int(n_nonpos),
        "n_patients_eval": int(joint.sum()),
        "n_patients_dropped": n_dropped,
    }


def paired_retained_c_index_draws(
    time: np.ndarray,
    event: np.ndarray,
    risk_draws: list[np.ndarray],
    risk_full: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    folds: np.ndarray | None = None,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate expected single-draw performance over multiple serial-section misregistration draws."""
    if len(risk_draws) == 1:
        single = paired_retained_c_index(
            time, event, risk_draws[0], risk_full, strata=strata, folds=folds, n_boot=n_boot, seed=seed, alpha=alpha
        )
        single["eval_mode"] = "single_draw"
        single["n_draws"] = 1
        return single

    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=bool)
    rf = np.asarray(risk_full, dtype=np.float64)
    draws_arr = [np.asarray(rd, dtype=np.float64) for rd in risk_draws]
    st = np.asarray(strata, dtype=np.int64) if strata is not None else None
    fl = np.asarray(folds, dtype=np.int64) if folds is not None else None

    joint = np.isfinite(t) & np.isfinite(rf)
    for rd in draws_arr:
        joint &= np.isfinite(rd)
    n_dropped = int((~joint).sum())
    t, e, rf = t[joint], e[joint], rf[joint]
    draws_arr = [rd[joint] for rd in draws_arr]
    if st is not None:
        st = st[joint]
    if fl is not None:
        fl = fl[joint]

    c_full = harrell_c_index(t, e, rf, strata=st, folds=fl)
    draw_cs = [harrell_c_index(t, e, rd, strata=st, folds=fl) for rd in draws_arr]
    if not np.isfinite(c_full) or c_full <= 0.5 or not all(np.isfinite(c) for c in draw_cs):
        reason = "C_full is not estimable" if not np.isfinite(c_full) else (
            f"C_full <= 0.5 ({c_full:.4f})" if c_full <= 0.5 else "per-draw C_panel not estimable"
        )
        return {
            "status": "UNMEASURED",
            "reason": reason,
            "eval_mode": "per_draw_expected",
            "n_draws": len(draws_arr),
            "c_panel": None if not all(np.isfinite(c) for c in draw_cs) else float(np.mean(draw_cs)),
            "c_full": None if not np.isfinite(c_full) else float(c_full),
            "retained_c_index": None,
            "ci_status": "UNMEASURED",
            "ci_reason": reason,
            "ci_low": None,
            "ci_high": None,
            "c_panel_ci": None,
            "c_full_ci": None,
            "n_boot": int(n_boot),
            "n_boot_nonpositive": int(n_boot),
            "n_patients_eval": int(joint.sum()),
            "n_patients_dropped": n_dropped,
        }

    draw_ret = [(c - 0.5) / (c_full - 0.5) for c in draw_cs]
    c_panel = float(np.mean(draw_cs))
    point = float(np.mean(draw_ret))

    rng = np.random.default_rng(seed)
    cp_mat, cf_arr = _batch_paired_c_indices(t, e, draws_arr, rf, st, fl, rng, int(n_boot))
    valid_cp = np.isfinite(cp_mat).all(axis=1)
    cp_mean = np.where(valid_cp, np.mean(cp_mat, axis=1), np.nan)
    cp_all = cp_mean[np.isfinite(cp_mean)]
    cf_all = cf_arr[np.isfinite(cf_arr)]
    good = np.isfinite(cp_mean) & np.isfinite(cf_arr) & (cf_arr > 0.5)
    n_nonpos = int((~good).sum())
    ret_all = (cp_mean[good] - 0.5) / (cf_arr[good] - 0.5)

    c_panel_ci = (
        [float(np.quantile(cp_all, alpha / 2.0)), float(np.quantile(cp_all, 1.0 - alpha / 2.0))]
        if len(cp_all) >= 2
        else None
    )
    c_full_ci = (
        [float(np.quantile(cf_all, alpha / 2.0)), float(np.quantile(cf_all, 1.0 - alpha / 2.0))]
        if len(cf_all) >= 2
        else None
    )
    if int(n_boot) > 0 and n_nonpos == 0 and len(ret_all) == int(n_boot):
        lo, hi = np.quantile(ret_all, [alpha / 2.0, 1.0 - alpha / 2.0])
        ci_status, ci_reason = "MEASURED", None
        ci_low, ci_high = float(lo), float(hi)
    else:
        ci_status = "UNMEASURED"
        ci_reason = f"{n_nonpos}/{int(n_boot)} paired bootstrap draws had C_full <= 0.5 or non-finite C"
        ci_low, ci_high = None, None

    return {
        "status": "MEASURED",
        "eval_mode": "per_draw_expected",
        "n_draws": len(draws_arr),
        "draw_c_panels": [float(c) for c in draw_cs],
        "draw_retained": [float(r) for r in draw_ret],
        "draw_c_panel_std": float(np.std(draw_cs)),
        "c_panel": c_panel,
        "c_full": float(c_full),
        "retained_c_index": point,
        "ci_status": ci_status,
        "ci_reason": ci_reason,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "c_panel_ci": c_panel_ci,
        "c_full_ci": c_full_ci,
        "n_boot": int(n_boot),
        "n_boot_nonpositive": int(n_nonpos),
        "n_patients_eval": int(joint.sum()),
        "n_patients_dropped": n_dropped,
    }
