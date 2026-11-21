"""A-15.6 hard negative (SYNTHETIC): candidates track C, C drives hazard and censoring, X has no effect.

The screen's family-wise rejection rate over independent null datasets should sit
at alpha. The raw-residual shuffle and the unadjusted (zero-covariate) screen run
on the same datasets so a method that ignores C is seen red.
Run: uv run python -m stats.hard_negative [n_datasets] > result.json
"""

import json
import sys

import numpy as np
from scipy.stats import beta as beta_dist

from .screen import _newton, _risk, screen


def dataset(rng, n=120, p=20):
    C = np.column_stack([rng.integers(0, 2, n), rng.normal(size=n)]).astype(float)
    X = C @ rng.uniform(0.5, 1.0, (2, p)) + rng.normal(size=(n, p))  # candidates correlated with C
    strata = rng.integers(0, 2, n)
    event_time = rng.weibull(1.3, n) / np.exp((0.9 * C[:, 0] + 0.6 * C[:, 1]) / 1.3)
    censor = np.minimum(rng.exponential(1.0 / (0.6 * np.exp(0.8 * C[:, 1] - 0.5 * C[:, 0]))), 2.5)
    return X, np.minimum(event_time, censor), (event_time <= censor).astype(float), C, strata


def raw_shuffle_p(X, time, event, C, strata, z, B, rng):
    """The invalid shortcut: shuffle the fitted adjusted residuals within strata, keep V fixed."""
    Cc = C - C.mean(0)
    same = strata[:, None] == strata[None]
    beta, _ = _newton(np, Cc, time[None], event[None], same)
    _, m, _ = _risk(np, time[None], event[None], np.exp(beta @ Cc.T), same)
    m = m[0]
    U_obs = (X - X.mean(0)).T @ m
    scale = np.divide(z, U_obs, out=np.zeros_like(z), where=U_obs != 0)  # 1 / sqrt(V)
    shuffled = np.empty((len(m), B))
    for b in range(B):
        for s in np.unique(strata):
            ix = np.flatnonzero(strata == s)
            shuffled[ix, b] = m[rng.permutation(ix)]
    maxima = np.abs(((X - X.mean(0)).T @ shuffled) * scale[:, None]).max(0)
    return (1 + np.sum(maxima >= np.abs(z).max())) / (B + 1)


def clopper_pearson(k, n, level=0.95):
    a = (1 - level) / 2
    lo = beta_dist.ppf(a, k, n - k + 1) if k else 0.0
    hi = beta_dist.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


def run(n_datasets=300, B=99, alpha=0.05, seed=2026):
    rng = np.random.default_rng(seed)
    rejections = {"bootstrap": 0, "raw_residual_shuffle": 0, "unadjusted": 0}
    for d in range(n_datasets):
        X, time, event, C, strata = dataset(rng)
        adjusted = screen(X, time, event, C, strata, n_perm=B, seed=d, backend="numpy")
        rejections["bootstrap"] += adjusted["p"] <= alpha
        rejections["raw_residual_shuffle"] += raw_shuffle_p(X, time, event, C, strata, adjusted["z"], B, rng) <= alpha
        rejections["unadjusted"] += screen(X, time, event, None, strata, n_perm=B, seed=d, backend="numpy")["p"] <= alpha
    return {
        "label": "SYNTHETIC",
        "fixture": "n=120, 2 strata, C=[binary, normal], 20 candidates correlated with C, "
                   "Weibull hazard on C, censoring on C, no candidate effect given C",
        "alpha": alpha, "B": B, "n_datasets": n_datasets, "seed": seed,
        "rates": {k: {"rejections": int(v), "rate": v / n_datasets,
                      "clopper_pearson_95": clopper_pearson(int(v), n_datasets)}
                  for k, v in rejections.items()},
    }


if __name__ == "__main__":
    json.dump(run(int(sys.argv[1]) if len(sys.argv) > 1 else 300), sys.stdout, indent=1)
