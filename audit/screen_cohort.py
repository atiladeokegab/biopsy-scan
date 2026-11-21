"""R-18: the discovery screen on any cohort, with no cohort-specific branch (A-18.1).

    uv run python -m audit.screen_cohort charville   # writes results/charville-screen.json

X is the v1+v2 truth matrix (3,780 features), there are no clinical covariates, and the strata
are the cohort's coverslip groups. Outcomes have already been seen, so every result is exploratory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import stats
from core import cohorts
from core.grammar import feature_names, grammar_v2
from core.truth import truth_v2
from stats import reference as stats_ref

ROOT = Path(__file__).resolve().parents[1]


def _shim_screen(X, time, event, clinical=None, strata=None, n_perm=999, backend="auto", seed=0):
    """Stand-in for R-15's stats.screen until it merges: the CPU reference maxT, zero covariates only."""
    # shortcut: no clinical covariates or GPU here; drop this shim when R-15's stats.screen lands.
    if clinical is not None:
        raise NotImplementedError("the shim has no clinical adjustment; needs R-15")
    strata = np.zeros(len(time), np.int32) if strata is None else strata
    out = stats_ref.permutation_test(X, time, event, strata, n_resamples=n_perm, seed=seed)
    return {"z": out["z"], "adjusted_p": out["adjusted_p"], "p": out["p"], "backend": "numpy_reference (shim)"}


screen = getattr(stats, "screen", None) or _shim_screen


def run(cohort: str, *, data_dir: Path = cohorts.DATA_DIR, n_perm: int = 999, seed: int = 0, top_k: int = 10,
        load_fn=cohorts.load, truth_fn=truth_v2, screen_fn=None) -> dict:
    t0 = time.perf_counter()
    cells = load_fn(cohort, data_dir=data_dir, include_channels=True)
    names = feature_names(grammar_v2(list(cells["type_names"])))
    X = np.asarray(truth_fn(cells), np.float64)
    if X.shape[1] != len(names):
        raise ValueError(f"truth matrix has {X.shape[1]} columns but the v2 grammar names {len(names)}")
    t, e = np.asarray(cells["time"], np.float64), np.asarray(cells["event"], np.float64)
    out = (screen_fn or screen)(X, t, e, clinical=None, strata=np.asarray(cells["strata"], np.int32),
                                n_perm=n_perm, backend="auto", seed=seed)
    z, adj = np.asarray(out["z"], np.float64), np.asarray(out["adjusted_p"], np.float64)
    top = sorted((i for i in range(len(names)) if np.isfinite(z[i])), key=lambda i: (-abs(z[i]), i))[:top_k]
    rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "cohort": cohort, "status": "MEASURED", "inference": "exploratory (outcomes already seen)",
        "n_patients": int(len(t)), "n_events": int(e.sum()), "n_features_tested": int(X.shape[1]),
        "clinical_covariates": None, "strata": "coverslip groups", "n_perm": n_perm, "seed": seed,
        "maxT": {"p": float(out["p"]), "statistic": float(np.nanmax(np.abs(z)))},
        "top_candidates": [{"rank": k, "feature": names[i], "z": float(z[i]), "adjusted_p": float(adj[i])}
                           for k, i in enumerate(top, 1)],
        "backend": out.get("backend", "auto"), "code_revision": rev or "unknown",
        "wall_s": time.perf_counter() - t0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort", choices=cohorts.COHORTS)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    res = run(a.cohort, n_perm=a.n_perm, seed=a.seed)
    path = ROOT / "results" / f"{a.cohort}-screen.json"
    path.write_text(json.dumps(res, indent=2) + "\n")
    print(path, "maxT p", res["maxT"]["p"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
