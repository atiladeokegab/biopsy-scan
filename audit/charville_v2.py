"""Real-data Charville v1+v2 exploratory audit (Issue #41, grammar v2)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

if __name__ == "__main__" and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.leakage import leakage_from_cells
from audit.runner import leave_one_patient_out
from audit.schema import (
    ALREADY_SEEN_DISCLOSURE,
    AUDIT_SCHEMA_VERSION,
    choose_prepared_outcome,
)

CONDITIONAL_CUTOFF_SCOPE = "conditional_within_stratum_permutation_not_reusable_design_calibration"


def _conditional_cutoff(null_z: np.ndarray, alpha: float = 0.05) -> float | None:
    arr = np.asarray(null_z, dtype=np.float64)
    if arr.ndim != 2 or len(arr) < 1 or not np.isfinite(arr).all():
        raise ValueError("null_z must be a finite non-empty 2D array")
    maxima = np.max(np.abs(arr), axis=1)
    k = math.ceil((len(maxima) + 1) * (1.0 - alpha))
    return float(np.partition(maxima, k - 1)[k - 1]) if k <= len(maxima) else None
from core import cohorts
from core.grammar import (
    EXPRESSION_CHANNELS,
    EXPRESSION_TRANSFORM,
    feature_names,
    grammar,
    grammar_hash,
    grammar_v2,
    n_features,
)
from core.truth import truth_v2
from stats import reference as stats_ref

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "charville-v2.json"


def _hardware_note() -> dict[str, Any]:
    gpu = None
    if shutil.which("nvidia-smi"):
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        gpu = q.splitlines()[0] if q else None
    return {"cpu": platform.processor() or platform.machine(), "gpu": gpu}


def run_charville_v2(
    *,
    data_dir: Path | None = None,
    out_path: Path | None = DEFAULT_OUT,
    n_resamples: int = 9999,
    seed: int = 0,
    top_k: int = 10,
    load_fn=cohorts.load,
    truth_fn=truth_v2,
    perm_fn=stats_ref.permutation_test,
    score_fn=stats_ref.score,
) -> dict[str, Any]:
    """Run exploratory maxT + omnibus over combined v1+v2 features on Charville."""
    t0 = time.perf_counter()
    kwargs = {"data_dir": data_dir} if data_dir is not None else {}
    cells = load_fn("charville", include_channels=True, **kwargs)
    type_names = list(cells["type_names"])
    g1 = grammar(type_names)
    g2 = grammar_v2(type_names)
    fnames = feature_names(g2)
    n_v1 = n_features(g1)
    n_tot = n_features(g2)

    x_full = np.asarray(truth_fn(cells), dtype=np.float64)
    time_arr = np.asarray(cells["time"], dtype=np.float64)
    event = np.asarray(cells["event"], dtype=np.float64)
    strata = np.asarray(cells["strata"], dtype=np.int32)
    patient_names = [str(p) for p in cells["patient_names"]]

    nan_count = int(np.isnan(x_full).sum())
    perm = perm_fn(x_full, time_arr, event, strata, n_resamples=n_resamples, seed=seed)
    z_obs = np.asarray(perm["z"], dtype=np.float64)
    adj_p = np.asarray(perm["adjusted_p"], dtype=np.float64)

    order = sorted(
        (i for i in range(len(fnames)) if np.isfinite(z_obs[i])),
        key=lambda idx: (-abs(float(z_obs[idx])), idx),
    )[:top_k]

    lopo = leave_one_patient_out(
        X=x_full,
        time=time_arr,
        event=event,
        strata=strata,
        patient_names=patient_names,
        observed_z=z_obs,
        top_indices=order,
        score_fn=score_fn,
    )
    top_hits = [
        {
            "rank": rank,
            "family": "v1" if idx < n_v1 else "v2",
            "feature": fnames[idx],
            "grammar_index": int(idx),
            "z": float(z_obs[idx]),
            "adjusted_p": float(adj_p[idx]),
            "lopo": lopo[idx],
        }
        for rank, idx in enumerate(order, start=1)
    ]

    leakage = leakage_from_cells(cells)
    top_hit_stable = bool(
        top_hits
        and top_hits[0]["lopo"]["sign_flips"] == 0
        and (top_hits[0]["lopo"]["rank1_retained_fraction"] or 0.0) >= 0.8
    )
    maxt_p = float(perm["p"])
    omnibus_p = float(perm["omnibus"]["p"])
    wall_s = time.perf_counter() - t0
    hw = _hardware_note()

    record: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "cohort": "charville",
        "exploratory": True,
        "already_seen": True,
        "already_seen_disclosure": ALREADY_SEEN_DISCLOSURE,
        "clinical_claims": False,
        "inference_scope": "patient_level_exploratory",
        "grammar_v1_hash": grammar_hash(g1),
        "grammar_v2_hash": grammar_hash(g2),
        "channels": list(EXPRESSION_CHANNELS),
        "m": len(EXPRESSION_CHANNELS),
        "transform": EXPRESSION_TRANSFORM,
        "n_patients": int(len(patient_names)),
        "n_events": int(np.sum(event)),
        "n_regions": int(len(cells["region_names"])),
        "n_features_v1": int(n_v1),
        "n_features_v2_expr": int(n_tot - n_v1),
        "n_features_total": int(n_tot),
        "n_features_tested": int(x_full.shape[1]),
        "nan_policy": f"complete_case_per_feature ({nan_count} NaN cells preserved)" if nan_count else "all_finite",
        "strata_note": "Coverslip strata c001-c004 mapped to 0-3 (patient-level cohort; 38 events).",
        "leakage_audit": leakage,
        "maxT": {
            "statistic": float(perm["maxT"]),
            "p": maxt_p,
            "conditional_cutoff_alpha_05": _conditional_cutoff(perm["null_z"], alpha=0.05),
            "cutoff_scope": CONDITIONAL_CUTOFF_SCOPE,
            "n_resamples": int(n_resamples),
            "top_hits": top_hits,
        },
        "omnibus": {
            "statistics": {k: float(v) for k, v in perm["omnibus"]["statistics"].items()},
            "component_p": {k: float(v) for k, v in perm["omnibus"]["component_p"].items()},
            "p": omnibus_p,
        },
        "prepared_outcome": choose_prepared_outcome(
            cohort="charville",
            maxt_p=maxt_p,
            omnibus_p=omnibus_p,
            top_hit_stable=top_hit_stable,
        ),
        "runtime": {
            "status": "MEASURED",
            "backend": "numpy_reference",
            "wall_s": wall_s,
            "hardware": hw,
        },
        "gpu_timing": {
            "status": "UNMEASURED",
            "reason": "no CUDA device on CPU run host" if hw["gpu"] is None else "not timed by CPU truth audit",
        },
        "labels": {
            "cohort_counts": "MEASURED",
            "grammar_hashes": "MEASURED",
            "maxT": "MEASURED",
            "omnibus": "MEASURED",
            "lopo": "MEASURED",
            "leakage_audit": "MEASURED",
            "runtime": "MEASURED",
            "gpu_timing": "UNMEASURED",
        },
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Charville v1+v2 exploratory audit (Issue #41).")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-resamples", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rec = run_charville_v2(data_dir=args.data_dir, out_path=args.out, n_resamples=args.n_resamples, seed=args.seed)
    hit = rec["maxT"]["top_hits"][0] if rec["maxT"]["top_hits"] else None
    print(
        f"charville-v2: P={rec['n_patients']} ({rec['n_events']} events), "
        f"m={rec['m']} channels, C={rec['n_features_total']}, "
        f"maxT={rec['maxT']['statistic']:.3f} (p={rec['maxT']['p']:.4f}), "
        f"omnibus p={rec['omnibus']['p']:.4f}, top={hit['family'] if hit else 'none'} {hit['feature'] if hit else ''}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
