"""Export the biomarker panel's data: demo/data/biomarker.json (preflight-biomarker/1), from Charville v2.

Run from the repo root (needs `make data`):
  uv run python -I scripts/export_biomarker.py [--out PATH]

The feature is results/charville-v2.json's top hit, Tumor 5|Macrophage|100um|PD1: for each Tumor 5 cell,
the mean cohort z-scored PD1 of the macrophages within 100 um, averaged over the patient's Tumor 5 cells
(core.truth.truth_v2). Exploratory: the outcomes were already seen, and the hit is not significant after
correcting for every pattern tested.

KM: patients with a finite value, high = value above the cohort median, low = at or below it. No p-value:
the cutoff and the feature were chosen on outcomes already seen.
Regions, one per arm, chosen by rule, never by eye: the arm's median patient (sort the arm by value, then
patient ID, and take index (n - 1) // 2), then that patient's region whose own value is closest to the
patient's (ties: label order).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # python -I drops the cwd

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from core import cohorts  # noqa: E402
from core.grammar import RADII_UM, feature_names, grammar_v2  # noqa: E402
from core.truth import truth_region_v2, truth_v2  # noqa: E402

RESULT = ROOT / "results" / "charville-v2.json"
FOCAL, NEIGHBOUR, RADIUS, CHANNEL = "Tumor 5", "Macrophage", 100, "PD1"
LABEL = "exploratory, cutoff chosen on the same cohort, no p-value"  # the panel shows this verbatim
PD1_POSITIVE_Z = 1.0  # display only; the feature itself uses the continuous z-score


def km(time: np.ndarray, event: np.ndarray) -> list[list[float]]:
    """Kaplan-Meier steps [[day, survival], ...], starting at [0, 1], one step per distinct event day."""
    curve, s = [[0.0, 1.0]], 1.0
    for t in np.unique(time[event]):
        s *= 1 - np.sum((time == t) & event) / np.sum(time >= t)
        curve.append([float(t), round(float(s), 4)])
    return curve


def _region(cells, p, xp, patient, t, j, r, ch, ptr, K, focal, names) -> dict:
    """The patient's region closest to its value, as drawable columns (µm from the region's corner)."""
    best = None
    for reg in np.flatnonzero(cells["patient_of_region"] == p):
        sl = slice(ptr[reg], ptr[reg + 1])
        xy = np.stack([cells["x"][sl], cells["y"][sl]], 1).astype(np.float64)
        _, s_expr, m = truth_region_v2(xy, cells["cell_type"][sl], cells["channels"][sl], focal, K)
        v = s_expr[t, j, r, ch] / m[t] if m[t] else np.nan
        if np.isfinite(v) and (best is None or abs(v - xp) < abs(best[1] - xp)):
            best = (reg, float(v))
    reg, v = best
    sl = slice(ptr[reg], ptr[reg + 1])
    xs, ys, ty = cells["x"][sl].astype(np.float64), cells["y"][sl].astype(np.float64), cells["cell_type"][sl]
    tum, mac = ty == names.index(FOCAL), ty == names.index(NEIGHBOUR)
    near = np.zeros(len(ty), bool)
    if tum.any():
        d, _ = cKDTree(np.stack([xs[tum], ys[tum]], 1)).query(np.stack([xs, ys], 1), distance_upper_bound=RADIUS)
        near = mac & (d <= RADIUS)
    pos = mac & (cells["channels"][sl, ch] > PD1_POSITIVE_Z)
    x0, y0 = xs.min(), ys.min()
    return {
        "patient": str(patient), "region_id": str(cells["region_names"][reg]),
        "patient_feature_value": round(float(xp), 4), "feature_value": round(v, 4),
        "width_um": round(float(xs.max() - x0), 1), "height_um": round(float(ys.max() - y0), 1),
        "type_names": [FOCAL, NEIGHBOUR, "other"],
        "counts": {"cells": int(len(ty)), "tumour": int(tum.sum()), "macrophages": int(mac.sum()),
                   "pd1_pos_macrophages": int(pos.sum()), "near_tumour_macrophages": int(near.sum()),
                   "pd1_pos_near_tumour": int((pos & near).sum())},
        "cells": {"x": np.round(xs - x0, 1).tolist(), "y": np.round(ys - y0, 1).tolist(),
                  "type": np.where(tum, 0, np.where(mac, 1, 2)).tolist(),
                  "pd1_pos": pos.astype(int).tolist(), "near_tumour": near.astype(int).tolist()},
    }


def build(data_dir: Path = cohorts.DATA_DIR) -> dict:
    hit = json.loads(RESULT.read_text())["maxT"]["top_hits"][0]
    cells = cohorts.load("charville", data_dir, include_channels=True)
    names = list(cells["type_names"])
    g = grammar_v2(names)
    name = f"{FOCAL}|{NEIGHBOUR}|{RADIUS}um|{CHANNEL}"
    col = feature_names(g).index(name)
    if hit["feature"] != name or hit["grammar_index"] != col:
        raise ValueError(f"top hit {hit['feature']} at {hit['grammar_index']} is not {name} at {col}")
    x = truth_v2(cells)[:, col]
    pats = np.asarray(cells["patient_names"])

    ok = np.flatnonzero(np.isfinite(x))
    med = float(np.median(x[ok]))
    time, event = np.asarray(cells["time"]), np.asarray(cells["event"], bool)
    t, j, r = g["focal_types"].index(names.index(FOCAL)), names.index(NEIGHBOUR), list(RADII_UM).index(RADIUS)
    ch = list(cells["channel_names"]).index(CHANNEL)
    ptr = np.searchsorted(cells["region_id"], np.arange(len(cells["region_names"]) + 1))
    arms, regions = [], []
    for arm, in_arm in (("high", x[ok] > med), ("low", x[ok] <= med)):
        idx = ok[in_arm]
        arms.append({"name": arm, "n": int(len(idx)), "events": int(event[idx].sum()), "curve": km(time[idx], event[idx])})
        order = sorted(idx, key=lambda q: (x[q], pats[q]))
        p = order[(len(order) - 1) // 2]
        regions.append(dict(arm=arm, patient_rank_in_arm=f"{order.index(p) + 1} of {len(order)}",
                            **_region(cells, p, x[p], pats[p], t, j, r, ch, ptr, len(names), g["focal_types"], names)))

    return {
        "schema": "preflight-biomarker/1", "source": "MEASURED", "exploratory": True, "already_seen": True,
        "label": LABEL,
        "note": (f"Exploratory: outcomes already seen, KM cutoff chosen on the same cohort, no p-value. Corrected for all {json.loads(RESULT.read_text())['n_features_tested']:,} "
                 f"tests, adjusted p = {hit['adjusted_p']:.3f}: not significant."),
        "feature": {"name": name, "z": round(hit["z"], 3), "adjusted_p": hit["adjusted_p"], "source": "results/charville-v2.json",
                    "definition": "per Tumor 5 cell, the mean cohort z-scored PD1 of macrophages within 100 µm; "
                                  "averaged over the patient's Tumor 5 cells"},
        "cohort": {"name": "Charville", "patients": int(len(pats)), "patients_with_value": int(len(ok)),
                   "events": int(event.sum())},
        "selection_rule": ("per KM arm, the arm's median patient: sort the arm by (value, patient ID), take index "
                           "(n - 1) // 2; then that patient's region whose own value is closest to the patient's"),
        "pd1_positive_rule": f"PD1 cohort z-score > {PD1_POSITIVE_Z:g} (display only; the feature uses the continuous z-score)",
        "regions": regions,
        "km": {"label": LABEL, "endpoint": "death; days from surgery to last contact", "cutoff": "the cohort median of the feature",
               "cutoff_value": round(med, 4), "p_value": None, "arms": arms},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "demo" / "data" / "biomarker.json")
    ap.add_argument("--data-dir", type=Path, default=cohorts.DATA_DIR)
    a = ap.parse_args(argv)
    b = build(a.data_dir)
    a.out.write_text(json.dumps(b, separators=(",", ":"), allow_nan=False) + "\n")
    for r in b["regions"]:
        print(f"{a.out}: {r['arm']} arm patient {r['patient']} ({r['patient_rank_in_arm']}), {r['region_id']}, {r['counts']}")
    print("km:", [(x["name"], x["n"], x["events"]) for x in b["km"]["arms"]], "cutoff", b["km"]["cutoff_value"])


if __name__ == "__main__":
    main()
