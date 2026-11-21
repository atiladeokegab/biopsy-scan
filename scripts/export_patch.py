"""Export one real Charville patch and a KM split for the top exploratory hit: demo/data/biomarker.json.

Run from the repo root (needs core/ and DATA_DIR): uv run python -I scripts/export_biomarker.py [out]

Feature "Tumor 5|Macrophage|100um|PD1" (results/charville-v2.json, maxT rank 1, grammar index 2600).
Definition, from the code: core/truth.py::truth_expression_sums + region_sums_v2, divided by the
patient's focal count in core/grammar.py::features_v2. Per Tumor 5 cell: mean cohort-z-scored PD1
(core/cohorts.py::load, include_channels=True) over its Macrophage neighbours within 100 um (pairs by
cKDTree, distance <= 100 um), 0 when it has none. Patient value = sum over all the patient's Tumor 5
cells (regions pooled) / their count; NaN for a patient with no Tumor 5 cells.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # python -I drops the cwd

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from core.cohorts import load  # noqa: E402
from core.grammar import RADII_UM, feature_names, grammar_v2  # noqa: E402
from core.truth import truth_region_v2  # noqa: E402

FEATURE, RADIUS, WINDOW, STEP, MAX_BYTES = "Tumor 5|Macrophage|100um|PD1", 100.0, 500.0, 25.0, 250_000
IMMUNE = {"B cell", "CD4 T cell", "CD8 T cell", "Granulocyte"}


def t5_mac_pairs(xy, types, t5, mac):
    """(t5 index, mac index) pairs within RADIUS, same distance rule as truth_expression_sums."""
    a, b = np.flatnonzero(types == t5), np.flatnonzero(types == mac)
    if not len(a) or not len(b):
        return np.zeros(0, int), np.zeros(0, int)
    hits = cKDTree(xy[b]).query_ball_point(xy[a], RADIUS)
    i = np.repeat(a, [len(h) for h in hits])
    j = b[np.concatenate([np.asarray(h, int) for h in hits])]
    keep = np.sqrt(((xy[i] - xy[j]) ** 2).sum(1)) <= RADIUS
    return i[keep], j[keep]


def km(time, event):
    """Kaplan-Meier step points [[t, S(t)], ...] from [0, 1], one point per event time."""
    steps, s = [[0.0, 1.0]], 1.0
    for t in np.unique(time[event]):
        d, n = np.sum((time == t) & event), np.sum(time >= t)
        s *= 1 - d / n
        steps.append([float(t), round(float(s), 4)])
    return steps


def main(out: Path):
    c = load("charville", include_channels=True)
    names = [str(n) for n in c["type_names"]]
    assert feature_names(grammar_v2(names))[2600] == FEATURE and RADII_UM[2] == RADIUS
    t5, mac, pd1 = names.index("Tumor 5"), names.index("Macrophage"), c["channel_names"].index("PD1")
    z = c["channels"][:, pd1].astype(np.float64)
    ptr = np.searchsorted(c["region_id"], np.arange(len(c["patient_of_region"]) + 1))
    P = len(c["time"])
    S, m = np.zeros(P), np.zeros(P)
    best = (-1,)
    for r in range(len(ptr) - 1):
        sl = slice(ptr[r], ptr[r + 1])
        xy = np.stack([c["x"][sl], c["y"][sl]], 1).astype(np.float64)
        ty, zr = c["cell_type"][sl], z[sl]
        i, j = t5_mac_pairs(xy, ty, t5, mac)
        nt5 = np.sum(ty == t5)
        cnt, tot = np.bincount(i, minlength=len(ty)), np.bincount(i, zr[j], minlength=len(ty))
        p = c["patient_of_region"][r]
        S[p] += np.divide(tot, cnt, out=np.zeros(len(ty)), where=cnt > 0)[ty == t5].sum()
        m[p] += nt5
        # Window score: PD1+ macrophages within RADIUS of a Tumor 5 cell, both inside the window.
        pos = zr[j] > 0
        if not pos.any():
            continue
        jp, pi, pj = j[pos], xy[i[pos]], xy[j[pos]]
        x0, y0 = xy.min(0)
        for wx in np.arange(x0, max(x0, xy[:, 0].max() - WINDOW) + STEP, STEP):
            for wy in np.arange(y0, max(y0, xy[:, 1].max() - WINDOW) + STEP, STEP):
                lo = np.array([wx, wy])
                both = np.all((pi >= lo) & (pi < lo + WINDOW) & (pj >= lo) & (pj < lo + WINDOW), 1)
                n = len(np.unique(jp[both]))
                if n > best[0]:
                    best = (n, r, wx, wy)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = S / m

    n_best, r, wx, wy = best
    sl = slice(ptr[r], ptr[r + 1])
    xy = np.stack([c["x"][sl], c["y"][sl]], 1).astype(np.float64)
    ty, zr = c["cell_type"][sl], z[sl]
    # Cross-check this region against the shared truth layer the audit used.
    sc, se, m_ = truth_region_v2(xy, ty, c["channels"][sl], [k for k, n in enumerate(names) if n.startswith("Tumor")], len(names))
    i, j = t5_mac_pairs(xy, ty, t5, mac)
    cnt, tot = np.bincount(i, minlength=len(ty)), np.bincount(i, zr[j], minlength=len(ty))
    mine = np.divide(tot, cnt, out=np.zeros(len(ty)), where=cnt > 0)[ty == t5].sum()
    focal = [k for k, n in enumerate(names) if n.startswith("Tumor")]
    assert np.isclose(mine, se[focal.index(t5), mac, 2, pd1], rtol=1e-9, atol=1e-9), "feature mismatch vs truth layer"

    lo = np.array([wx, wy])
    inside = np.flatnonzero(np.all((xy >= lo) & (xy < lo + WINDOW), 1))
    near = set(j[np.isin(i, inside) & np.isin(j, inside)].tolist())

    def kind(k):
        n = names[ty[k]]
        if ty[k] == t5:
            return "t5"
        if n.startswith("Tumor"):
            return "tumour"
        if ty[k] == mac:
            return "mac_pd1" if zr[k] > 0 else "mac"
        return "immune" if n in IMMUNE else "other"

    cells = [{"x": round(float(xy[k, 0] - wx), 1), "y": round(float(xy[k, 1] - wy), 1), "k": kind(k)} for k in inside]

    ok = np.isfinite(x)
    tt, ee, xx = c["time"][ok], c["event"][ok], x[ok]
    med = float(np.median(xx))
    arms = {}
    for name, sel in (("high", xx > med), ("low", xx <= med)):
        arms[name] = {"n": int(sel.sum()), "events": int(ee[sel].sum()), "steps": km(tt[sel], ee[sel])}
    doc = {
        "schema": "preflight-biomarker/1", "source": "MEASURED", "feature": FEATURE,
        "patient": f"CHV-{c['patient_names'][c['patient_of_region'][r]]}", "region": c["region_names"][r],
        "window_um": WINDOW, "radius_um": RADIUS, "cells": cells,
        "pd1_rule": "A macrophage is drawn PD1+ when its PD1 is above the cohort mean (cohort z-score > 0, the same z the "
                    "feature averages; about 20% of Charville macrophages). The feature itself uses no threshold: it is "
                    "the mean PD1 z of each Tumor 5 cell's macrophages within 100 um, averaged over the patient's Tumor 5 cells.",
        "window_rule": f"{WINDOW:.0f} um square on a {STEP:.0f} um grid over every region; picked for the most PD1+ "
                       f"macrophages within {RADIUS:.0f} um of a Tumor 5 cell, both inside the window ({n_best}); "
                       "first such window in region, x, y order.",
        "km": {"split": "median of the per-patient feature value", "median": round(med, 4),
               "split_rule": "high = value > median, low = value <= median",
               "time_unit": "days (surgery to last contact)", "n_dropped_no_tumor5": int((~ok).sum()),
               **arms, "label": "MEASURED", "note": "exploratory; already-seen cohort; not a discovery"},
        "meta": {"source": "Charville CODEX (Zenodo 13179600, CC-BY-4.0; Wu et al. 2022), real cells",
                 "hit": "results/charville-v2.json maxT rank 1: z 3.957, adjusted p 0.098, exploratory",
                 "script": "scripts/export_biomarker.py"}}
    text = json.dumps(doc, separators=(",", ":"))
    if len(text) > MAX_BYTES:  # shortcut: never hit at 500 um; subsample "other" if a denser window ever needs it
        raise SystemExit(f"{len(text)} bytes > {MAX_BYTES}; subsample 'other'")

    # Self-check.
    assert cells and any(q["k"] == "t5" for q in cells) and any(q["k"] == "mac_pd1" for q in cells)
    assert all(0 <= q["x"] <= WINDOW and 0 <= q["y"] <= WINDOW for q in cells)
    for a in ("high", "low"):
        st = np.array(doc["km"][a]["steps"])
        assert st[0].tolist() == [0, 1] and np.all(np.diff(st[:, 0]) >= 0) and np.all(np.diff(st[:, 1]) <= 0)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    counts = {k: sum(q["k"] == k for q in cells) for k in ("t5", "tumour", "mac_pd1", "mac", "immune", "other")}
    print(f"{out}: {doc['patient']} {doc['region']} window ({wx:.1f},{wy:.1f}) {counts}; "
          f"km high {arms['high']['n']}/{arms['high']['events']} low {arms['low']['n']}/{arms['low']['events']}; "
          f"{out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo" / "data" / "biomarker.json")
