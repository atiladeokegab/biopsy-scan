"""Export real Charville cells for the demo's 3D tissue view: demo/data/tissue.json (preflight-tissue/1).

Run from the repo root (needs core/ and `make data`): uv run python -I scripts/export_tissue.py [out]
12 patients spanning the outcome range (6 deaths, 6 censored, spread over follow-up time), one region
each (the patient's largest), subsampled to about 3k cells with a per-type floor so rare immune
cells still show. Coordinates are microns from the region's corner.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # python -I drops the cwd

import numpy as np  # noqa: E402

from core.cohorts import load  # noqa: E402

N_PATIENTS, TARGET, FLOOR, SEED = 12, 3000, 60, 7

# Display classes for the page's three colours. "Tumor 6 / DC" is labelled tumour by the source;
# "Other" (unassigned) is drawn as stroma, the non-tumour, non-immune class.
DISPLAY = {"Stroma": "stroma", "Blood vessel": "stroma", "Other": "stroma",
           "B cell": "immune", "CD4 T cell": "immune", "CD8 T cell": "immune",
           "Macrophage": "immune", "Granulocyte": "immune"}


def display_class(name: str) -> str:
    return "tumour" if name.startswith("Tumor") else DISPLAY[name]  # KeyError on an unmapped type


def pick_patients(time, event, n):
    """n/2 deaths and n/2 censored, each spread evenly over its follow-up order."""
    out = []
    for flag in (True, False):
        idx = np.flatnonzero(event == flag)
        idx = idx[np.argsort(time[idx], kind="stable")]
        out += idx[np.linspace(0, len(idx) - 1, n // 2).round().astype(int)].tolist()
    return sorted(out, key=lambda p: time[p])


def subsample(types, rng):
    """Per type: its proportional share of TARGET, at least FLOOR (or all it has)."""
    keep = []
    for k in np.unique(types):
        idx = np.flatnonzero(types == k)
        n = min(len(idx), max(FLOOR, round(TARGET * len(idx) / len(types))))
        keep.append(rng.choice(idx, n, replace=False))
    return np.sort(np.concatenate(keep))


def main(out: Path):
    c = load("charville")
    names = c["type_names"]
    cls = np.array([display_class(n) for n in names])
    ptr = np.searchsorted(c["region_id"], np.arange(len(c["patient_of_region"]) + 1))
    rng = np.random.default_rng(SEED)
    patients, side = [], 0.0
    for p in pick_patients(c["time"], c["event"], N_PATIENTS):
        regs = np.flatnonzero(c["patient_of_region"] == p)
        r = regs[np.argmax(ptr[regs + 1] - ptr[regs])]
        sl = slice(ptr[r], ptr[r + 1])
        x, y, t = c["x"][sl].astype(float), c["y"][sl].astype(float), c["cell_type"][sl]
        x, y = x - x.min(), y - y.min()
        side = max(side, x.max(), y.max())
        keep = subsample(t, rng)
        patients.append({
            "id": f"CHV-{c['patient_names'][p]}",
            "outcome": {"event": bool(c["event"][p]), "time_days": float(c["time"][p])},
            "region": c["region_names"][r], "cells_in_region": int(len(t)),
            "cells": [{"x": int(round(x[i])), "y": int(round(y[i])), "type": str(cls[t[i]])} for i in keep]})
    doc = {
        "schema": "preflight-tissue/1", "source": "MEASURED",
        "note": "Charville CODEX (Zenodo 13179600, CC-BY-4.0; Wu et al. 2022), real cells, subsampled.",
        "region_um": float(np.ceil(side / 10) * 10),
        "meta": {"source": "Charville CODEX (Zenodo, CC-BY-4.0), real cells, subsampled",
                 "selection": f"{N_PATIENTS} patients, {N_PATIENTS // 2} deaths and {N_PATIENTS // 2} censored "
                              "spread over follow-up; each patient's largest region",
                 "subsample": f"per cell type: proportional share of {TARGET}, at least {FLOOR} (or all), seed {SEED}",
                 "display_classes": {n: str(k) for n, k in zip(names, cls)},
                 "microns_per_pixel": 0.377, "script": "scripts/export_tissue.py"},
        "patients": patients}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")))
    n = sum(len(p["cells"]) for p in patients)
    print(f"{out}: {len(patients)} patients, {n} cells, region_um {doc['region_um']}, {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo" / "data" / "tissue.json")
