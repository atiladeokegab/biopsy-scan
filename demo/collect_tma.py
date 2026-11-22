"""Fill demo/data/tma.json (preflight-tma/1, the Preflight workflow section) with real Charville cores.

Run from the repo root: DATA_DIR=<dir with charville_labels.csv, charville_raw_data.zip> uv run python -I demo/collect_tma.py [out]
36 cores, one per patient: the hero Charville_c003_v001_r001_reg112 (CHV-24133, the 3D view's region, kept at
full resolution and named by `pick`) plus 35 patients drawn with a seed, each patient's largest region, uniformly
subsampled to at most 1,200 cells. Cell row: [x, y, type, m0..m5], x and y integer microns from the region's
bounding-box min, type indexing type_names, markers 0-255 between each marker's 1st and 99th percentile over the
whole export (the source's own normalised intensities; lo/hi in meta). Charville only; UPMC is never read.
"""
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # python -I drops the cwd
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DATA_DIR", "/home/atilade/code/sota-parallelism-nov21/.worktrees/1-spine/data")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.cohorts import DATA_DIR, PIXEL_UM, load  # noqa: E402
from export_tissue import display_class  # noqa: E402

N_CORES, MAX_CELLS, SEED = 36, 1200, 7
HERO_REGION, HERO_PATIENT = "Charville_c003_v001_r001_reg112", "24133"
MARKERS = {"DAPI": "nuclear", "PanCK": "tumour/epithelial", "CD8": "cytotoxic T cell",
           "CD68": "macrophage", "aSMA": "stromal", "CD20": "B cell"}


def region_cells(z, r):
    """Cells of region r in cell_data order, with type and marker columns."""
    d = pd.read_csv(z.open(f"raw_data/{r}.cell_data.csv"), usecols=["CELL_ID", "X", "Y"])
    t = pd.read_csv(z.open(f"raw_data/{r}.cell_types.csv"), usecols=["CELL_ID", "CELL_TYPE"])
    e = pd.read_csv(z.open(f"raw_data/{r}.expression.csv"), usecols=["CELL_ID", *MARKERS])
    m = d.merge(t, on="CELL_ID", how="inner", validate="one_to_one").merge(e, on="CELL_ID", how="inner", validate="one_to_one")
    if len(m) != len(d):
        raise ValueError(f"{r}: {len(d) - len(m)} cells without a type or marker values")
    return m


def main(out: Path):
    c = load("charville")
    pnames = list(c["patient_names"])
    ptr = np.searchsorted(c["region_id"], np.arange(len(c["patient_of_region"]) + 1))
    hero_p = pnames.index(HERO_PATIENT)
    if HERO_REGION not in [c["region_names"][r] for r in np.flatnonzero(c["patient_of_region"] == hero_p)]:
        raise ValueError(f"{HERO_REGION} is not a region of patient {HERO_PATIENT}")
    rng = np.random.default_rng(SEED)
    others = rng.choice([p for p in range(len(pnames)) if p != hero_p], N_CORES - 1, replace=False)
    picks = [(hero_p, HERO_REGION)]
    for p in sorted(others, key=lambda p: pnames[p]):
        regs = np.flatnonzero(c["patient_of_region"] == p)
        picks.append((p, c["region_names"][regs[np.argmax(ptr[regs + 1] - ptr[regs])]]))

    type_names = sorted(c["type_names"])
    code = {n: i for i, n in enumerate(type_names)}
    cores, raw = [], []
    with zipfile.ZipFile(Path(DATA_DIR) / "charville_raw_data.zip") as z:
        for p, r in picks:
            m = region_cells(z, r)
            hero = r == HERO_REGION
            n_all = len(m)
            if not hero and n_all > MAX_CELLS:
                m = m.iloc[np.sort(rng.choice(n_all, MAX_CELLS, replace=False))]
            # bbox of every cell in the region, so a subsampled core keeps its true size
            xy = m[["X", "Y"]].to_numpy(float) * PIXEL_UM
            raw.append(m[list(MARKERS)].to_numpy(float))
            cores.append({"region": r, "patient": f"CHV-{pnames[p]}", "n_cells": n_all, "xy": xy,
                          "t": [code[n] for n in m.CELL_TYPE]})
            full = pd.read_csv(z.open(f"raw_data/{r}.cell_data.csv"), usecols=["X", "Y"]).to_numpy(float) * PIXEL_UM
            cores[-1]["lo"], cores[-1]["size_um"] = full.min(0), np.ceil(full.max(0) - full.min(0)).astype(int).tolist()
    allv = np.concatenate(raw)
    lo, hi = np.percentile(allv, 1, axis=0), np.percentile(allv, 99, axis=0)
    for core, v in zip(cores, raw):
        q = np.clip(np.round((v - lo) / (hi - lo) * 255), 0, 255).astype(int)
        xy = np.round(core.pop("xy") - core.pop("lo")).astype(int)
        core["cells"] = [[int(a), int(b), t, *map(int, qs)] for (a, b), t, qs in zip(xy, core.pop("t"), q)]

    doc = {
        "schema": "preflight-tma/1", "source": "MEASURED",
        "note": "Charville CODEX, Wu et al. 2022, Zenodo 13179600, CC-BY-4.0; real cells, downsampled",
        "type_names": type_names,
        "type_groups": [display_class(n) for n in type_names],  # the 3D view's tumour/stroma/immune classes
        "markers": list(MARKERS),
        "pick": 0,
        "meta": {"label": "MEASURED cells", "script": "demo/collect_tma.py", "seed": SEED,
                 "selection": f"{N_CORES} cores, one per patient: {HERO_REGION} (CHV-{HERO_PATIENT}, the 3D view's "
                              f"region) plus {N_CORES - 1} patients drawn uniformly (seed {SEED}), each patient's largest region",
                 "downsampling": f"uniform random sample of at most {MAX_CELLS} cells per core without replacement, "
                                 f"seed {SEED}; the pick core keeps every cell",
                 "units": "x, y integer microns from the region's bounding-box min; size_um the bounding box",
                 "marker_roles": MARKERS,
                 "marker_scale": "q = clip(round((v - lo) / (hi - lo) * 255), 0, 255); v the source's normalised "
                                 "intensity, lo/hi its 1st/99th percentile over the exported cells",
                 "marker_lo": [round(float(a), 4) for a in lo], "marker_hi": [round(float(b), 4) for b in hi]},
        "cores": cores}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")))
    n = sum(len(k["cells"]) for k in cores)
    print(f"{out}: {len(cores)} cores, {n} cells, {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo" / "data" / "tma.json")
