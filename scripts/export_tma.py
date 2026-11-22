"""Export real Charville cores for the demo's Preflight workflow section: demo/data/tma.json (preflight-tma/1).

Run from the repo root (needs core/ and `make data`): uv run python -I scripts/export_tma.py [out]
36 cores, one per patient (the hero patient plus 35 drawn with a seed; each patient's largest region).
Grid cores are uniformly subsampled to at most 2,500 cells; the hero core keeps every cell.
Columnar per core: x, y in microns from the region's corner (0.1 um), t = index into `types`, and
per-marker intensities quantised to 0-255 between the exported cells' 1st and 99th percentile of the
source's own normalised values (value = lo + q / 255 * (hi - lo), clipped). Charville only; UPMC is never read.
"""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # python -I drops the cwd
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.cohorts import DATA_DIR, PIXEL_UM, load  # noqa: E402
from export_tissue import display_class  # noqa: E402

N_CORES, MAX_CELLS, SEED = 36, 2500, 7
HERO_REGION, HERO_PATIENT = "Charville_c003_v001_r001_reg112", "24133"
# Nuclear stain, tumour/epithelial, pan-immune, stromal: the four channels the IHC re-render uses.
MARKERS = {"DAPI": "nuclear", "PanCK": "tumour/epithelial", "CD45": "immune", "Vimentin": "stromal"}


def region_cells(z, r):
    """Cells of region r in load_cells order (cell_data order), with the marker columns."""
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
    rng = np.random.default_rng(SEED)
    others = rng.choice([p for p in range(len(pnames)) if p != hero_p], N_CORES - 1, replace=False)
    picks = [(hero_p, HERO_REGION)]
    for p in sorted(others, key=lambda p: pnames[p]):
        regs = np.flatnonzero(c["patient_of_region"] == p)
        picks.append((p, c["region_names"][regs[np.argmax(ptr[regs + 1] - ptr[regs])]]))
    if HERO_REGION not in [c["region_names"][r] for r in np.flatnonzero(c["patient_of_region"] == hero_p)]:
        raise ValueError(f"{HERO_REGION} is not a region of patient {HERO_PATIENT}")

    types = sorted(c["type_names"])
    code = {n: i for i, n in enumerate(types)}
    cores, raw = [], []
    with zipfile.ZipFile(Path(DATA_DIR) / "charville_raw_data.zip") as z:
        for p, r in picks:
            m = region_cells(z, r)
            hero = r == HERO_REGION
            keep = np.arange(len(m)) if hero or len(m) <= MAX_CELLS else np.sort(rng.choice(len(m), MAX_CELLS, replace=False))
            n_all, m = len(m), m.iloc[keep]
            x, y = m.X.to_numpy(float) * PIXEL_UM, m.Y.to_numpy(float) * PIXEL_UM
            x, y = x - x.min(), y - y.min()
            raw.append(m[list(MARKERS)].to_numpy(float))
            cores.append({"region": r, "patient": f"CHV-{pnames[p]}", "hero": hero, "cells_in_region": n_all,
                          "x": np.round(x, 1).tolist(), "y": np.round(y, 1).tolist(),
                          "t": [code[n] for n in m.CELL_TYPE]})
    allv = np.concatenate(raw)
    lo, hi = np.percentile(allv, 1, axis=0), np.percentile(allv, 99, axis=0)
    for core, v in zip(cores, raw):
        q = np.clip(np.round((v - lo) / (hi - lo) * 255), 0, 255).astype(int)
        core["m"] = {k: q[:, i].tolist() for i, k in enumerate(MARKERS)}

    doc = {
        "schema": "preflight-tma/1", "source": "MEASURED",
        "meta": {"source": "Charville CODEX, Wu et al. 2022, Zenodo 13179600, CC-BY-4.0",
                 "label": "MEASURED cells", "script": "scripts/export_tma.py",
                 "selection": f"{N_CORES} cores, one per patient: hero {HERO_REGION} (CHV-{HERO_PATIENT}) plus "
                              f"{N_CORES - 1} patients drawn uniformly (seed {SEED}), each patient's largest region",
                 "downsampling": f"grid cores: uniform random sample of at most {MAX_CELLS} cells without replacement, "
                                 f"seed {SEED}; hero core: every cell",
                 "units": "x, y in microns from the region's min-x, min-y corner, 0.1 um; t indexes types",
                 "microns_per_pixel": PIXEL_UM, "seed": SEED,
                 "markers": {k: {"role": role, "lo": round(float(a), 4), "hi": round(float(b), 4)}
                             for (k, role), a, b in zip(MARKERS.items(), lo, hi)},
                 "marker_scale": "q = clip(round((v - lo) / (hi - lo) * 255), 0, 255); v is the source's normalised "
                                 "intensity, lo/hi the 1st/99th percentile over the exported cells"},
        "types": [{"name": n, "group": display_class(n)} for n in types],
        "hero": HERO_REGION,
        "cores": cores}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")))
    n = sum(len(k["t"]) for k in cores)
    print(f"{out}: {len(cores)} cores, {n} cells, {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo" / "data" / "tma.json")
