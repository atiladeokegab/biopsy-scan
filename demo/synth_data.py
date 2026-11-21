"""Write SYNTHETIC placeholders demo/data/grid.json and demo/data/tissue.json so the page works
before the atlas (#5) and the core loaders export real data.

Neither file is real: the grid comes from a toy attenuation model, the tissue from random tumour
nests with an immune rim. Replace both; the schemas are in demo/README.md.
Run: python3 -I demo/synth_data.py
"""

import json
import math
import random
from pathlib import Path

OUT = Path(__file__).parent / "data"

# ---- grid.json: a toy power model -------------------------------------------------------------
PATIENTS = [20, 40, 80, 120, 160, 200, 240, 300]
REGIONS = [1, 2, 4, 8]
FOV = [250, 500, 1000, 2000, "full"]
MISCALL = [0.0, 0.1, 0.2, 0.3]
HR_PER_SD = 2.0
EVENT_FRACTION = 0.3  # deaths per patient
Z_CRIT = 3.6  # roughly a family-wise 5% threshold over a few hundred patterns


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def point(n, k, fov, eps):
    side = 2500 if fov == "full" else fov
    cells = k * 6.5 * (side / 1000) ** 2 * 1000  # about 6.5k cells per mm^2
    reliability = 1 / (1 + 3000 / cells)  # sampling noise in the observed feature
    signal = math.log(HR_PER_SD) * math.sqrt(reliability) * (1 - eps / 0.5)
    z = signal * math.sqrt(EVENT_FRACTION * n)
    power = phi(z - Z_CRIT)
    identification = power * (0.55 + 0.45 * reliability) * (1 - eps)
    return {"patients": n, "regions": k, "fov_um": fov, "miscall": eps,
            "power": round(power, 3), "identification_rate": round(identification, 3)}


grid = {
    "schema": "preflight-grid/1",
    "source": "SYNTHETIC",
    "note": "Toy attenuation model from demo/synth_data.py. Not simulated studies; replace with the atlas output.",
    "studies_per_point": 1000,
    "effect_hr_per_sd": HR_PER_SD,
    "cohort": {"name": "Charville", "patients": 159, "regions": 2, "regions_mean": 1.8, "fov_um": "full", "miscall": 0.0},
    "axes": {"patients": PATIENTS, "regions": REGIONS, "fov_um": FOV, "miscall": MISCALL},
    "points": [point(n, k, f, e) for n in PATIENTS for k in REGIONS for f in FOV for e in MISCALL],
}

# ---- tissue.json: tumour nests in stroma, immune cells on the nest rims ---------------------------
REGION_UM = 2500
CELLS = 2500
rng = random.Random(7)


def patient(i):
    nests = [(rng.uniform(400, 2100), rng.uniform(400, 2100), rng.uniform(220, 420)) for _ in range(rng.randint(3, 5))]
    rim = rng.uniform(0.3, 0.8)  # how strongly immune cells hug the nests: the spatial pattern
    cells = []
    while len(cells) < CELLS:
        x, y = rng.uniform(0, REGION_UM), rng.uniform(0, REGION_UM)
        d = min(math.hypot(x - nx, y - ny) / r for nx, ny, r in nests)  # 1.0 = on a nest edge
        if d < 1 and rng.random() < 0.92:
            t = "tumour"
        elif 0.9 < d < 1.35 and rng.random() < rim:
            t = "immune"
        elif rng.random() < 0.06:
            t = "immune"
        elif rng.random() < 0.3:
            t = "stroma"
        else:
            continue  # thin the stroma so the nests read
        cells.append({"x": round(x), "y": round(y), "type": t})
    return {"id": f"SYN-{i + 1:02d}", "cells": cells}


tissue = {
    "schema": "preflight-tissue/1",
    "source": "SYNTHETIC",
    "note": "Random tumour nests with an immune rim from demo/synth_data.py. Not real cells.",
    "region_um": REGION_UM,
    "patients": [patient(i) for i in range(12)],
}

OUT.mkdir(exist_ok=True)
for name, doc in (("grid.json", grid), ("tissue.json", tissue)):
    (OUT / name).write_text(json.dumps(doc, separators=(",", ":")) + "\n")
print(f"wrote {OUT}/grid.json ({len(grid['points'])} points) and tissue.json "
      f"({len(tissue['patients'])} patients x {CELLS} cells), both SYNTHETIC")
