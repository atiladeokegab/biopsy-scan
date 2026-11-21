"""Write a SYNTHETIC placeholder demo/data/grid.json so the page works before the atlas (#5) runs.

The numbers come from a toy attenuation model, not from re-imaged studies. Replace the file with
the atlas output; the schema is in demo/README.md. Run: python3 -I demo/synth_grid.py
"""

import json
import math
from pathlib import Path

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
    "note": "Toy attenuation model from demo/synth_grid.py. Not re-imaged studies; replace with the atlas output.",
    "studies_per_point": 1000,
    "effect_hr_per_sd": HR_PER_SD,
    "cohort": {"name": "Charville", "patients": 162, "regions": 2, "fov_um": "full", "miscall": 0.0},
    "headline": {"status": "UNMEASURED", "studies": 1000, "gpu_seconds": None, "cpu_seconds": None},
    "axes": {"patients": PATIENTS, "regions": REGIONS, "fov_um": FOV, "miscall": MISCALL},
    "points": [point(n, k, f, e) for n in PATIENTS for k in REGIONS for f in FOV for e in MISCALL],
}

out = Path(__file__).parent / "data" / "grid.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(grid, indent=1) + "\n")
print(f"wrote {out} ({len(grid['points'])} points, SYNTHETIC)")
