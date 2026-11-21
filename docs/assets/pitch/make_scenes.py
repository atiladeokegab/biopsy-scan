"""Pitch scenes from real data: two UPMC patients side by side, and the wall of 3,780 features.

Run from the repo root (needs google-chrome and the Zenodo 13179600 zips in DATA_DIR, default data/):
    uv run python -I docs/assets/pitch/make_scenes.py
Writes two-patients.png, wall.png and wall-expert.png next to this file (1920x1080 at 2x).
Captions use label fields only: HPV status, recurred, vital status and survival_day.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))
PIXEL_UM = 0.377  # same as core.cohorts.PIXEL_UM

# Two HPV-positive patients with verified case numbers: no recurrence and alive (NED), against
# recurred and died of disease (DOD). Chosen for visibly different tumour borders: the non-tumour
# cells within 30 um of tumour are 93% immune in the first and 37% immune in the second.
PATIENTS = ["UPMC_c004_v001_r001_reg009", "UPMC_c006_v001_r001_reg058"]
WINDOW_UM = (1000, 750)  # the same field, centred, for both regions

IMMUNE = {"CD4 T cell", "CD8 T cell", "B cell", "Macrophage", "Granulocyte", "APC", "Naive immune cell"}

# Tokens from demo/tokens.css.
CSS = """
@font-face { font-family: Inter; src: local("Inter"); }
:root { --bg:#0B1018; --canvas:#070B11; --surface:#121A25; --border:#22303F; --head:#F4F7FA; --body:#B9C6D4;
  --muted:#8898AA; --accent:#2BD4A4; --tumour:#FF7A59; --immune:#F5C451; --stroma:#56687A; }
* { box-sizing: border-box; margin: 0; }
body { width: 1920px; height: 1080px; background: var(--bg); color: var(--body); font-family: Inter, sans-serif;
  padding: 72px 96px; display: flex; flex-direction: column; gap: 28px; overflow: hidden; }
h1 { color: var(--head); font-size: 44px; font-weight: 700; letter-spacing: -0.5px; }
.sub { font-size: 22px; color: var(--muted); }
.foot { margin-top: auto; display: flex; justify-content: space-between; font-size: 16px; color: var(--muted);
  font-family: "JetBrains Mono", monospace; }
.key { display: flex; gap: 28px; font-size: 20px; align-items: center; }
.key i { display: inline-block; width: 14px; height: 14px; border-radius: 7px; margin-right: 8px; vertical-align: -1px; }
"""


def region_cells(z: zipfile.ZipFile, region: str) -> pd.DataFrame:
    xy = pd.read_csv(io.TextIOWrapper(z.open(f"raw_data/{region}.cell_data.csv")))
    ty = pd.read_csv(io.TextIOWrapper(z.open(f"raw_data/{region}.cell_types.csv")))
    d = xy.merge(ty, on="CELL_ID")
    d["x"], d["y"] = d.X * PIXEL_UM, d.Y * PIXEL_UM
    d["cls"] = np.where(d.CELL_TYPE.str.startswith("Tumor"), 0, np.where(d.CELL_TYPE.isin(IMMUNE), 2, 1))
    return d


def caption(r: pd.Series) -> tuple[str, str]:
    hpv = {"1": "HPV-positive", "0": "HPV-negative"}[r.hpvstatus]
    rec = {"1": "Recurred", "0": "No recurrence"}[r.recurred]
    vital = {"NED": "alive", "AWD": "alive", "DOD": "died of disease", "DOC": "died of other causes"}[r.status]
    return f"{rec}, {vital}", f"{hpv} · followed {int(r.survival_day):,} days"


def two_patients() -> str:
    lab = pd.read_csv(DATA / "upmc_labels.csv", dtype=str, keep_default_na=False).set_index("region_id")
    a, b = (lab.loc[p] for p in PATIENTS)
    assert a.hpvstatus == b.hpvstatus and a.recurred == "0" and a.status == "NED" and a.survival_status == "0"
    assert b.recurred == "1" and b.status == "DOD"
    W, H = WINDOW_UM
    panels = []
    with zipfile.ZipFile(DATA / "upmc_raw_data.zip") as z:
        for p in PATIENTS:
            d = region_cells(z, p)
            x0, y0 = (d.x.max() + d.x.min() - W) / 2, (d.y.max() + d.y.min() - H) / 2
            d = d[(d.x >= x0) & (d.x < x0 + W) & (d.y >= y0) & (d.y < y0 + H)]
            head, sub = caption(lab.loc[p])
            panels.append({"cells": np.c_[(d.x - x0).round(1), (d.y - y0).round(1), d.cls].tolist(),
                           "head": head, "sub": sub, "id": f"case {lab.loc[p].sample_label_visualizer.split(' ')[0]} · {p}"})
    return f"""<!doctype html><meta charset="utf-8"><style>{CSS}
.row {{ display: flex; gap: 64px; }}
.panel {{ display: flex; flex-direction: column; gap: 14px; }}
canvas {{ width: 832px; height: 624px; background: var(--canvas); border: 1px solid var(--border); border-radius: 10px; }}
.panel h2 {{ color: var(--head); font-size: 32px; font-weight: 600; }}
.panel p {{ font-size: 22px; }}
.panel .id {{ font-size: 15px; color: var(--muted); font-family: "JetBrains Mono", monospace; }}
</style>
<h1>Two real patients. Same HPV status.</h1>
<div class="key"><span><i style="background:var(--tumour)"></i>Tumour</span><span><i style="background:var(--stroma)"></i>Stroma</span>
<span><i style="background:var(--immune)"></i>Immune</span><span class="sub">One {W:,} × {H} µm field per patient, every cell, coloured by type</span></div>
<div class="row" id="row"></div>
<div class="foot"><span>UPMC head and neck cohort, CODEX (Wu et al., Zenodo 13179600, CC-BY-4.0) · captions are label fields only</span><span>MEASURED tissue</span></div>
<script>
const P = {json.dumps(panels)}, C = ["#FF7A59", "#56687A", "#F5C451"], S = 832 / {W};
for (const p of P) {{
  const el = document.createElement("div"); el.className = "panel";
  el.innerHTML = `<canvas width="1664" height="1248"></canvas><h2>${{p.head}}</h2><p>${{p.sub}}</p><p class="id">${{p.id}}</p>`;
  document.getElementById("row").append(el);
  const g = el.querySelector("canvas").getContext("2d"); g.scale(2, 2);
  for (const k of [1, 2, 0]) {{ g.fillStyle = C[k];  // stroma under immune under tumour
    for (const [x, y, c] of p.cells) if (c === k) {{ g.beginPath(); g.arc(x * S, y * S, 2.4, 0, 7); g.fill(); }} }}
}}
</script>"""


# The wall: Charville grammar v1 (630) and v2 (3,150), from core.grammar and results/charville-v2.json (PR #47).
FOCAL = ["Tumor 1", "Tumor 2 (Ki67 Proliferating)", "Tumor 3", "Tumor 4", "Tumor 5", "Tumor 6 / DC", "Tumor 7"]
TYPES = ["B cell", "Blood vessel", "CD4 T cell", "CD8 T cell", "Granulocyte", "Macrophage", "Other", "Stroma"] + FOCAL
RADII = [25, 50, 100]
FAMILIES = [("fraction", "Neighbour share"), ("enrichment", "Neighbour enrichment"), ("PD1", "PD-1"), ("PDL1", "PD-L1"),
            ("Ki67", "Ki-67"), ("GranzymeB", "Granzyme B"), ("HLA-DR", "HLA-DR"), ("aSMA", "αSMA"),
            ("CollagenIV", "Collagen IV"), ("Vimentin", "Vimentin"), ("Podoplanin", "Podoplanin"), ("SIZE", "Cell size")]
# shortcut: an illustrative hand-picked study (tumour neighbourhood shares at 50 um), not taken from a paper.
EXPERT = {f"Tumor 1|{j}|50um|fraction" for j in ["CD8 T cell", "CD4 T cell", "Macrophage", "B cell", "Granulocyte", "Stroma"]}


def wall(expert: bool) -> str:
    blocks = [{"label": label, "v": 1 if i < 2 else 2,
               "on": [expert and f"{t}|{j}|{r}um|{key}" in EXPERT for t in FOCAL for r in RADII for j in TYPES]}
              for i, (key, label) in enumerate(FAMILIES)]
    n1 = sum(len(b["on"]) for b in blocks if b["v"] == 1)
    n2 = sum(len(b["on"]) for b in blocks if b["v"] == 2)
    assert (n1, n2) == (630, 3150)
    n_on = sum(map(sum, (b["on"] for b in blocks)))
    title = (f"What one expert study tests: {n_on} of {n1 + n2:,}" if expert
             else f"{n1 + n2:,} patterns in every study")
    sub = ("A hand-picked handful: tumour cells' neighbours within 50 µm" if expert
           else "Every tumour type × every neighbour type × three distances, by neighbour mix, protein and size")
    return f"""<!doctype html><meta charset="utf-8"><style>{CSS}
.wall {{ display: grid; grid-template-columns: repeat(6, auto); gap: 24px 0; justify-content: space-between; }}
.b p {{ font-size: 17px; margin-bottom: 8px; color: var(--body); white-space: nowrap; }}
.b p span {{ color: var(--muted); }}
.g {{ display: grid; grid-template-columns: repeat(15, 13px); gap: 2px; }}
.g i {{ width: 13px; height: 13px; border-radius: 2px; background: #263546; }}
.v1 .g i {{ background: #3A4B5E; }}
.dim .g i {{ opacity: .35; }}
.g i.on {{ background: var(--accent); opacity: 1; box-shadow: 0 0 0 2px var(--accent); }}
</style>
<h1>{title}</h1><p class="sub">{sub}</p>
<div class="wall {'dim' if expert else ''}" id="w"></div>
<div class="foot"><span>Charville cohort · v1 neighbour mix {n1} · v2 protein and size {n2:,} (PR #47) · one tile = one feature</span><span>7 tumour types × 15 neighbour types × 3 radii per family</span></div>
<script>
for (const b of {json.dumps(blocks)}) {{
  const el = document.createElement("div"); el.className = "b v" + b.v;
  el.innerHTML = `<p>${{b.label}} <span>· ${{b.on.length}}</span></p><div class="g">${{b.on.map((o) => `<i${{o ? ' class="on"' : ""}}></i>`).join("")}}</div>`;
  document.getElementById("w").append(el);
}}
</script>"""


def shoot(html: str, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "scene.html"
        page.write_text(html)
        subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--hide-scrollbars", f"--user-data-dir={tmp}/p",
                        "--force-device-scale-factor=2", "--window-size=1920,1080", "--virtual-time-budget=5000",
                        f"--screenshot={OUT / name}", page.as_uri()], check=True, capture_output=True, timeout=120)
    print(OUT / name)


if __name__ == "__main__":
    which = sys.argv[1:] or ["patients", "wall"]
    if "wall" in which:
        shoot(wall(False), "wall.png")
        shoot(wall(True), "wall-expert.png")
    if "patients" in which:
        shoot(two_patients(), "two-patients.png")
