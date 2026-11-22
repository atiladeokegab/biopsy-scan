"""Build demo/app/data.json, every number and name the Current vs Ours page shows, from the real result files.

Run from the repo root:
  python3 -I demo/app/build_data.py           write data.json
  python3 -I demo/app/build_data.py --check   exit 1 if the committed data.json differs from a fresh build
  DATA_DIR=... uv run python -I demo/app/build_data.py --cells
      rebuild demo/app/cells.json: the same 36 Charville regions as demo/data/tma.json, every cell kept
      (needs the raw Charville zip; see scripts/fetch_data.sh)
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "demo" / "app"
OUT, CELLS = HERE / "data.json", HERE / "cells.json"
COHORT = "results/charville-v2.json"
CURRENT = "bench/results/external-baseline-20261122T043823Z.json"
OURS = "bench/results/modal-design-20261121T205536Z.json"
GRID = "demo/data/grid.json"
HPA = "demo/workflow/img/hpa_manifest.json"
RACE_S, OURS_FINISH_S = 10.0, 0.7  # the founder's ruling: a 10 s race that Ours finishes about 0.7 s in
# Display pair: the stain search withheld a recommendation on this cohort, so this is the founder's example pair,
# each routine IHC gene with the multiplex panel marker it stands in for.
PAIR = [("KRT5", "PanCK"), ("CD8A", "CD8")]
# Display colour of each multiplex marker (fluorescence-style); a marker missing here is an error, not a default.
MARKER_RGB = {"DAPI": "#3f6dff", "PanCK": "#ff5f4f", "CD8": "#7dff4a", "CD68": "#ffc23c", "aSMA": "#1fa37a", "CD20": "#ea4fe0"}


def read(rel):
    return json.loads((ROOT / rel).read_text())


def build():
    cohort, cur, ours, grid = read(COHORT), read(CURRENT), read(OURS), read(GRID)
    cells = json.loads(CELLS.read_text())
    s = cur["summary"]
    fused = ours["arms"]["fused"]
    studies = grid["studies_per_point"]
    cur_s, ours_s = s["seconds_per_study"], fused["stages"]["wall_per_study_s"]
    scale = OURS_FINISH_S / (studies * ours_s)  # race seconds per real second
    photos = {p["gene"]: p for p in read(HPA)}
    for gene, marker in PAIR:
        assert gene in photos and marker in cells["markers"], (gene, marker)
    return {
        "schema": "preflight-race-ihc/1",
        "sources": [COHORT, CURRENT, OURS, GRID, HPA, "demo/app/cells.json"],
        "cohort": {"name": cohort["cohort"], "patients": cohort["n_patients"], "events": cohort["n_events"],
                   "regions": cohort["n_regions"], "cores_shown": len(cells["cores"])},
        "studies": studies,
        "race": {"seconds": RACE_S, "ours_finish_s": OURS_FINISH_S, "race_s_per_real_s": scale,
                 "current_studies_at_end": int(RACE_S / scale / cur_s)},
        "current": {"s_per_study": cur_s, "workers": s["workers"],
                    "worker_s_per_study": statistics.median(r["study_s"] for r in cur["records"]),
                    "total_s": studies * cur_s},
        "ours": {"s_per_study": ours_s, "workers": fused["workers"], "total_s": studies * ours_s},
        "ratio": cur_s / ours_s,
        "markers": cells["markers"],
        "marker_colors": [MARKER_RGB[m] for m in cells["markers"]],
        "nuclear": [cells["meta"]["marker_roles"][m] == "nuclear" for m in cells["markers"]],
        "kept": [m for _, m in PAIR],
        "ihc": [{"gene": g, "marker": m, "photo": "../workflow/img/" + photos[g]["file"], "page": photos[g]["page_url"],
                 "credit": photos[g]["attribution"] + ", " + photos[g]["licence"]} for g, m in PAIR],
        "cells_credit": cells["note"],
    }


def build_cells(out=CELLS):
    """Every cell of demo/data/tma.json's 36 regions: same picks and row format, no 1,200-cell cap."""
    sys.path.insert(0, str(ROOT / "demo"))
    import collect_tma
    collect_tma.MAX_CELLS = 10**9
    collect_tma.main(out)
    d = json.loads(out.read_text())
    d["note"] = "Charville CODEX, Wu et al. 2022, Zenodo 13179600, CC-BY-4.0; real cells, every cell of each region"
    d["meta"].update(script="demo/app/build_data.py --cells", downsampling="none: every cell of each region")
    out.write_text(json.dumps(d, separators=(",", ":")))


def main(argv):
    if "--cells" in argv:
        build_cells()
    data = build()
    if "--check" in argv:
        ok = OUT.exists() and json.loads(OUT.read_text()) == json.loads(json.dumps(data))
        print(f"demo/app/data.json: {'ok' if ok else 'STALE, run python3 -I demo/app/build_data.py'}")
        return 0 if ok else 1
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print("wrote demo/app/data.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
