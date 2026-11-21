![Preflight](docs/assets/banner.png)

# Preflight

**Thousands of studies, simulated in seconds, before you image a slide.**
Preflight re-images simulated spatial-biology studies cell by cell from real head and neck and colorectal cancer tissue (CODEX), and does it at scale on a GPU.
A lab sees which study designs (patients, regions per patient, field of view, mis-call rate) can actually detect a survival-linked spatial pattern, before it spends a slide.

![The demo page, "Can my study find it?", with real Charville tissue cores stacked in 3D](docs/img/demo-1280.png)

*The demo at 1280 px, showing 12 real Charville colorectal cores (24,033 cells, subsampled for display). The power figure on the right still comes from the synthetic grid, as its banner says.*

## Run it

Python 3.11 with [uv](https://docs.astral.sh/uv/).

```sh
gh repo clone atiladeokegab/sota-parallelism-nov21
cd sota-parallelism-nov21
make smoke                                  # uv sync --dev, then the unit tests; offline, no real data
python3 -m http.server 8000 -d demo         # the demo; open http://localhost:8000
```

The demo is static HTML and JS with three.js vendored, so it needs no network once served. Browsers block `file://`, so serve the page rather than double-clicking it. `python3 -I demo/check_data.py` validates the demo's data files.

Real data, which you only need for the cohort runs (not for `make smoke`):

```sh
make data      # fetches the Zenodo files into data/ (gitignored)
make summary   # per-cohort patients, regions, events and the frozen feature grammar
```

## What is measured

| What you see | Status | Where it comes from |
|---|---|---|
| Tissue in the 3D view | **MEASURED**: real cells | Charville CODEX cells, 12 patients, subsampled for display (`scripts/export_tissue.py`, #20; until #20 merges, the page shows a SYNTHETIC placeholder) |
| Power grid and chart | **SYNTHETIC** until #5 | `demo/data/grid.json` is a placeholder. The design atlas (#5) replaces it with simulated studies on the real cohorts |
| GPU speed-up | **UNMEASURED** until the event GPU | No timing is shown until `bench.json` holds a run on the event GPU (#3). The page says UNMEASURED and shows no ratio |

The demo labels every data file SYNTHETIC unless the file itself says `"source": "MEASURED"`.

## How it works

- `core/`: cohort loaders, patient identity, the frozen feature grammar, and the NumPy reference re-imager. The binding interfaces are in [docs/CONTRACT.md](docs/CONTRACT.md).
- `stats/`: float64 stratified Cox score statistics, maxT, omnibus and the null threshold for a design.
- `demo/`: the "Can my study find it?" page.

The architecture and the plan are in [IDEA.md](IDEA.md).

## Data credits

The CODEX cohorts come from Zenodo record [13179600](https://zenodo.org/records/13179600), licensed [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Please cite Wu et al., *Nature Biomedical Engineering*, 2022.

- **UPMC**: head and neck cancer.
- **Charville**: the Stanford colorectal cancer cohort, which is the patient-level cohort used here. The demo's tissue comes from it.

The data is not committed to this repository. `make data` downloads it.

## Team

Team Dry Lab, SoTA Scientific Parallelism Hackathon 2026.

- [Atilade](https://github.com/atiladeokegab)
- [Faisal](https://github.com/FHL-08)
