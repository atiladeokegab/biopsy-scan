![Preflight](docs/assets/banner.png)

# Preflight

**Preflight simulates thousands of cell-by-cell spatial-biomarker studies on real CODEX tissue before you image a slide, screens every candidate pattern with honest multiple-testing correction, and tests how much signal survives one- or two-stain clinical IHC.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
Live interactive demo: <https://atiladeokegab.github.io/sota-parallelism-nov21/>

![Preflight 3D tissue viewer and power gap demo](docs/assets/demo-1280.png)

Every number below carries its evidence label — **MEASURED**, **ESTIMATE**, **ANALYTIC**, **SYNTHETIC**, or **UNMEASURED** — and links its committed source file. Real-cohort results are exploratory (*outcomes already seen; no clinical survival claims*).

## Committed Results (on `integration`)

| Finding | Result & Label | Source |
|---|---|---|
| **Textbook vs simulated study power** (`n = 200`, `HR = 2.0/SD`, `M = 630`, `alpha = 0.05`) | Schoenfeld formula predicts `80.1%` (**ANALYTIC**); cell-by-cell real-tissue simulation gives `27.0%`, 95% CI `[22.1%, 32.4%]` over `300` studies (**ESTIMATE**) | [`bench/results/vs-schoenfeld-300.json`](bench/results/vs-schoenfeld-300.json) |
| **Charville v1 screen** (`159` patients, `38` events, `630` patterns) | Top `Tumor 3\|Blood vessel\|100um\|enrichment`: `z = 3.46`, `maxT p = 0.1275`, `omnibus p = 0.1365` (**MEASURED**, exploratory) | [`audit/results/charville.json`](audit/results/charville.json) |
| **Charville v2 screen** (`159` patients, `38` events, `3,780` patterns, `10` channels) | Top `Tumor 5\|Macrophage\|100um\|PD1`: `z = 3.96`, `maxT p = 0.0981`, `omnibus p = 0.1688`, NumPy ref `2,149.61 s` (**MEASURED**, exploratory) | [`results/charville-v2.json`](results/charville-v2.json) |
| **UPMC cohort counts** (verified case identity, descriptive only) | `50` patients, `10` events (`188` regions, `612` features); provisional tuple sensitivity `79–81` patients, `27` events (**MEASURED**, descriptive only) | [`audit/results/upmc.json`](audit/results/upmc.json) |
| **UPMC R-16 / R-17 locked-split evaluation** | R-16 rule pre-registered ([`audit/rediscovery.py`](audit/rediscovery.py)); live discovery/locked scores and Translate run pending (**UNMEASURED**) | [`results/upmc-translate.json`](results/upmc-translate.json) |
| **Lab today baseline** (`rung 0`, 1-core `cKDTree` + `1,000` permutations) | `7.52 s/study` mean of `3` Charville studies (**MEASURED**); `7,521 s` extrapolated for `1,000` studies (**ESTIMATE**) | [`bench/results/cpu-ladder.json`](bench/results/cpu-ladder.json) |
| **CPU re-imaging ladder** (`100` studies, `159` patients, `619,645` cells, load `10–26`) | NumPy ref `2.98 s/study` vs active-wait 8-thread Numba `3.26 s/study` (**MEASURED**); with `OMP_WAIT_POLICY=PASSIVE` at load `50–60`, 8-thread Numba `2.48 s` vs NumPy `8.72 s` per study (**MEASURED**) | [`bench/results/cpu-ladder.json`](bench/results/cpu-ladder.json), [`bench/results/numba-negative.json`](bench/results/numba-negative.json) |
| **Implementation A workflow comparator** (`3,780` `statsmodels` `PHReg` fits, Charville v2) | `63.84 s` on `8` workers (`AMD EPYC 7B13`, `4.67x` vs 1 worker; workflow comparator only, not matched vs R-15 score screen) (**MEASURED**) | [`bench/results/impl-a-charville-amd-epyc-7b13.json`](bench/results/impl-a-charville-amd-epyc-7b13.json) |
| **GPU re-imaging, multi-GPU fleet & GPU score screen** | CUDA kernels pass offline simulator parity; real GPU hardware timings not yet committed (**UNMEASURED**) | [`bench/results/cpu-ladder.json`](bench/results/cpu-ladder.json), [`results/charville-v2.json`](results/charville-v2.json) |
| **Interactive demo data** (`demo/`) | `12` Charville CODEX patients subsampled (**MEASURED**, [`demo/data/tissue.json`](demo/data/tissue.json)); power grid placeholder (**SYNTHETIC**, [`demo/data/grid.json`](demo/data/grid.json)) | [`demo/`](demo/) |

## Reproduce in 3 Commands

Requires Python 3.11 ([`uv`](https://docs.astral.sh/uv/)) and Node.js (for the offline demo regression test):

```sh
git clone https://github.com/atiladeokegab/sota-parallelism-nov21.git && cd sota-parallelism-nov21
make smoke   # uv sync --dev, offline pytest suite, and node demo/race.test.mjs (no raw data or GPU needed)
make demo    # validates demo/data and serves the interactive demo at http://localhost:8000 (PORT=8001 if busy)
```

Real cohort runs additionally fetch the public CODEX datasets into gitignored `data/` via `make data` (Zenodo [`13179600`](https://zenodo.org/records/13179600), [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)).

## Repository Layout

- [`core/`](core/): Cohort loaders, UPMC discovery/locked split enforcement, frozen v1/v2 spatial grammar, and NumPy reference re-imager ([`docs/CONTRACT.md`](docs/CONTRACT.md)).
- [`reimage/`](reimage/): Numba CPU and CUDA cell-re-imaging kernels plus multi-GPU shard coordinator.
- [`stats/`](stats/): Stratified Cox score statistics, Westfall–Young `maxT`, and omnibus permutation/bootstrap null calibration.
- [`audit/`](audit/): Real-cohort exploratory audits (`audit/results/`) and pre-registered R-16 biomarker definition.
- [`translate/`](translate/): Nested-CV 1-/2-marker panel search (`A-17.1`) and single-slide / serial-section (`0–100 um`) / tile (`100–200 um`) IHC simulation (`A-17.2`).
- [`bench/`](bench/): Rung-0 baseline, `PHReg` workflow comparator (`impl_a`), Schoenfeld vs simulated power, CPU/OpenMP diagnostics, and committed results (`bench/results/`).
- [`demo/`](demo/): Static three.js showcase (`make deploy` / GitHub Pages). Specifications live in [`docs/PRD.md`](docs/PRD.md).

## MIT Licence & How to Cite

- **Licence:** Code and documentation in this repository are released under the **[MIT License](LICENSE)** (`Copyright (c) 2026 Team Dry Lab`, Team: [Atilade](https://github.com/atiladeokegab) & [Faisal](https://github.com/FHL-08)). Cohort data is sourced from Zenodo record [`13179600`](https://zenodo.org/records/13179600) under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
- **How to cite:**

```bibtex
@software{preflight_2026,
  author       = {{Team Dry Lab}},
  title        = {Preflight: Cell-by-Cell Spatial-Biomarker Study Simulation and Clinical IHC Translation},
  year         = {2026},
  url          = {https://github.com/atiladeokegab/sota-parallelism-nov21},
  license      = {MIT}
}
```

If using the underlying cohorts or biomarker definitions, please also cite Wu et al., *Nature Biomedical Engineering* (2022; Zenodo [`13179600`](https://zenodo.org/records/13179600)) and Punovuori et al., *Cell* 187:7267 (2024).

## Next: Contributing Back

**Next — contributing back *(planned, not done)*:** upstream pull requests to [`squidpy`](https://squidpy.readthedocs.io/) (batched/CSR radius-prefix neighbourhood counts), [`lifelines`](https://lifelines.readthedocs.io/) (`pandas` 3 / Python 3.11+ compatibility and stratified score-test helpers), and [`scikit-survival`](https://scikit-survival.readthedocs.io/) (vectorized permutation/bootstrap concordance and Cox score utilities).
