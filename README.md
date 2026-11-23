![Biopsy Scan](docs/assets/banner.png)

# Biopsy Scan

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![smoke](https://github.com/atiladeokegab/biopsy-scan/actions/workflows/smoke.yml/badge.svg)](https://github.com/atiladeokegab/biopsy-scan/actions/workflows/smoke.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
[![Docs](https://img.shields.io/badge/docs-read%20the%20docs-blue?logo=readthedocs&logoColor=white)](https://atiladeokegab.github.io/biopsy-scan/)

Studies that look for tumour cell patterns linked to patient survival are usually sized with a
textbook formula that assumes perfect measurement. Biopsy Scan instead simulates the planned study on
real tumour tissue, cell by cell, and runs hundreds of those simulations in seconds by parallelising
the work across CPUs and GPUs. On colorectal tissue the formula promised 80% power where the
simulation gave 27%; on head-and-neck tissue the two nearly agreed, so the answer has to come from
real tissue. It is a planning tool for researchers; it has not itself discovered or validated a
biomarker.

![The Biopsy Scan race: off-the-shelf tools vs Biopsy Scan on the same biopsy](docs/assets/race.gif)

<sub>The race replays two separately measured rates: off-the-shelf squidpy + lifelines on 8 CPU workers vs
Biopsy Scan's fused kernel on one H100, so the GIF's 10,800× mixes hardware; the same-hardware figures are in the
table below. Details in [RESULTS.md](RESULTS.md#demo-race-replay).</sub>

## Why it matters

Textbook power formulas assume every cell pattern is measured exactly on the whole tissue. In one
modelled scenario on the Charville colorectal cohort, they are far too optimistic:

| 200 simulated patients, HR 2.0 per SD, 630 candidate patterns, Bonferroni at α = 0.05 | Study power |
|---|---|
| Schoenfeld formula (**ANALYTIC**) | 80.1% |
| Biopsy Scan, simulated on real tissue, 300 studies (**ESTIMATE**) | **27.0%** (95% CI 22.1–32.4%) |

The gap depends on the design (the formula overstated power at 6 of 7 patient counts tested).

Simulating thousands of studies per design needs speed. Each pipeline below times one simulated study on
8 CPU workers; what each timing includes differs (see [RESULTS.md](RESULTS.md)):

| Pipeline | Seconds per study | Biopsy Scan's fused kernel is |
|---|---|---|
| **Off the shelf** (what a lab would run today): squidpy + lifelines | 22.35 s | **534× faster** per evaluated study (same host with 8 CPU workers per side; **MEASURED**; different statistics; the timing windows differ; 312× on the same 48 studies) |
| **Lab method, our optimised rewrite:** neighbour recount and a permutation test per study | 0.881 s | **21× faster** per evaluated study (8 workers each; **MEASURED**, cross-run) |
| **Biopsy Scan**, fused kernel | 0.0426 s (same host), 0.0419 s (cross-run) | |

<sub>Medians. 534× is the median of the per-container ratios, so it differs slightly from 22.35 / 0.0426. Biopsy Scan's time covers re-imaging and scoring; its calibrated threshold and calls are a separate step, not timed.</sub>

**Controlled comparisons.** The table above sets different statistics or separate runs side by side. These rows match worker count or timing scope, as indicated; only the H100 kernel pair runs the same workload:

| Comparison | Speed-up |
|---|---|
| **Lab method vs Biopsy Scan's Numba kernel** (not the fused kernel), each on 8 workers, through scoring with calibration charged; different procedures | **5.5× faster** (median of 3 fresh repeats, range 4.6–5.6×); 3.7× with setup; decisions agree on 362 of 400 (timings **MEASURED**, outcomes **SYNTHETIC**) |
| **Fused kernel on one H100 vs the same fused kernel on 8 CPU workers**, one container, same timing scope (re-imaging and batched scoring) | **19.5× faster** (0.00207 s vs 0.0404 s per evaluated study); features bit-equal on 3,000/3,000 studies (**MEASURED**) |

Speed doesn't change the features: every re-imaging kernel is tested against a float64 reference
implementation (bit-equal features, or within a stated tolerance). When squidpy's neighbour graph and
lifelines' score test are given Biopsy Scan's feature definition, the top pattern agrees in 16 of 16
simulated studies; squidpy + lifelines as documented compute different statistics and agree in 14 of 50
([details](docs/external-agreement.md)). One calibrated threshold instead of a permutation test per study
does cost some power at the race's design point (15.3% vs 20.3%).

Every number, with its evidence label, source file and caveats: **[RESULTS.md](RESULTS.md)**.

| Start from real tissue | Race the off-the-shelf tools | Take it to routine IHC |
|---|---|---|
| ![The Biopsy Scan workflow](docs/assets/screen-design.png) | ![The speed race](docs/assets/screen-race.png) | ![Routine IHC](docs/assets/screen-ihc.png) |

## Installation

You need `git`, `make`, [`uv`](https://docs.astral.sh/uv/), `python3` (3.9+) and Node.js. uv
installs the pinned Python 3.11.

```sh
git clone https://github.com/atiladeokegab/biopsy-scan.git
cd biopsy-scan
make smoke   # install, offline tests and demo checks (about 7 minutes, no data or GPU)
make demo    # serve the demo at http://localhost:8000
make data    # optional: download the public CODEX cohorts (about 730 MB)
```

## Documentation

Guides, architecture (C4 diagrams), methods, pre-registration and results:
**[atiladeokegab.github.io/biopsy-scan](https://atiladeokegab.github.io/biopsy-scan/)**.

## Contact

- Questions: [Discussions](https://github.com/atiladeokegab/biopsy-scan/discussions)
- Bugs and feature requests: [issues](https://github.com/atiladeokegab/biopsy-scan/issues)
- Security: report privately, see [SECURITY.md](SECURITY.md)
- Authors: Atilade Oke ([GitHub](https://github.com/atiladeokegab), [LinkedIn](https://www.linkedin.com/in/atiladeoke/)) and Faisal Lawan ([GitHub](https://github.com/FHL-08), [LinkedIn](https://www.linkedin.com/in/faisal-lawan/))

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

```bibtex
@software{biopsyscan_2026,
  author  = {Oke, Atilade and Lawan, Faisal},
  title   = {Biopsy Scan: Cell-by-Cell Spatial-Biomarker Study Simulation and Clinical IHC Translation},
  year    = {2026},
  url     = {https://github.com/atiladeokegab/biopsy-scan},
  license = {MIT}
}
```

If you use the cohorts or the biomarker definitions, please also cite Wu et al., *Nature Biomedical
Engineering* (2022) and Punovuori et al., *Cell* 187:7267 (2024). The biology of the pre-registered
rule comes from Punovuori et al.

## License

The code is [MIT](LICENSE) © 2026 Atilade Oke and Faisal Lawan. The data keep their own licences: the cohorts come from
Wu et al., *Nature Biomedical Engineering* (2022), Zenodo
[13179600](https://zenodo.org/records/13179600), CC BY 4.0, and some derived cell-level extracts
are committed under the same licence. The IHC photos are from the
[Human Protein Atlas](https://v25.proteinatlas.org/) v25, CC BY 4.0 (Uhlén et al., *Science* 2015),
cropped and resized; sources in [`hpa_manifest.json`](demo/workflow/img/hpa_manifest.json). Details:
[docs/data.md](docs/data.md).
