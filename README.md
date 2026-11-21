![Preflight](docs/assets/banner.png)

# Preflight

**Preflight tells a lab, before it images a single slide, whether its study can find the pattern that decides a patient's treatment.**

It re-images thousands of simulated studies cell by cell from real cancer tissue (CODEX), and it is being rebuilt so the whole search runs on GPUs.

![The Preflight demo page, with real Charville tissue cores side by side in 3D](docs/assets/demo-1280.png)

*The demo at 1280 px: 12 real Charville colorectal cores (24,033 cells, subsampled for display). The power chart on the right is still SYNTHETIC, as its banner says.*

Every number in this README carries one label: **MEASURED**, **ESTIMATE**, **SYNTHETIC**, **UNMEASURED** or **ANALYTIC**.

## The problem

About half of patients with locally advanced head and neck cancer relapse ([PMC7902285](https://pmc.ncbi.nlm.nih.gov/articles/PMC7902285/)). After surgery, the choice of radiotherapy, chemotherapy or watching still rests mostly on stage and margins. The Helsinki study behind MultivisionDx (Punovuori et al., *Cell* 2024; [code](https://github.com/WickstromLab/Punovuori-et-al.-Cell-2024)) showed the tissue knows more. Partial-EMT tumour cells surrounded by immune-rich stroma meant about 90% five-year survival. The same cells surrounded by fibroblasts meant about 21%. Stage cannot see that. The arrangement of cells can.

Finding a pattern like that takes enough patients. Real cohorts are small:

- **Charville** (colorectal, 159 patients, 38 events): the best of 630 spatial features has p = 0.13 after maxT correction. MEASURED, exploratory: the outcomes had already been seen. Nothing survives correction.
- **The textbook formula** says 200 patients give 80% power (ANALYTIC, Schoenfeld). Simulating those studies on real tissue, with sampling and mis-calls, gives about 27–32% (ESTIMATE: 32% from 60 studies per point, merged in #44; 27% [22%, 32%] from 300, merged in #53).

A lab that trusts the formula images 200 patients and most likely finds nothing. Preflight shows that before the first slide.

## The plan

The plan has two halves. **Decode** finds the biomarker in multiplex fluorescence (22 markers on UPMC). **Translate** turns it into one or two stains a hospital lab can run. The full plan is the founder's "Rebuilding the MultivisionDx Pipeline on GPUs". The parts buildable by code freeze are spec change C-003, approved and applied: the PRD is now v4, and R-13 to R-18 are in build (issues #66 to #71).

The Helsinki team examined 18 compound signatures. Preflight searches 3,780 patterns: 630 neighbourhood patterns in grammar v1, plus protein, size and stroma channels in v2.

One screen of 3,780 patterns fits on a laptop. The GPU load is the simulation around it: thousands of simulated studies, each re-imaged cell by cell from real tissue, and each running its own screen with 10,000 shuffles. The screen is the per-study kernel, batched across studies. The fleet spreads the studies across GPUs.

We build it twice and compare:

| | Implementation A: the CPU baseline | Implementation B: the GPU build |
|---|---|---|
| Re-imaging each study | What a lab runs today (rung 0): per study, a fresh cKDTree recount and 1,000 permutations on one core | Numba and CUDA kernels that give the same output cell for cell |
| Screening each study | One statsmodels PHReg Cox fit per pattern, on all CPU cores | Fit once, then score all 3,780 patterns as one matrix product |
| Honest correction | The same shuffles, run on all CPU cores | 10,000 shuffles as one matrix product, maximum taken on the fly |
| Thousands of studies | One after another | Batched on each GPU, and sharded across GPUs by the fleet |

Speed-ups count only at equal output. Re-imaging ratios are against rung 0. A GPU screen ratio is against the same statistic and the same shuffles on all CPU cores. The full-fit comparison with Implementation A is reported as a workflow comparison, not a speed-up.

**Decode, in plain words:**

1. **R1. Every radius for the price of one.** Build each cell's neighbour list once at the largest radius, sorted by distance. Every smaller radius is a running total.
2. **R2. Every candidate is a column.** A candidate is a cell state, a neighbourhood and a threshold. Each becomes one column of a patients-by-candidates matrix: 3,780 columns today. So choosing a cut-off is covered by the same correction.
3. **R3. Fit once, score everything.** Fit the clinical-only survival model once. Each candidate's score is then one matrix-vector product.
4. **R4. 10,000 shuffles as one matrix product.** Westfall–Young maxT asks how often the best shuffled candidate beats the real one. All shuffles become one product, streamed in blocks.
5. **R5. The smallest panel.** Try every one- and two-marker panel (253 with UPMC's 22 markers) and score each on held-out folds.
6. **R6. Learned features as more columns.** Foundation-model embeddings and graph features join the same screen.

**Translate, in plain words:**

1. **Distil the panel.** Find the fewest markers that keep the biomarker's predictive power, scored as held-out C-index retained.
2. **Simulate each stain format.** Recompute the biomarker as single-slide IHC, as two serial sections misaligned by 0–100 µm, and on 100–200 µm tiles. This tells a lab whether two routine slides are enough.

**Validation.** Every real-data result here is exploratory: the outcomes had already been seen. UPMC cannot carry a survival claim. Strict case identity gives 50 patients and 10 events (16 and 4 locked); grouping by clinical record gives 80 patients and 27 deaths (23 and 8 locked). These counts are MEASURED. UPMC still hosts the Helsinki test, because pEMT is a head and neck signature and Charville is colorectal. Its patients are split before any search. The locked patients are opened once, after the search is frozen, to score the Helsinki pattern (pEMT tumour bordered by fibroblasts), and that result first reports Preflight's own power for so few patients. HANCOCK, a cohort with real brightfield IHC, is the next validation step; it is not claimed by freeze.

## Status

Each row has one status. **BUILT**: merged on `integration`. **IN REVIEW**: open PR. **IN PROGRESS**: being built on a branch, no PR yet. **PLANNED**: in the v4 PRD (C-003), not built yet. **NOT CLAIMED**: not in this build.

| Item | Status | Where |
|---|---|---|
| Cohort loaders, patient identity, frozen feature grammar, NumPy reference re-imager | BUILT | #19 |
| R1: one neighbour list at the largest radius; 25, 50 and 100 µm read as running totals | BUILT | #19, #23 |
| Re-imaging kernel on Numba CPU and CUDA, cell for cell equal to the reference (GPU checked in the CUDA simulator; no hardware run yet) | BUILT | #23 |
| Studies sharded across GPUs (`--devices N`), bit-identical to one device | BUILT | #34 |
| R3 and R4 on CPU: score statistic U = X'r in float64 (no clinical covariates), maxT and omnibus over shuffles | BUILT | #19 |
| Real-data audit: Charville best feature p = 0.13 after maxT; UPMC descriptive | BUILT | #17, #29, #35 |
| R2 widened: protein, size and stroma channels, 3,780 features tested as one family | BUILT | #47 |
| Charville v2 result: strongest feature is PD1 on macrophages within 100 µm of "Tumor 5" cells (one tumour cell type), z = 3.96, maxT-adjusted p = 0.098 (MEASURED). Not significant after correction: exploratory | BUILT | #47 |
| Rung 0: what a lab runs today, timed on real Charville | BUILT | #43 |
| Textbook formula versus simulated power, 60 studies per point | BUILT | #44 |
| The same comparison at 300 studies per point (ESTIMATE) | BUILT | #53 |
| Design atlas: calibrated study simulations and streamed runner (draft) | IN REVIEW | #45 |
| Demo race plays only on "Run the studies" | BUILT | #48 |
| Pitch scenes: two real UPMC patients and the wall of 3,780 features | BUILT | #55 |
| Submission check: 5-minute pitch, live demo, video as backup | BUILT | #54 |
| C-003 applied: PRD v4 with R-13 to R-18 | BUILT | #60 |
| UPMC patient-level loader and locked held-out split (R-13, issue #66) | IN REVIEW | #62 |
| Helsinki candidate defined before the search (R-16 pre-registration) | BUILT | #61 |
| Implementation A analysis baseline: per-candidate PHReg on all cores (R-14, issue #67). All 3,780 Charville v2 patterns: 67.1 s on 8 cores, 222.8 s on 1 core (MEASURED, on a loaded shared VM; `bench/results/impl-a-*.json`) | IN REVIEW | #64 |
| R2–R4 on GPU as the per-study kernel: clinical-adjusted score screen, 10,000-shuffle maxT as one product, a valid adjusted null (R-15) | PLANNED | #68 |
| Rediscovery of the Helsinki pattern on the locked UPMC patients, with Preflight's power for that set first; exploratory, no survival claim (R-16) | PLANNED | #69 |
| R5 and Translate 1–2: minimal panels and IHC format simulation on UPMC (R-17) | PLANNED | #70 |
| The same screen rerun on Charville (R-18, issue #71) | IN PROGRESS | branch `r18-charville-screen` |
| R6: learned features and knockoffs | NOT CLAIMED | |
| Translate 3 (brightfield scoring engine) and 4 (markers from H&E) | NOT CLAIMED | |
| HANCOCK validation on real brightfield IHC | NOT CLAIMED | |
| The image half: registration, segmentation, quantification; GPU drop-ins for clustering | NOT CLAIMED | |

## Speed

| What | Time | Label |
|---|---|---|
| Implementation A, rung 0: one study of 159 Charville patients, cKDTree recount plus 1,000 permutations, one core of an AMD EPYC 7B13 | 7.52 s per study (mean of 3 studies) | MEASURED |
| Implementation A, analysis half: one full PHReg Cox fit for each of the 3,780 Charville v2 patterns (in review, #64; `bench/results/impl-a-*.json`; a workflow comparison, not an equal-work baseline) | 67.1 s on 8 cores; 222.8 s on 1 core (loaded shared VM) | MEASURED |
| Implementation B, re-imaging on a GPU | [__] | UNMEASURED |
| Implementation B, fleet at 1, 2, 4 and 8 GPUs | [__] | UNMEASURED |
| Implementation B, GPU screen versus the same work on all CPU cores | [__] | UNMEASURED |

GPU rows stay UNMEASURED until the GPU credits land (issue #2). A CUDA simulator run is never reported as a GPU time. Source: `bench/results/cpu-ladder.json`.

## Run it

Python 3.11 with [uv](https://docs.astral.sh/uv/), and Node for the demo's race test.

```sh
gh repo clone atiladeokegab/sota-parallelism-nov21
cd sota-parallelism-nov21
make smoke     # uv sync --dev, the unit tests and the demo race test; offline, no real data, no GPU
make demo      # checks the demo's data files, then serves it at http://localhost:8000
```

Use `make demo PORT=8001` if port 8000 is taken. The demo is static HTML and JS with three.js vendored, so it works with Wi-Fi off once served. Browsers block `file://`, so serve the page rather than opening the file. The public copy is at https://atiladeokegab.github.io/sota-parallelism-nov21/, published from `integration` by `.github/workflows/pages.yml`.

Real data, needed only for cohort runs:

```sh
make data      # fetches the Zenodo files into data/ (gitignored)
make summary   # per-cohort patients, regions, events and the frozen feature grammar
make tissue    # rebuilds demo/data/tissue.json from real Charville cells
```

What the demo shows:

| On the page | Label | Source |
|---|---|---|
| Tissue in the 3D view | MEASURED: real cells | Charville CODEX, 12 patients, subsampled (`scripts/export_tissue.py`) |
| Power grid and chart | SYNTHETIC until the atlas lands | `demo/data/grid.json` is a placeholder; the atlas (#45) replaces it |
| GPU speed-up | UNMEASURED | No timing is shown until `bench.json` holds a run on a real GPU |

The demo labels every data file SYNTHETIC unless the file itself says `"source": "MEASURED"`.

## Where the code lives

- `core/`: cohort loaders, patient identity, the frozen feature grammar and the NumPy reference re-imager. Interfaces: [docs/CONTRACT.md](docs/CONTRACT.md).
- `reimage/`: the Numba CPU and CUDA kernels, and the fleet.
- `stats/`: float64 stratified score statistics, maxT, omnibus and the null threshold.
- `audit/`: the real-data audit; `results/` and `audit/results/` hold its outputs.
- `bench/`: rung 0, the timing ladder and the textbook-versus-simulation comparison.
- `demo/`: the demo page.

The requirements are in [docs/PRD.md](docs/PRD.md).

## Data credits

The CODEX cohorts come from Zenodo record [13179600](https://zenodo.org/records/13179600), licensed [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Please cite Wu et al., *Nature Biomedical Engineering*, 2022.

- **UPMC**: head and neck cancer.
- **Charville**: the Stanford colorectal cancer cohort. The demo's tissue comes from it.

The data is not committed to this repository. `make data` downloads it.

## Team

Team Dry Lab, SoTA Scientific Parallelism Hackathon 2026.

- [Atilade](https://github.com/atiladeokegab)
- [Faisal](https://github.com/FHL-08)
