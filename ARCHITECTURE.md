# Architecture

How Biopsy Scan fits together, from the outside in. The diagrams follow the
[C4 model](https://c4model.com/): system context, then containers, then the components of the
simulation engine. GitHub and the docs site render them from the Mermaid source below.

## 1. System context

Who uses Biopsy Scan, and what it depends on.

```mermaid
C4Context
  title Biopsy Scan: system context
  Person(researcher, "Study designer", "Plans a spatial-biomarker study: patients, regions, field of view, stains")
  System(preflight, "Biopsy Scan", "Simulates thousands of studies on real tissue, screens every pattern, and simulates the frozen, pre-registered R-16 biomarker on routine IHC")
  System_Ext(zenodo, "Zenodo 13179600", "Public CODEX cohorts (Charville, UPMC), CC BY 4.0")
  System_Ext(hpa, "Human Protein Atlas", "Routine IHC photos used by the demo (v25), CC BY 4.0")
  System_Ext(modal, "Modal", "Cloud CPUs and GPUs for the large runs")
  Rel(researcher, preflight, "Chooses a design, reads power and the routine-IHC simulation of the R-16 biomarker")
  Rel(preflight, zenodo, "make data downloads cell tables")
  Rel(preflight, modal, "Runs large sweeps and GPU kernels")
  Rel(preflight, hpa, "Shows example stains")
```

## 2. Containers

The runnable pieces and how data moves between them. Everything runs locally except the Modal jobs.

```mermaid
C4Container
  title Biopsy Scan: containers
  Person(researcher, "Study designer")
  System_Ext(zenodo, "Zenodo cohorts")
  System_Ext(modal, "Modal")
  Container_Boundary(repo, "This repository") {
    Container(engine, "Simulation engine", "Python 3.11, NumPy, Numba, CUDA", "core/, reimage/, stats/, atlas/: re-imaging, survival statistics, design calibration")
    Container(translate, "Clinical translation", "Python", "translate/: IHC panel search and slide, section and tile simulation for the frozen, pre-registered R-16 biomarker")
    Container(audit, "Real-cohort audits", "Python", "audit/: pre-registered, exploratory audits of the real cohorts")
    Container(bench, "Benchmarks", "Python", "bench/, gpu/: baselines, off-the-shelf comparison, Modal jobs")
    ContainerDb(results, "Committed results", "JSON, Parquet", "bench/results/, results/, audit/results/, results/claims.json")
    Container(exporters, "Exporters", "Python", "scripts/export_*.py, demo/collect_*.py: results to demo data")
    Container(demo, "Demo", "HTML, ES modules, three.js", "demo/: static page, served by make demo")
  }
  Rel(engine, zenodo, "Reads cell tables from data/")
  Rel(bench, modal, "Submits jobs")
  Rel(bench, engine, "Times and calls")
  Rel(translate, engine, "Uses the frozen grammar and statistics")
  Rel(translate, audit, "Takes the pre-registered R-16 biomarker definition")
  Rel(audit, engine, "Uses loaders and statistics")
  Rel(bench, results, "Writes")
  Rel(audit, results, "Writes")
  Rel(translate, results, "Writes")
  Rel(exporters, results, "Reads")
  Rel(exporters, demo, "Writes demo/data/*.json")
  Rel(researcher, demo, "Opens in a browser")
```

## 3. Components of the simulation engine

One simulated study: load a cohort, re-image every patient under the planned design, score every
candidate pattern, and decide with a calibrated multiple-testing threshold.

```mermaid
C4Component
  title Simulation engine: components
  Container_Boundary(engine, "Simulation engine") {
    Component(cohorts, "Cohort loaders", "core/cohorts.py, core/heldout.py", "Patients, strata, survival endpoints, cells; the locked held-out split")
    Component(grammar, "Pattern grammar", "core/grammar.py", "The frozen candidate patterns and their hash")
    Component(truth, "Truth layer", "core/truth.py", "Neighbour lists and true counts, computed once")
    Component(ref, "Reference re-imager", "core/reimage_ref.py", "Float64 NumPy oracle every kernel must match")
    Component(kernels, "Re-imaging kernels", "reimage/cpu.py, fused.py, gpu.py, gpu_fused.py, fleet.py", "Field-of-view crops and mis-calls, then counts, on CPU or GPU")
    Component(stats, "Statistics", "stats/reference.py, stats/screen.py", "Stratified Cox score tests, maxT permutations, clinical-null screen")
    Component(atlas, "Design calibration", "atlas/", "Outcome simulation, power per design point, held-out validation")
  }
  Rel(cohorts, truth, "Cells")
  Rel(grammar, kernels, "Which patterns to count")
  Rel(truth, kernels, "Adjacency")
  Rel(ref, kernels, "Checked against, cell for cell")
  Rel(kernels, stats, "Feature matrix X")
  Rel(stats, atlas, "Test statistics, calls")
```

## Where to start reading

| If you want to… | Read |
|---|---|
| See what's frozen and why | [docs/CONTRACT.md](docs/CONTRACT.md), [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) |
| Follow one re-imaged region | `core/reimage_ref.py`, then `reimage/fused.py` |
| Follow the multiple-testing screen | `stats/screen.py` |
| Check a published number | [RESULTS.md](RESULTS.md) and `results/claims.json` (`python3 -I scripts/check_claims.py`) |
| Run on GPUs | [docs/GPU.md](docs/GPU.md), `gpu/modal_app.py` |

## Invariants

- **Every kernel matches the reference.** The CPU, fused and GPU kernels are tested against
  `core/reimage_ref.py` (bit-equal feature matrices, or within a stated tolerance).
- **The held-out split is locked before any search.** `core/heldout.py` writes the manifest and the
  loaders refuse to load locked patients for discovery.
- **Every published number has a source.** `results/claims.json` binds each headline number to the
  file it came from; the claims test fails on drift.
