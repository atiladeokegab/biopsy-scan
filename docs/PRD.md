# sota-parallelism-nov21: PRD v1

> C-001 is proposed, not approved: R-12 and the novelty principle are marked "proposed (C-001)"; R-11 moves to rank 12. Spec of record: v1.

## Problem
Before a lab images a spatial-biology cohort it must choose how many patients, how many regions per patient, how large a field of view and how fine a cell typing. Each choice trades cost against the chance of finding a real survival-linked spatial pattern, and today a lab sizes that by guesswork or a textbook formula that ignores the imaging. Simulating it honestly means re-imaging thousands of studies cell by cell from real tissue, which is CPU-days with the per-study cKDTree recount and permutation test a lab runs today.

## Users
- Lab scientist: sets patients, regions per patient, field of view and mis-call rate, presses Run the studies, and sees power and identification rate for that design before imaging a slide
- Judge: hears a live 2-3 minute pitch with a clean PDF deck; the demo video is the backup; scores correctness and assumptions, fundamental reshaping, meaningful performance improvement and scientific value
- Teammate agent (any laptop): reads this PRD and its GitHub issues to build one requirement without asking

## Goals
- **Simulate thousands of study designs on real tissue in seconds**: A lab sees what its design can and cannot find before it images a slide. Measure: One design point of 1,000 re-imaged studies, timed end to end on a named GPU and labelled MEASURED; no number before it is measured
- **Quote every speed-up against what a lab runs today**: The performance criterion is scored against a real baseline, not our own slowest bar. Measure: The ladder JSON has rung 0 (per-study cKDTree recount plus a permutation test) and every ratio is computed against it
- **Show the parallelism on stage**: Studies are independent, so the fleet slide makes the reshaping visible. Measure: bench/fleet.py reports studies/s at 1, 2, 4 and 8 GPUs, each MEASURED or UNMEASURED, with results bit-identical to one device
- **Show why a textbook formula is not enough**: Scientific value: imaging error and spatial sampling change power. Measure: bench/results/vs-schoenfeld.json compares Schoenfeld power with Preflight simulated power on the same design points

## Principles
- Build in rank order; stop when time runs out: main is always demoable
- Conflict rule: when this PRD and the merged code or real data disagree, the code and data win; say so in the PR
- No invented numbers. Every figure carries one label: MEASURED, SYNTHETIC, UNMEASURED, ESTIMATE or ANALYTIC; an empty slot is [__], never a guess
- Charville is the only patient-level cohort (P=159, 38 events, strata are coverslips). UPMC is descriptive only
- Real-data results are exploratory: their outcomes have already been seen
- Every speed-up is quoted against rung 0, what a lab runs today
- The tagline is the h1: "Thousands of studies, simulated in seconds, before you image a slide." The button reads "Run the studies". The phrase "Can my study find it?" appears nowhere in the project
- Tests come from the goal: unseen geometries and tied studies, a planted signal and a hard negative; a check is believed only after it has been seen red
- Novelty: Preflight produces new biology by combining protein expression, cell morphology, neighbourhood relations and stromal architecture at single-cell resolution, with an algorithm parallelised on GPU clusters so it is much faster than on a CPU _(proposed, C-001)_

## Non-goals
- Segmentation, a backend service, or any clinical claim
- Pooled-cohort atlases or patient-level inference on UPMC
- A custom GEMM, a fused max, TF32/FP16, GPU neighbour search, LOCO, Helsinki, or any 300k pattern figure
- Step-down maxT, stress-test slices and source bootstrap before P1 is green (waiting room)

## Owner
Atilade (lead), Team Dry Lab

## Glossary
- **study**: One simulated study: n patients, k regions per patient, a field-of-view crop per region and a per-cell mis-call draw, re-imaged from real tissue
- **design point**: One setting of (patients, regions per patient, field of view, mis-call rate, effect); 1,000 studies by default
- **rung 0**: What a lab runs today: per study, a cKDTree feature recount plus a permutation test
- **fleet**: Studies sharded by seed across N GPUs (--devices N); merged results are bit-identical to one device
- **biomarkers**: The tissue's cell types (tumour, immune, stroma), shown true and as a study sees them

## Constraints
- Stack: `pyproject.toml and uv.lock (Python 3.11 via uv: NumPy, SciPy, Numba, CUDA); demo/ is static HTML, JS and three.js`
- Tests run offline: yes
- Limit: make smoke runs offline from a fresh clone with no real data and no GPU (why: CI and teammates' laptops have neither) (check: make smoke)
- Limit: A CUDA simulator run never yields a GPU timing rung (why: A simulator time is not a GPU speed-up) (check: bench/test_bench.py)
- Limit: The demo works with Wi-Fi off from saved runs (why: Venue Wi-Fi fails on stage) (check: demo/check_data.py)
- Invariant: With no crop and mis-call rate 0, the re-imager reproduces the truth layer exactly (why: The kernel's correctness anchor) (check: core/test_spine.py)
- Invariant: NumPy reference, Numba CPU and GPU kernels agree cell for cell (why: A speed-up is only meaningful at equal output) (check: reimage/test_backends.py)
- Invariant: Real-data result records are exploratory, already seen and carry no clinical claim; UPMC stays descriptive (why: Both cohorts' outcomes were examined before the event) (check: audit/test_audit.py)
- Invariant: Speed-up ratios are against the named reference rung, never the slowest bar (why: A ratio against a strawman scores nothing) (check: bench/test_bench.py)
- Tag: stopping_rule: build in rank order; stop when time runs out
- Tag: conflict_rule: merged code and real data win over prose
- Tag: labels: MEASURED, SYNTHETIC, UNMEASURED, ESTIMATE, ANALYTIC
- Tag: submission: live 2-3 minute pitch with a PDF deck from the Slides artifact https://claude.ai/artifact/HsYFYZZygbAzNALjz2bm5Q; demo video is the backup
- Tag: deck_design: design/deck/system.md (hub project dir)
- Tag: plan_of_record: ~/hub/projects/sota-parallelism-nov21/IDEA.md (v6 and v5 overrides beat older text); coverage map prd-coverage.md beside it
- Tag: freeze: code freeze 2026-11-22T16:00Z, submit 2026-11-22T17:00Z (HACKATHON.md UTC column)

## Requirements
### R-1 [P0 · Core] Cohorts, patient identity, frozen grammar and the NumPy reference re-imager
rank 1 · state built · needs nothing · touches `core/`, `scripts/fetch_data.sh`
Why: Every later number rests on correct patients, endpoints and features
- A-1.1@1 Scenario: with no crop and mis-call rate 0 the reference re-imager reproduces the truth layer exactly; a crop equals the truth of the visible cells (`core/test_spine.py`)
- A-1.2@1 Scenario: Charville loads as 159 patients with 38 events, strata are coverslips, and patients with no dates or spanning coverslips are excluded and named; the grammar count and hash are frozen (`core/test_spine.py`)

### R-2 [P0 · Stats] Stratified score, permutation variance, maxT, omnibus and the unconditional threshold
rank 2 · state built · needs nothing · touches `stats/`
Why: Correctness and assumptions: the statistic must match exact answers before any power is quoted
- A-2.1@1 Scenario: U matches statsmodels PHReg (Breslow, ties) and the variance matches exhaustive within-stratum permutations, including event-free and singleton strata (`stats/test_reference.py`)
- A-2.2@1 Scenario: a planted ordered signal is found and a batch-confounded null is a hard negative; the null threshold uses independent maxima (`tests/test_score_contract.py`)

### R-3 [P0 · Kernel] Re-imaging kernel on CPU and GPU, cell for cell equal to the reference
rank 3 · state approved · needs R-1 · touches `reimage/`
Why: The GPU hero workload; a speed-up only counts at identical output
- A-3.1@1 Scenario: Numba CPU and GPU (CUDA simulator in CI) match the NumPy reference per region and per study batch on unseen irregular geometries; focal cells are relabelled too (`reimage/test_backends.py`)
- A-3.2@1 Scenario: on a real GPU, 1,000 studies at one design point equal the CPU kernel's output; the GPU model is named in the result. Not yet built: no hardware GPU run recorded (`reimage/test_backends.py`)

### R-4 [P1 · Atlas] Design atlas: simulator, shared-threshold shortcut and its nested check, grid runner and charts (#5)
rank 4 · state approved · needs R-2, R-3 · touches `atlas/`, `scripts/export_grid.py`, `demo/data/grid.json`
Why: Scientific value: power and identification across the designs a lab controls
- A-4.1@1 Scenario: the shared-threshold shortcut agrees with nested permutations on 3 design points within Monte Carlo error, and FWER from independent null studies is reported with binomial intervals. Built on branch 5-study-atlas, not in integration yet (`atlas/test_simulation.py`)
- A-4.2@1 Scenario: export_grid writes the atlas as preflight-grid/1 for the demo, each cell labelled. Open as PR #30 (`tests/test_export_grid.py`)

### R-5 [P1 · Ladder] Benchmark ladder from what a lab runs today to the GPU (#3)
rank 5 · state approved · needs R-3 · touches `bench/ladder.py`, `bench/chart.py`, `bench/results/`
Why: Meaningful performance improvement, quoted against the lab's own method as the Bio-hack winners did
- A-5.1@1 Scenario: the ladder JSON has rung 0, a per-study cKDTree recount plus a permutation test, and its baseline field names rung 0; every ratio is against it. Not yet built: the current baseline is numpy_reference (`bench/test_bench.py`)
- A-5.2@1 Scenario: rungs algebra, Numba all-core and GPU are measured on named hardware; a CUDA simulator run never yields a GPU rung; truncated or misshapen outputs are rejected (`bench/test_bench.py`)

### R-6 [P1 · Fleet] Studies sharded across a fleet of GPUs (--devices N)
rank 6 · state approved · needs R-3 · touches `reimage/fleet.py`, `bench/fleet.py`
Why: The pitch must show the parallelism: studies are independent
- A-6.1@1 Scenario: --devices N shards study seeds deterministically and the merged result is bit-identical to one device. Built on branch 33-gpu-fleet (PR #34), not in integration yet (`reimage/test_fleet.py`)
- A-6.2@1 Scenario: bench/fleet.py reports studies/s at 1, 2, 4 and 8 GPUs, each MEASURED on named GPUs or UNMEASURED. Not yet measured on hardware (`reimage/test_fleet.py`)

### R-7 [P1 · Schoenfeld] Textbook formula versus Preflight simulated power
rank 7 · state approved · needs R-4 · touches `bench/vs_schoenfeld.py`, `bench/results/vs-schoenfeld.json`
Why: Scientific value: shows what the imaging design does that a formula cannot see
- A-7.1@1 Scenario: bench/results/vs-schoenfeld.json gives, per design point, Schoenfeld power (ANALYTIC) and Preflight simulated power with its interval. Code on branch 32-vs-schoenfeld; the results file is not yet built (`bench/test_vs_schoenfeld.py`)

### R-8 [P1 · Audit] Real-data audit, exploratory and already seen
rank 8 · state built · needs R-1, R-2 · touches `audit/`
Why: Correctness and assumptions: say exactly what the real cohorts can and cannot support
- A-8.1@1 Scenario: Charville results are patient-level exploratory, UPMC results descriptive only; every record is marked already seen with no clinical claim; leave-one-patient-out influence and the leakage audit run (`audit/test_audit.py`)

### R-9 [P1 · Demo] The demo page: Run the studies (#7)
rank 9 · state approved · needs R-4, R-5 · touches `demo/`, `README.md`, `docs/design.md`
Why: The judge's button: a lab's own knobs, an answer in seconds, Wi-Fi off
- A-9.1@1 Scenario: given measured CPU 1 s and GPU 2 s, the race reports "GPU 2.0× slower", never faster (`demo/race.test.mjs`)
- A-9.2@1 Scenario: the h1 is "Thousands of studies, simulated in seconds, before you image a slide.", the button reads "Run the studies", and grep finds no "Can my study find it?" anywhere in the repo. Not yet built: the page still uses the old phrase (`demo/copy.test.mjs`)

### R-10 [P1 · Deck] Clean PDF deck for a live 2-3 minute pitch
rank 10 · state approved · needs R-5, R-6, R-7 · touches `design/deck/system.md`, `docs/SUBMISSION.md`
Why: The submission is a live pitch; the deck carries the numbers and the before/after
- A-10.1@1 Scenario: the deck exports from the Slides artifact as PDF and follows design/deck/system.md; every figure carries MEASURED, SYNTHETIC, UNMEASURED, ESTIMATE or ANALYTIC. Not yet built: check by reading the exported PDF (`docs/SUBMISSION.md`)
- A-10.2@1 Scenario: two before/after slides show the tissue's cell types: the true tissue, and the same tissue as a study sees it; one slide shows the fleet and the ladder against rung 0. Not yet built (`docs/SUBMISSION.md`)

### R-12 [P1 · Expression] Protein, morphology and stroma channels in the per-cell kernel (grammar v2, #41)
rank 11 · state proposed (C-001) · needs R-1, R-3 · touches `core/cohorts.py`, `core/`, `reimage/gpu.py`, `results/charville-v2.json`
Why: The novelty claim: grammar v1 has only neighbourhood relations; v2 adds protein expression, morphology and stromal architecture, computed by the same per-cell kernel
- A-12.1@1 Scenario: for the 10 fixed channels (PD1, PDL1, Ki67, GranzymeB, HLA-DR, aSMA, CollagenIV, Vimentin, Podoplanin, SIZE; log1p then cohort-wide z-score fixed before re-imaging), the GPU kernel under the CUDA simulator matches the NumPy reference exactly or within a stated float tolerance; crop and mis-call change who the neighbours are, never a cell's expression; the v1 grammar hash is unchanged. Not yet built (`reimage/test_expression.py`)
- A-12.2@1 Scenario: with v2 features on, the fleet result is still bit-identical to one device. Not yet built (`reimage/test_fleet_expression.py`)
- A-12.3@1 Scenario: results/charville-v2.json on real Charville (159 patients, coverslip strata) gives grammar hashes for v1 and v2, the channel m, maxT, p and omnibus p over v1+v2, labelled exploratory (outcomes already seen), every number MEASURED or UNMEASURED. Not yet built (`audit/test_charville_v2.py`)

### R-11 [P2 · Submission] README, backup demo video and the submission form (#9-#12)
rank 12 (proposed, C-001; was 11) · state approved · needs R-9, R-10 · touches `README.md`, `docs/film/`, `docs/SUBMISSION.md`
Why: The video is the backup if the live demo fails; the form is how the event receives us
- A-11.1@1 Scenario: README states what it is, how to run it, and a screenshot from the final run; the demo video plays the same 2-3 minute story; every submission checklist row is done before 2026-11-22T17:00Z. Not yet built (`docs/SUBMISSION.md`)

## References
- design: `design/deck/system.md` sha256 467717e415e3

## Open questions
- Which GPU model and how many devices are available for the measured GPU and fleet rungs?
- Judging weights across the four criteria are unknown

## Waiting room
- Stress-test slices: non-proportional hazards, heavy censoring, unmeasured confounder
- Step-down maxT under subset pivotality
- Source bootstrap bands on the atlas

