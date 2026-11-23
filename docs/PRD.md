# Product requirements v5 (hackathon, 22 Nov 2026)

> **Historical.** This is the hackathon product requirements document as written on 22 Nov 2026. The hosted demo is
> retired: run the demo locally with `make demo`. Current results are in
> [RESULTS.md](https://github.com/atiladeokegab/biopsy-scan/blob/main/RESULTS.md).

## Problem
After surgery for head and neck cancer, treatment is chosen on stage and margins, which cannot see the tissue patterns that predict relapse. Finding one such pattern took the Helsinki team years and 18 hand-picked tests, and turning it into a hospital test needs proof that it survives fewer stains, imaging error and small cohorts. Testing that honestly means re-imaging thousands of simulated studies cell by cell from real tissue, each running every pattern's survival test with 10,000 shuffles: CPU-days today (rung 0: 7.52 s per study on one core, MEASURED).

## Users
- Biomarker scientist (Decode): searches every spatial pattern with a family-wise correction and sees which survive
- Diagnostic developer (Translate): sets patients, regions per patient, field of view and mis-call rate, presses Run the studies, and sees whether a pattern survives one or two routine stains before a slide is imaged
- Judge: hears a live 5-minute pitch with a clean PDF deck; the demo video is the backup; scores correctness and assumptions, fundamental reshaping, meaningful performance improvement and scientific value
- Teammate agent (any laptop): reads this PRD and its GitHub issues to build one requirement without asking

## Goals
- **Test whether a biomarker survives the real world, thousands of simulated studies in seconds**: A lab sees what its design can and cannot find before it images a slide. Measure: One design point of 1,000 re-imaged studies, timed end to end on a named GPU and labelled MEASURED; no number before it is measured
- **Quote every speed-up against what a lab runs today**: The performance criterion is scored against a real baseline, not our own slowest bar. Measure: The ladder JSON has rung 0 (per-study cKDTree recount plus a permutation test) and every ratio is computed against it
- **Show the parallelism on stage**: Studies are independent, so the fleet slide makes the reshaping visible. Measure: bench/fleet.py reports studies/s at 1, 2, 4 and 8 GPUs, each MEASURED or UNMEASURED, with results bit-identical to one device
- **Show why a textbook formula is not enough**: Scientific value: imaging error and spatial sampling change power. Measure: bench/results/vs-schoenfeld.json compares Schoenfeld power with Biopsy Scan simulated power on the same design points

## Principles
- Build in rank order; stop when time runs out: main is always demoable
- Conflict rule: when this PRD and the merged code or real data disagree, the code and data win; say so in the PR
- No invented numbers. Every figure carries one label: MEASURED, SYNTHETIC, UNMEASURED, ESTIMATE or ANALYTIC; an empty slot is [__], never a guess
- Charville is patient-level (P=159, 38 events, strata are coverslips). UPMC is patient-level only through the R-13 patient loader and its locked held-out split; everything else on UPMC stays descriptive
- Real-data results are exploratory: their outcomes have already been seen
- Every re-imaging speed-up is quoted against rung 0, what a lab runs today. A GPU analysis-half speed-up is quoted only against the same statistic and resampling workload on the CPU (R-15); the full-fit comparison against Implementation A (R-14) is a workflow comparison, labelled as one
- The tagline is the h1: "Find the tissue patterns that decide a cancer patient's treatment, prove they are not luck, and check they still work as a test a hospital can run." The button reads "Run the studies". Until after freeze (2026-11-22 16:00Z) the demo/ h1 keeps the old tagline, "Thousands of studies, simulated in seconds, before you image a slide." The phrase "Can my study find it?" appears nowhere in the project
- Tests come from the goal: unseen geometries and tied studies, a planted signal and a hard negative; a check is believed only after it has been seen red
- Novelty goal (not yet measured): Biopsy Scan aims to produce new biology by combining protein expression, cell morphology, neighbourhood relations and stromal architecture at single-cell resolution, with an algorithm parallelised on GPU clusters, much faster than on a CPU
- No discovery code reads the locked UPMC patients; they are opened once, for the R-16 score, after the search is frozen

## Non-goals
- Segmentation, a backend service, or any clinical claim
- Pooled-cohort atlases, or UPMC patient-level inference outside the R-13 loader and locked split
- A custom GEMM, a fused max, TF32/FP16, GPU neighbour search, LOCO, the Helsinki cohort, or any 300k pattern figure
- Step-down maxT, stress-test slices and source bootstrap before P1 is green (waiting room)
- By freeze (2026-11-22 16:00Z): the image half (registration to quantification), HANCOCK validation, the brightfield scoring engine (Translate step 3), Translate step 4, Tinker, and ligand-receptor analysis

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
- Tag: submission: live 5-minute pitch with a PDF deck; demo video is the backup
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

### R-7 [P1 · Schoenfeld] Textbook formula versus Biopsy Scan simulated power
rank 7 · state approved · needs R-4 · touches `bench/vs_schoenfeld.py`, `bench/results/vs-schoenfeld.json`
Why: Scientific value: shows what the imaging design does that a formula cannot see
- A-7.1@1 Scenario: bench/results/vs-schoenfeld.json gives, per design point, Schoenfeld power (ANALYTIC) and Biopsy Scan simulated power with its interval. Code on branch 32-vs-schoenfeld; the results file is not yet built (`bench/test_vs_schoenfeld.py`)

### R-8 [P1 · Audit] Real-data audit, exploratory and already seen
rank 8 · state built · needs R-1, R-2 · touches `audit/`
Why: Correctness and assumptions: say exactly what the real cohorts can and cannot support
- A-8.1@1 Scenario: Charville results are patient-level exploratory, UPMC results descriptive only; every record is marked already seen with no clinical claim; leave-one-patient-out influence and the leakage audit run (`audit/test_audit.py`)

### R-9 [P1 · Demo] The demo page: Run the studies (#7)
rank 9 · state approved · needs R-4, R-5 · touches `demo/`, `README.md`, `docs/design.md`
Why: The judge's button: a lab's own knobs, an answer in seconds, Wi-Fi off
- A-9.1@1 Scenario: given measured CPU 1 s and GPU 2 s, the race reports "GPU 2.0× slower", never faster (`demo/race.test.mjs`)
- A-9.2@1 Scenario: the h1 is "Thousands of studies, simulated in seconds, before you image a slide.", the button reads "Run the studies", and grep finds no "Can my study find it?" anywhere in the repo. Not yet built: the page still uses the old phrase (`demo/copy.test.mjs`)

### R-10 [P1 · Deck] Clean PDF deck for a live 5-minute pitch
rank 10 · state approved · needs R-5, R-6, R-7 · touches `design/deck/system.md`, `docs/SUBMISSION.md`
Why: The submission is a live pitch; the deck carries the numbers and the before/after
- A-10.1@1 Scenario: the deck exports from the Slides artifact as PDF and follows design/deck/system.md; every figure carries MEASURED, SYNTHETIC, UNMEASURED, ESTIMATE or ANALYTIC. Not yet built: check by reading the exported PDF (`docs/SUBMISSION.md`)
- A-10.2@1 Scenario: two before/after slides show the tissue's cell types: the true tissue, and the same tissue as a study sees it; one slide shows the fleet and the ladder against rung 0. Not yet built (`docs/SUBMISSION.md`)

### R-12 [P1 · Expression] Protein, morphology and stroma channels in the per-cell kernel (grammar v2, #41)
rank 11 · state proposed · needs R-1, R-2, R-3, R-6 · touches `core/`, `reimage/cpu.py`, `reimage/gpu.py`, `results/charville-v2.json`
Why: The novelty goal: grammar v1 has only neighbourhood relations; v2 adds protein expression, morphology and stromal architecture from the same per-cell kernel, searched jointly with v1
- A-12.1@1 Scenario: feature (t, j, r, m) is the mean of channel m over the visible type-j neighbours within r of each focal cell of type t, 0 when there are none; patient feature is the focal-cell-weighted mean, regions pooled; v2 columns are ordered t, j, r, m and appended after v1. Channels PD1, PDL1, Ki67, GranzymeB, HLA-DR, aSMA, CollagenIV, Vimentin, Podoplanin, SIZE take a cohort-wide z-score directly (no log1p; the source pre-normalises), z = 0 for a zero-variance channel, and any missing value fails loudly. Not yet built (`core/test_expression.py`)
- A-12.2@1 Scenario: v1 outputs, hash and default ordering are byte-for-byte unchanged; CPU and GPU backends (CUDA simulator in CI) match the NumPy reference within a stated float tolerance; crop and mis-call change who the neighbours are, never a cell's expression; with v2 on, the fleet result is still bit-identical to one device. Not yet built (`reimage/test_expression.py`)
- A-12.3@1 Scenario: maxT and omnibus run over v1+v2 jointly as one family; results/charville-v2.json on real Charville (159 patients, coverslip strata) gives v1 and v2 grammar hashes, the channel m, maxT, p and omnibus p, labelled exploratory (outcomes already seen), every number MEASURED or UNMEASURED. Not yet built (`audit/test_charville_v2.py`)

### R-11 [P2 · Submission] README, backup demo video and the submission form (#9-#12)
rank 12 · state approved · needs R-9, R-10 · touches `README.md`, `docs/film/`, `docs/SUBMISSION.md`
Why: The video is the backup if the live demo fails; the form is how the event receives us
- A-11.1@2 Scenario: README states what it is, how to run it, and a screenshot from the final run; the demo video plays the same 5-minute story; every submission checklist row is done before 2026-11-22T17:00Z. Not yet built (`tests/test_submission.py`)

### R-13 [P1 · UPMC patients] UPMC patient-level loader and a locked held-out split (#56)
rank 13 · state proposed · needs R-1 · touches `core/cohorts.py`, `core/holdout.py`, `data/holdout-manifest.json`
Why: UPMC labels have no patient ID column; grouping regions by patient and locking validation patients before any search is what makes a UPMC discovery claim honest
- A-13.1@1 Scenario: UPMC regions are grouped by patient using the documented rule (the sample_label_visualizer case number where present); the loader reports the number of patients, events and unmapped regions from the data, and no two patients share a region. Not yet built (`core/test_upmc_patients.py`)
- A-13.2@1 Scenario: a held-out set of patients is drawn stratified by event with a frozen seed and written to a committed manifest whose sha256 is recorded; re-running the split reproduces the same manifest byte for byte. Not yet built (`core/test_holdout.py`)
- A-13.3@1 Scenario: discovery-only patient IDs govern every discovery step, including fitted preprocessing, transforms and caches as well as the named entry points (candidate build, screen, maxT, panel search); a discovery entry point handed a locked patient raises an error, and changing any locked patient's values leaves every scientific discovery output (candidates, statistics, ranks, fitted parameters) byte for byte unchanged; wall-clock and run metadata are excluded. Not yet built (`tests/test_no_leak.py`)

### R-14 [P1 · Implementation A] CPU baseline for the analysis half: per-candidate statsmodels PHReg Cox fits in a ProcessPoolExecutor on all cores
rank 14 · state proposed · needs R-13 · touches `bench/impl_a.py`, `bench/results/`
Why: A full-fit CPU workflow comparator for the analysis half: one Cox fit per candidate gives coefficients and Wald p-values, a different output from the R-15 score screen, so it is reported as a workflow comparison, not an equal-work speed-up. The plan named lifelines; statsmodels PHReg (already locked) is substituted because lifelines 0.30.3 needs pandas<3 and the lead ruled out a pandas downgrade (#59)
- A-14.1@1 Scenario: on a small fixture, each candidate's statsmodels PHReg fit (clinical covariates plus the candidate) run through concurrent.futures.ProcessPoolExecutor with workers=all cores gives the same coefficients and p-values as workers=1, and as a direct serial PHReg call. Not yet built (`bench/test_impl_a.py`)
- A-14.2@1 Scenario: the baseline writes a results JSON naming the CPU model, cores used, library versions, the sha256 fingerprints of the patient rows, the candidate matrix X and the clinical matrix C, the outputs timed (coefficients and Wald p-values per candidate), and wall time labelled MEASURED; it is compared only with a run whose three fingerprints match, and the comparison is labelled a workflow comparison. Not yet built (`bench/test_impl_a_result.py`)

### R-15 [P1 · GPU screen] Every candidate as a column: a clinical-null Cox score screen on GPU with a valid adjusted null (R2-R4)
rank 15 · state proposed · needs R-2, R-6, R-13, R-14 · touches `stats/`, `bench/screen.py`
Why: The reshaping the judges score: discovery becomes dense matrix products on the GPU. Exists today: stats/reference.py computes the stratified score U = X'r with its permutation variance, maxT and omnibus in float64 on the CPU, with no clinical covariates; reimage/ plus the fleet compute features on GPU. Inputs per cohort: X is patients by candidates (one column per candidate feature from the R-2 feature build, NaN where a feature is undefined); C is the clinical matrix: UPMC C = [HPV status from the hpvstatus label, binary], Charville C = none (zero-covariate mode, as the loaders supply no stage, margins or age); strata are coverslip groups in both. Any other clinical covariate needs a C-NNN. Reference statistic: for candidate j, the analysis set is the patients with a finite X_j, complete C, an endpoint and a stratum; a stratified Cox null model on C alone (Breslow ties) is fitted on that set, refitted once per distinct complete-case mask; U_j is the Cox score for beta_j at (beta_C-hat, 0) and V_j = I_jj - I_jC I_CC^-1 I_Cj, the nuisance-adjusted Cox information; Z_j = U_j / sqrt(V_j). The legacy permutation-variance statistic is a different statistic and stays only as a regression for the zero-covariate exchangeable case. The plan named lifelines; statsmodels PHReg (already locked) is substituted because lifelines 0.30.3 needs pandas<3 and the lead ruled out a pandas downgrade (#59)
- A-15.1@1 Scenario: on random candidates and on fixtures with tied times and missing X, Z_j from the reference statistic equals the oracle within a stated tolerance: statsmodels PHReg on [C, X_j] with the same strata and ties='breslow' over the same analysis set, its score and hessian evaluated at (beta_C-hat, 0) and V formed as the adjusted Schur complement; on the 3-patient fixture X=[0,NaN,2] from the review on #60 the screen returns the complete-case U, not the full-residual U. Not yet built (`stats/test_vs_statsmodels.py`)
- A-15.2@1 Scenario: the GPU U, V and Z (CUDA simulator in CI) equal a float64 CPU implementation of the reference statistic on the same X, C, time, event and strata within a stated float tolerance; in zero-covariate mode a separate regression check keeps the legacy permutation-variance path equal to stats.reference. Not yet built (`stats/test_gpu_score.py`)
- A-15.3@1 Scenario: in the zero-covariate exchangeable setting only, maxT over B shuffles computed as X'[P_1 m, ..., P_B m], streamed in column blocks with the maximum taken on the fly, equals a brute-force loop over the same permutations on sampled candidates; this checks implementation agreement, not false-positive control. Not yet built (`stats/test_perm_product.py`)
- A-15.4@1 Scenario: a timing JSON reports the same statistic and the same resampling workload (identical patient-row, X and C fingerprints and resample count B) on a named GPU and on all CPU cores, as the equal-work speed-up; the R-14 full-fit time appears separately, labelled a workflow comparison; each figure MEASURED or UNMEASURED. Not yet built (`bench/test_screen_result.py`)
- A-15.5@1 Scenario: the adjusted null is a clinical-null parametric bootstrap: per resample, event times are drawn from the fitted stratified clinical-null model, censoring from a censoring model on C within strata, the null model is refitted and U, V and Z recomputed for every candidate, and maxT is taken; the method and its assumptions (proportional hazards in C, censoring independent of X given C) are written into the result JSON with B; per-resample U for all candidates is one dense product on GPU and equals a per-candidate CPU loop on sampled candidates. Not yet built (`stats/test_null_bootstrap.py`)
- A-15.6@1 Scenario: a hard-negative synthetic fixture (SYNTHETIC) has candidates correlated with C, hazard depending on C, censoring depending on C, and no candidate effect given C; over a stated number of null datasets the screen's family-wise rejection rate at alpha = 0.05 has a Clopper-Pearson 95% interval that contains 0.05, and on the same fixture the raw-residual shuffle and the unadjusted screen are each run and their rejection rates reported, so the check is seen red on a method that ignores C. Not yet built (`stats/test_hard_negative.py`)

### R-16 [P1 · Rediscovery] pEMT tumour bordered by fibroblasts, rediscovered on UPMC and scored on the locked patients
rank 16 · state proposed · needs R-13, R-15 · touches `audit/rediscovery.py`, `results/upmc-rediscovery.json`
Why: Discovery credibility: does the blind search rank a known biology among its strongest signals in an independent cohort
- A-16.1@1 Scenario: the pEMT-tumour-with-fibroblasts candidate, its transforms and its score direction are defined from UPMC cell types or markers by a rule committed before the search (hash recorded) and frozen before the locked set is opened; if UPMC cannot express it, the result says so and stops. Not yet built (`audit/test_rediscovery_definition.py`)
- A-16.2@1 Scenario: results/upmc-rediscovery.json gives the candidate's rank among all candidates on discovery patients, then, with the frozen candidate and direction, its Harrell C-index and Cox hazard ratio on the locked patients, opened once, each labelled MEASURED and exploratory (outcomes already seen, internal split), or UNMEASURED with a reason when not estimable. Not yet built (`audit/test_rediscovery_result.py`)

### R-17 [P1 · Translate 1-2] Minimal panels and IHC format simulation on UPMC
rank 17 · state proposed · needs R-13, R-15, R-16 · touches `translate/`, `results/upmc-translate.json`
Why: A discovery helps patients only as a test a pathology lab can run; this shows in software how much prognostic power survives one or two stains and each IHC format
- A-17.1@1 Scenario: step 1 enumerates every one- and two-marker panel from UPMC's M markers (M + M(M-1)/2 panels, 820 when M = 40); every transform, gate and panel selection is fitted inside the training folds of nested cross-validation on discovery patients only, the panel model hash is recorded, and R-17 never reads or uses locked patients; each panel is scored on the same outer test patients as the full model by retained C-index (C_panel - 0.5)/(C_full - 0.5), with C the Harrell C-index of the fitted Cox risk score, paired bootstrap intervals over outer test patients, and UNMEASURED with a reason when C_full <= 0.5.; if R-16 reports the candidate unsupported (UPMC cannot express it), every R-17 result is UNMEASURED with that reason. Not yet built (`translate/test_panels.py`)
- A-17.2@1 Scenario: step 2 recomputes the biomarker as single-slide IHC (two-component Gaussian mixture positivity on arcsinh intensities, the mixture and arcsinh cofactor fitted on training folds only), as serial sections with misregistration swept 0-100 um, and on 100, 150 and 200 um tiles, reporting retained C-index per format, defined and paired as in A-17.1, with intervals; at zero misregistration the serial-section result equals the single-slide result. Not yet built (`translate/test_formats.py`)

### R-18 [P1 · Colorectal] The discovery stage rerun unchanged on Charville
rank 18 · state proposed · needs R-15 · touches `audit/`, `results/charville-screen.json`
Why: Shows the discovery stage generalises beyond one cancer
- A-18.1@1 Scenario: the same screen code path as UPMC runs on Charville (159 patients, coverslip strata) in declared zero-covariate mode (C = none) with no cohort-specific branch, and results/charville-screen.json gives maxT, the adjusted-null p and the top candidates, records C and the null method, and is labelled exploratory (outcomes already seen) and MEASURED. Not yet built (`audit/test_charville_screen.py`)

## References
- design: `design/deck/system.md` sha256 467717e415e3

## Open questions
- Which GPU model and how many devices are available for the measured GPU and fleet rungs?
- Judging weights across the four criteria are unknown

## Waiting room
- Stress-test slices: non-proportional hazards, heavy censoring, unmeasured confounder
- Step-down maxT under subset pivotality
- Source bootstrap bands on the atlas
