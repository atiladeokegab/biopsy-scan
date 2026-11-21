# Preflight against the four judging criteria

Every number below is copied from a results file on `integration`, except the one row marked otherwise. Labels:
- **MEASURED**: a committed results file from a real run.
- **TESTED**: a passing test in `make smoke`.
- **ESTIMATE**: arithmetic on measured rows, not a run.
- **PENDING**: not on `integration` yet.

All real-data results are exploratory: the outcomes have already been seen.

## 1. Correctness and assumptions

| Claim | Evidence | Test | Label |
|---|---|---|---|
| With no crop and no mis-calls, re-imaging reproduces the full-tissue truth exactly | `core/` | `core/test_spine.py::test_no_crop_eps0_reproduces_truth_exactly` | TESTED |
| The CPU and GPU (CUDA simulator) backends count the same neighbours as direct geometry, on unseen irregular regions | `reimage/` | `tests/test_reimage_contract.py`, `reimage/test_backends.py::test_region_matches_reference` | TESTED |
| The score equals statsmodels PHReg (Breslow ties); its variance equals exhaustive permutation enumeration | `stats/reference.py` | `stats/test_reference.py::test_score_matches_phreg_breslow_with_ties`, `tests/test_score_contract.py::test_variance_is_exhaustive_permutation_variance` | TESTED |
| A planted signal is found; batch confounding gives no evidence (hard negative) | `stats/reference.py` | `tests/test_score_contract.py::test_planted_ordered_signal_is_found`, `::test_batch_confounding_is_a_hard_negative` | TESTED |
| When nothing is there, the false-alarm rate stays near 5%: simulated FWER is 0.0 to 0.1 at all 7 patient counts, and every 95% interval contains 0.05 (20 null studies per point) | `bench/results/vs-schoenfeld.json` | `bench/test_vs_schoenfeld.py` | MEASURED |
| Two GPU shards give bit-identical results to one device | `reimage/fleet.py` (#34) | `reimage/test_fleet.py::test_two_fake_shards_on_cuda_simulator_are_bit_identical` | TESTED on the CUDA simulator; real GPU PENDING |

**Assumptions and why they hold**
- **Patients and strata.** Charville is patient-level: 159 patients, 38 events, coverslip strata, and 0 patients span two coverslips (`results/charville-v2.json` `leakage_audit`). UPMC stays descriptive until its patient loader and locked split land (#62, PENDING).
- **Missing values.** A feature that is undefined for a patient is missing, not zero; risk sets are recomputed per feature on complete cases (`stats/test_reference.py::test_missing_features_use_complete_case_risk_sets_and_batch`). Charville v2 keeps 11,880 NaN cells.
- **Clinical covariates in the C-003 screen.** UPMC adjusts for HPV status only. Charville uses no covariates, because its labels carry no stage, margins or age. Both use coverslip strata (PRD R-15).
- **Imaging model.** A study is patients drawn with replacement, up to two regions each, a square field of view and a cell-typing mis-call rate eps (`bench/results/cpu-ladder.json` `sampling`). The model is a simplification of real imaging; it is seeded and reproducible (`core/test_spine.py::test_reimage_design_draws_are_seeded_and_change_x`).
- **Timing workloads.** Rung 0 re-uses resampled outcomes as a timing workload only; no p-value is claimed from it (`cpu-ladder.json` `lab_today.inference_note`).

## 2. Reshaping for parallelism

| Reshaping | Evidence | Test | Label |
|---|---|---|---|
| Every study is independent, and every cell's neighbour count is its own thread: one kernel counts a whole batch of studies | `reimage/` | `reimage/test_backends.py::test_study_batch_matches_shared_draws` | TESTED |
| Studies are sharded across a fleet of GPUs, with results equal to one device | `reimage/fleet.py` | `reimage/test_fleet.py`, `bench/test_bench.py::test_fleet_reports_1_2_4_8_measured_or_unmeasured_without_guessing` | TESTED; fleet timing PENDING |
| Testing every candidate becomes one matrix product: U = X'r, with one column per candidate (3,780 on Charville) | `stats/reference.py` | `stats/test_reference.py::test_batch_and_patient_order_invariance` | TESTED on CPU; GPU screen PENDING (R-15) |
| 10,000 shuffles become one product X'[P_1 m, ..., P_B m]; the clinical-adjusted null is a refit bootstrap | PRD R-15 (A-15.3, A-15.5) | not yet built | PENDING |

## 3. Speed-up against a credible baseline

| Workload | Result | Source | Label |
|---|---|---|---|
| Rung 0, what a lab runs today (cKDTree recount plus 1,000 permutations, single core) | 7.52 s per study (mean of 3) | `bench/results/cpu-ladder.json` `lab_today` | MEASURED |
| Rung 0 for one 1,000-study design | 7,521 s, about 2.1 h | same, `extrapolation` | ESTIMATE |
| Preflight re-imaging on CPU, Charville, 100 studies | NumPy reference 2.98 s; Numba with 8 threads 3.26 s per study; the two agree exactly, including where NaNs fall | `cpu-ladder.json` `rows` | MEASURED |
| R-14 baseline: one full PHReg fit for each of the 3,780 Charville patterns, 8 cores | 67.1 s | PR #64 (open); its results JSON is not committed yet | MEASURED per #64 (not verifiable on `integration`) |
| R-15 CPU score screen, same work as the GPU | n/a | R-15 | PENDING |
| GPU re-imaging and GPU screen | n/a | Modal run | PENDING |

The rung-0 row and the reshaped row cover different scopes, so their ratio is not quoted as a speed-up (`lab_today.comparison_scope`). R-14 fits a full model per candidate, whereas the R-15 screen computes a score test. The R-14 comparison is therefore a workflow comparison, not an equal-work speed-up (PRD principle 5).

## 4. Scientific value, including negative results

| Result | Source | Label |
|---|---|---|
| The textbook formula overstates power at 5 of 7 patient counts. At 200 patients it says 80%; simulation on real tissue gives 32% (95% CI 20 to 45%) | `bench/results/vs-schoenfeld.json` | MEASURED (60 studies per point) |
| Charville, 630 cell-mix patterns: the best pattern has maxT p = 0.13 (9,999 resamples) and omnibus p = 0.14. Chance explains it | `audit/results/charville.json` | MEASURED, negative |
| Charville, 3,780 patterns with protein channels: the top candidate moves to PD1 on macrophages within 100 µm of tumour, z = 3.96. Adjusted p = 0.098 and omnibus p = 0.17, so it is not significant. It stays rank 1 in 158 of 159 leave-one-patient-out drops | `results/charville-v2.json` (#47) | MEASURED, negative |
| On a loaded shared 8-core VM, Numba all-core was no faster than NumPy (3.26 vs 2.98 s per study) | `bench/results/cpu-ladder.json` | MEASURED, negative |
| The R-16 rediscovery candidate is fixed before any search | #61 | TESTED (`audit/test_rediscovery_definition.py`) |
