# Preflight against the four judging criteria

Every number below is copied from a committed results file. Where that file is on an open PR, the row names the PR. Labels:
- **MEASURED**: a committed results file from a real run.
- **TESTED**: a passing test in `make smoke`.
- **ESTIMATE**: arithmetic on measured rows, not a run.
- **PENDING**: not built or not run yet. A row whose evidence sits on an open PR names the PR.

All real-data results are exploratory: the outcomes have already been seen.

## 1. Correctness and assumptions

| Claim | Evidence | Test | Label |
|---|---|---|---|
| With no crop and no mis-calls, re-imaging reproduces the full-tissue truth exactly | `core/` | `core/test_spine.py::test_no_crop_eps0_reproduces_truth_exactly` | TESTED |
| The CPU and GPU (CUDA simulator) backends count the same neighbours as direct geometry, on unseen irregular regions | `reimage/` | `tests/test_reimage_contract.py`, `reimage/test_backends.py::test_region_matches_reference` | TESTED |
| The score equals statsmodels PHReg (Breslow ties); its variance equals exhaustive permutation enumeration | `stats/reference.py` | `stats/test_reference.py::test_score_matches_phreg_breslow_with_ties`, `tests/test_score_contract.py::test_variance_is_exhaustive_permutation_variance` | TESTED |
| A planted signal is found; batch confounding gives no evidence (hard negative) | `stats/reference.py` | `tests/test_score_contract.py::test_planted_ordered_signal_is_found`, `::test_batch_confounding_is_a_hard_negative` | TESTED |
| When nothing is there, the false-alarm rate holds at the promised 5%: 28 of 700 null studies, 4.0% (95% CI 2.7 to 5.7%). This pools the 7 patient counts at 100 null studies each; the CI is Clopper-Pearson | `bench/results/vs-schoenfeld-300.json` (#53) | `bench/test_vs_schoenfeld.py` | ESTIMATE |
| Two GPU shards give bit-identical results to one device | `reimage/fleet.py` (#34) | `reimage/test_fleet.py::test_two_fake_shards_on_cuda_simulator_are_bit_identical` | TESTED on the CUDA simulator; on a real A100, see the GPU parity row below |

### Assumptions, and where each is tested

| Assumption | Why it's reasonable | Test or result that checks it | Label |
|---|---|---|---|
| A simulated study resamples real Charville patients: patients with replacement, up to two regions each without replacement, from their real cells | A study is a sample from the same population the cohort came from. Re-imaging keeps each patient's real tissue | `core/test_spine.py::test_reimage_design_draws_are_seeded_and_change_x`; `atlas/test_runner.py::test_patient_rows_match_shared_core_draws_and_stage_streams_are_disjoint` (#45) | TESTED (atlas test on #45, open) |
| The planted effect is HR 2 per SD of the feature | The formula and the simulation get the same effect, so the comparison is like for like; at 200 patients the formula gives the textbook 80% | `bench/results/vs-schoenfeld-300.json` `hr_per_sd`; `bench/test_vs_schoenfeld.py::test_formula_hits_the_textbook_80_percent_point`; `atlas/test_simulation.py::test_source_sd_frozen_and_hr_quantile_identity` (#45) | TESTED (atlas test on #45, open) |
| Real censoring and coverslip strata: censoring follows each stratum's reverse Kaplan-Meier from real follow-up; the unresolved tail sits at the stratum's last follow-up, a declared administrative horizon | Simulated studies lose patients to follow-up the way the cohort did. Strata are safe because 0 of 159 patients span two coverslips | `atlas/test_simulation.py::test_reverse_km_event_first_ties_and_unresolved_tail`, `::test_tail_mass_becomes_declared_administrative_horizon` (#45); `results/charville-v2.json` `leakage_audit` | TESTED (#45, open); strata MEASURED |
| The score test matches statsmodels PHReg | PHReg is an independent, standard Cox implementation; the tie convention is the same on both sides (Breslow) | `stats/test_reference.py::test_score_matches_phreg_breslow_with_ties` | TESTED |
| maxT holds family-wise false alarms at 5% | 28 of 700 null studies rejected, 4.0% (95% CI 2.7 to 5.7%), pooled over 7 patient counts | `bench/results/vs-schoenfeld-300.json` (#53) | ESTIMATE |
| R-15 bootstrap maxT holds family-wise false alarms at 5% after clinical adjustment | On a hard negative (candidates correlated with C, hazard and censoring depending on C, no candidate effect given C), 0.060 (60 of 1,000) [0.046, 0.077] Clopper-Pearson 95%, SYNTHETIC, stats/results/hard-negative-f458ad1.json, B = 99, rerun on integration f458ad1 after #80. The pre-#80 run's 0.051 and the earlier step-function draws' 0.074 (0.059 to 0.092) are historical, in `stats/results/hard-negative.json`. The unadjusted screen rejects 0.997, so the check goes red on a method that ignores C | `stats/results/hard-negative-f458ad1.json` (#99) | SYNTHETIC |
| A missing feature is missing, not zero | Risk sets are recomputed per feature on complete cases; Charville v2 keeps 11,880 NaN cells | `stats/test_reference.py::test_missing_features_use_complete_case_risk_sets_and_batch` | TESTED |
| UPMC's locked held-out split: no patient on both sides, and discovery never sees locked patients | Validation patients are fixed before any search, so a later score on them is honest | `core/test_holdout.py::test_no_patient_is_on_both_sides_under_any_identity`; `tests/test_no_leak.py::test_changing_locked_patients_leaves_discovery_outputs_byte_identical` (#62) | TESTED (#62, open) |
| GPU kernels agree with the CPU reference | The same backend, fleet and expression tests run on real hardware: 39 passed, 0 failed, 1 skipped (that check needs CUDA absent), NVIDIA A100-SXM4-40GB, from clean commit b7f2992 | `bench/results/modal-a100-20261121T173127Z.json` (#75) | MEASURED (#75, open) |
| One null threshold per design gives the same answer as a fresh shuffle for every study | n/a | build-modal | PENDING |
| Clinical covariates in the C-003 screen: UPMC adjusts for HPV status only; Charville uses none | The labels carry no stage, margins or age; both cohorts use coverslip strata | PRD R-15 | PENDING (R-15) |
| Rung 0 reuses resampled outcomes only as a timing workload | No p-value is claimed from it | `bench/results/cpu-ladder.json` `lab_today.inference_note` | MEASURED |

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
| R-14 workflow baseline: one full PHReg fit for each of the 3,780 Charville patterns (a workflow comparison, not equal work) | 63.84 s on 8 workers; 298.30 s on 1 (shared VM, 1-minute load average 49 to 56) | `bench/results/impl-a-charville-amd-epyc-7b13.json` (#64) | MEASURED |
| R-15 CPU score screen, same work as the GPU | n/a | R-15 | PENDING |
| GPU re-imaging and GPU screen | n/a | Modal run | PENDING |

The rung-0 row and the reshaped row cover different scopes, so their ratio is not quoted as a speed-up (`lab_today.comparison_scope`). R-14 fits a full model per candidate, whereas the R-15 screen computes a score test. The R-14 comparison is therefore a workflow comparison, not an equal-work speed-up (PRD principle 5).

## 4. Scientific value, including negative results

| Result | Source | Label |
|---|---|---|
| The textbook formula overstates power at 6 of 7 patient counts and understates it at 1. At 200 patients it says 80%; simulation on real tissue gives 27% (95% CI 22 to 32%)[^60] | `bench/results/vs-schoenfeld-300.json` (#53) | ESTIMATE (300 studies per point) |
| Charville joint screen (v2): v1's 630 cell-mix patterns plus 3,150 more with protein channels, on the same 159 patients and 38 events. The top candidate is PD1 on macrophages within 100 µm of Tumor 5 cells (`Tumor 5\|Macrophage\|100um\|PD1`), z = 3.96. maxT p = 0.098 over 3,780 patterns (9,999 resamples) and omnibus p = 0.17: not significant after correction (maxT p = 0.098); consistent with chance. It stays rank 1 in 158 of 159 leave-one-patient-out drops | `results/charville-v2.json` (#47) | MEASURED, exploratory, negative |
| On a loaded shared 8-core VM, Numba all-core was no faster than NumPy (3.26 vs 2.98 s per study) | `bench/results/cpu-ladder.json` | MEASURED, negative |
| The R-16 rediscovery candidate is fixed before any search | #61 | TESTED (`audit/test_rediscovery_definition.py`) |
| R-16 locked test on UPMC: the frozen rule's feature (`Tumor (Podo+)\|Stromal / Fibroblast\|25um\|fraction`) gives HR per SD 1.82 [0.47, 6.97], one-sided p 0.193, C-index 0.548, so the locked test does not pass. It is inconclusive: underpowered, conditional power 18% < 50% with 4 design events. Discovery rank 366 of 612 (MEASURED, discovery patients only). The Charville top hit is not testable on UPMC (PD1 absent). Exploratory: outcomes already seen (Punovuori et al., Cell 2024); internal held-out split | `results/upmc-rediscovery.json` (#109) | UNMEASURED (underpowered) |

[^60]: The earlier 60-study run (`bench/results/vs-schoenfeld.json`) gave 32% (95% CI 20 to 45%) at 200 patients.
