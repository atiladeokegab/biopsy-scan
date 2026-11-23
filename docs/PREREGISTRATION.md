# Preregistration: Prospective Analysis Plan for Final Runs (`stats-9`, `upmc-9`, and Scaling)

**Baseline commit:** `5e5dab3` (on main, 2026-11-22).  
**Prospective scope vs. pre-run protocols and retrospective documentation:** Prospective claims are bounded strictly to protocols fixed **before each respective run started**. Specifically: (1) previously inspected simulation, benchmark, and real-data outputs committed before `5e5dab3` (`atlas/results/production/`, `bench/results/*`, `stats/results/hard-negative-f458ad1.json`, `results/charville-*.json`, and the historical one-time R-16 run in `results/upmc-rediscovery.json`) are **retrospective methods documentation**, not newly pre-registered endpoints; (2) the `GO-FOR-9` performance runs launched at `13:06Z` prior to this revision follow their already-declared paired protocol (including three lab one-vs-eight-worker scaling pairs executed inside one job on the same host, with fair lab/fused repeats running separately); and (3) newly scheduled statistical/replication and post-merge scaling runs (`stats-9`, `upmc-9`, and subsequent scaling jobs) follow the protocols fixed in this document.  
**Data status and labels:** Public survival outcomes in both cohorts (Wu et al., *Nature Biomedical Engineering* 2022, Zenodo `13179600`, CC BY 4.0) were previously published and inspected; therefore real-cohort results carry **MEASURED (exploratory, already-seen outcomes)** or **UNMEASURED** when underpowered, model-generated simulations carry **SYNTHETIC** (outcomes) and **ESTIMATE** (rates/intervals), closed-form comparisons carry **ANALYTIC**, and invariant checks in code carry **TESTED**. No clinical claim is made.

---

## 1. Null Method (`stats/screen.py`)

New `stats-9` and `upmc-9` screens use `NULL_METHOD["name"] = "clinical-null parametric bootstrap"` implemented in `stats/screen.py` (`screen`, `screen_batch`, `screen_studies`, `_bootstrap`, `_null_draws`, `_phreg`, `_newton`).

1. **Observed-data null fit (`screen` -> `_groups` -> `_phreg`):** For each distinct complete-case patient mask across candidate columns (`keep_all`), after verifying via `_newton` that the observed clinical-null likelihood has a finite maximum, `statsmodels.duration.hazard_regression.PHReg` (`ties="breslow"`, stratified by coverslip group) fits the observed clinical-null model on `C` alone. Zero-covariate mode (`C = None`, e.g. Charville) fixes linear predictor `eta = 0`.
2. **Parametric bootstrap generating model (`_null_draws` -> `_fit_or_raise`):** On the eligible base cohort (`base`: patients with finite nonnegative `time`, `event in {0, 1}`, finite `strata`, and complete `C`), zero-span covariates in `C` are dropped and `C` is mean-centered. Both the event model (`e`) and censoring model (`1 - e`) are fitted by **batched Newton** (`_newton`), which raises if any generating likelihood is monotone (`no finite maximum`).
3. **Continuous cumulative baseline hazard interpolation (`_breslow_steps`, `_draw`):** Ordered stratum jump times `0 = t_0 < t_1 < ... < t_K` and cumulative Breslow baseline hazards `0 = H_0 < H_1 < ... < H_K` are linearly interpolated between Breslow steps (`_draw`, avoiding artificial step ties). With exponential draws `E_i ~ Exp(1)`, target hazard `E_i / exp(eta_i)` is inverted piecewise-linearly up to `H_K` (`+inf` beyond `H_K`). Censoring times are simulated analogously under the censoring Cox fit and truncated at each stratum's maximum observed follow-up (`end`). Observed replicate time and event are `(min(T, Cens), T <= Cens)`.
4. **Per-resample, per-mask refit (`_bootstrap`, `_group_scores`):** For each bootstrap replicate `b in {0, ..., B - 1}` and each distinct complete-case mask group, the clinical-null model on `C` is refitted to `(tb, eb)` by batched Newton (`_newton`), yielding martingale residuals `m*_b`, score numerator `U*_b`, nuisance-adjusted variance `V*_b`, studentised score `z*_b`, and `maxT*_b`.
5. **Independent streams & nonconvergence conservation:** Replicate `b` draws from `np.random.SeedSequence(seed).spawn(B)[b]`, yielding identical draws across feature chunks and CPU/GPU backends. Replicates whose batched Newton refit fails convergence (`nonconverged`) emit `NaN` for that mask's statistics (`NaN` propagates into `null_maxT`) and conservatively count as exceeding observed statistics in both global `p` and feature-wise `adjusted_p` (`_exceed`). Zero-event resamples are retained cleanly with `U = V = z = 0`.
6. **Assumptions of the clinical null:**
   - Proportional hazards in `C` within coverslip strata;
   - Censoring independent of candidate feature `X_j` given `(C, stratum)`;
   - Event and censoring times conditionally independent given `(C, stratum)`;
   - Global null given clinical covariates (`beta_X = 0` conditional on `C`).

*(Retrospective methods note: earlier exploratory R-4/R-9 audits in `stats/reference.py` and `results/charville-v2.json` used within-stratum outcome permutations with permutation variance rather than this parametric clinical-null bootstrap.)*

---

## 2. Studentised Score Definition (`stats/screen.py`)

For each candidate feature `j in {0, ..., m - 1}`:

1. **Per-feature analysis set (`_inputs`, `_groups`):** Eligible patients with finite `X_{ij}` (`NaN` preserved as missing per feature, never zero-imputed), finite nonnegative `time_i`, `event_i in {0, 1}`, finite `strata_i`, and complete clinical covariates `C_i` (when supplied).
2. **Stratified Cox risk operator & score numerator (`_risk`, `_stats`):** At clinical-null risk weights `w_i = exp(C_i' beta_hat_C)` within stratum, risk-set expectation operator `M` produces expected cumulative hazard weights `a_i` and martingale residuals `m_i = event_i - a_i`, giving score numerator `U_j = (X' m)_j` on within-stratum centered columns (`_centre_within_strata`).
3. **Schur-complement nuisance-adjusted variance (`_finish`, `_solve`):** With observed Breslow Cox information `[I_jj, I_jC; I_Cj, I_CC]` centered within risk sets (`_cross`), the nuisance-adjusted score variance is:
   `V_j = I_jj - I_jC * I_CC^{-1} * I_Cj`.
   - **Pseudoinverse fallback (`_solve`):** Linear solves `A x = b` attempt `linalg.solve`; if `LinAlgError` occurs or any entry is non-finite (e.g., singular `I_CC` in zero-event resamples), `_solve` falls back to Hermitian Moore-Penrose pseudoinverse `np.linalg.pinv(A, hermitian=True) @ b`.
   - **Zero-covariate mode (`C` is `None`):** No clinical nuisance parameter exists, so `V_j = I_jj` (Breslow Cox information, distinct from the legacy within-stratum permutation variance `sum_s S_{xx,s} S_{rr,s} / (n_s - 1)` in `stats/reference.py:score`).
   - **Relative numerical floor (`_finish`):** A candidate is informative iff `V_j > 1e-10 * I_xx_j`; otherwise `V_j = 0.0` and `z_j = 0.0`.
4. **Studentised score & common `maxT` multiplicity tolerance (`screen`):**
   - Studentised statistic: `z_j = U_j / sqrt(V_j)` where `V_j > 1e-10 * I_xx_j`, else `0.0`.
   - Family-wise maximum: `maxT_obs = max_j |z_j|`.
   - Both global `p` and feature-wise `adjusted_p_j` share one common float64 roundoff tolerance anchored on `maxT_obs`:
     `tol = 16 * eps_float64 * max(1.0, maxT_obs)`.
   - `p = (1 + _exceed(null_maxT, maxT_obs - tol)) / (B + 1)`, and `adjusted_p_j = (1 + _exceed(null_maxT, |z_j| - tol)) / (B + 1)` where `_exceed` counts finite `null_maxT >= target` plus all `NaN` (nonconverged) replicates.

---

## 3. Family-Wise Error Rate (FWER) Target and Pass Rules

- **Global FWER target:** `alpha = 0.05` (two-sided `|z_j|` / `maxT` for discovery screens and simulation design calibration; one-sided directional test at `alpha = 0.05` for locked single-candidate validation).
- **Discovery screen pass rule (`stats/screen.py`, `audit/screen_cohort.py`):** Global screen rejects at FWER `alpha = 0.05` iff `p <= 0.05` (with `B >= 999` resamples unless raised). Any individual candidate is reported positive iff `adjusted_p_j <= 0.05`.
- **Simulation calibration cutoff rule (`stats/reference.py:threshold`, `atlas/runner.py`):** From `calibration` independent null `maxT` draws (`len(maxima) >= 2`), rank `k = ceil((len(maxima) + 1) * (1 - alpha))`. If `k <= len(maxima)` (sufficient valid draws), `threshold` is the `k`-th order statistic (1-indexed) and independent studies reject iff `maxT > threshold`; **if `k > len(maxima)`, `threshold` returns `+inf`** (zero rejections possible). Both empirical null FWER and alternative power carry two-sided 95% Clopper-Pearson intervals (`atlas/runner.py:interval`).
- **Hard-negative clinical confounding calibration check (`stats/hard_negative.py`, PRD A-15.6):** On `n_datasets` (`300` code default, `1000` verification scale) synthetic confounded cohorts (`n = 120`, random stratum assignments `rng.integers(0, 2, n)`, `C = [binary, normal]`, `p = 20` correlated candidates with zero effect conditional on `C`, Weibull event shape `1.3`, exponential censoring depending on `C`, `B = 99`, `alpha = 0.05`, `seed = 2026`):
  - **Pass criterion (A-15.6):** Only the clinical-adjusted (`bootstrap`) screen's rejection rate is required to have its 95% Clopper-Pearson interval contain `alpha = 0.05`; both unadjusted (`screen(..., clinical=None)`) and raw residual shuffle (`raw_residual_shuffle`) run on the same datasets and are reported alongside as negative controls (`unadjusted` demonstrates false-positive failure when `C` is ignored).
  - **Retrospective baseline reference (`stats/results/hard-negative-f458ad1.json`, SYNTHETIC):** `bootstrap`: `60 / 1000 = 0.060` (CP 95% `[0.0461, 0.0766]`, contains `0.05`); `unadjusted`: `997 / 1000 = 0.997`; `raw_residual_shuffle`: `1 / 1000 = 0.001`.
- **Power gate (`audit/rediscovery.py`):** Locked single-candidate confirmation requires pre-outcome simulated conditional power `>= MIN_POWER = 0.50` at `EFFECT_HR_PER_SD = 2.0` (`POWER_SIMS = 400`, `POWER_SEED = 20261121`); below `0.50`, status is locked to `UNMEASURED` (`underpowered`).

---

## 4. Power Grid and Outcome Simulator Specifications

### 4.1 Pooled-Scale Outcome & Stratum-Censoring Model (`atlas/simulation.py`)

For each empirical source `(X_src, time_src, event_src, strata_src)`:
- **Planted feature:** First finite column of the whole-tissue reference matrix with positive within-stratum variation (`plant_feature = 0` on Charville), standardized with source `mean` and population SD (`ddof=0`), held invariant across simulated sample sizes and imaging designs.
- **One pooled Weibull scale per effect (`fit_model`):** Fixed `shape = 1.5` and effect `hr = 2.0` per source SD (`hr = 1.0` for null calibration/evaluation). **One pooled baseline `log_scale`** per effect (`brentq`) is solved across all source strata jointly so expected event probability under **stratum-specific** reverse Kaplan-Meier censoring hits `target` (`event.mean()` on source, e.g., `38 / 159 = 0.23899371...` on Charville).
- **Stratum-specific reverse Kaplan-Meier censoring:** Computed per stratum with events processed before censors at tied times; unresolved tail mass sits at an administrative horizon equal to that stratum's maximum observed follow-up (`Charville`: `[1562.0, 1675.0, 5812.0, 4821.0]` days for strata `0..3`, verified in `bench/power_at_scale.py:FROZEN_MODEL`).

### 4.2 Design Grids & Conditional Source Rules (`atlas/__main__.py`, `bench/`)

All power curves remain strictly **conditional on their own source cohorts** (`Charville` vs `UPMC`); curves across cohorts are never treated as directly comparable because source tissue variety, censoring distributions, and feature grammars differ.

| Suite | Source Cohort | Grid Axes (`P`, `k`, `fov_um`, `eps`) | Budget / Method | Status & Scope |
|---|---|---|---|---|
| Atlas Charville (`atlas/results/production/charville/`) | Charville (`159` cases, `286` regions) | `P in {80, 159, 318, 477}`, `k in {1, 2}`, `fov in {200, 300, 400, 500, 600, full}`, `eps = 0.1` (`48` points) | `DEFAULT_BUDGET`: `cal=10000`, `null=10000`, `alt=1000`, `perm=999`, `bank_draws=64`, `nested=20`, `direct=20` on 3 points | **Retrospective** committed grid (**SYNTHETIC** / **ESTIMATE**); reference template for new scaling runs |
| Atlas UPMC (`atlas/results/production/upmc/`) | UPMC verified cases (`50` total source) | `P in {25, 50, 100, 150}`, `k in {1, 2, 4, 8}`, `fov in {200, 400, 600, 800, 1000, full}`, `eps = 0.1` (`96` points) | Same `DEFAULT_BUDGET`, `bank_draws = 64` | **Retrospective** descriptive grid only (**SYNTHETIC** / **ESTIMATE**) |
| `vs-schoenfeld` (`bench/vs_schoenfeld.py`) | Charville (`k=2`, `fov=400`, `eps=0.1`, `hr=2.0`, `m=630`) | `P in {40, 80, 120, 160, 200, 240, 300}` (`7` points) | `cal=200`, `null=100`, `alt=300` vs Schoenfeld/Hsieh-Lavori Bonferroni (`alpha/630`) | **Retrospective** benchmark (**ESTIMATE** vs **ANALYTIC**) |
| `power-at-scale` (`bench/power_at_scale.py`) | Charville (`k=2`, `fov=400`, `eps=0.1`) | `[40..300]` (`vs-schoenfeld`) + `{400, 500, 600, 700, 800, 1000}` | `200 / 100 / 300`; `n_80` = first linear interpolation crossing of `0.80` (no extrapolation) | **Retrospective** (`n_80 ~ 524` **ESTIMATE**) + scaling anchor for new GPU scaling runs |
| `hn-power` (`bench/hn_power.py`) | UPMC **discovery only** (`34` cases, zero outcome read) | `P in {50, 100, 200, 300, 400, 600}` (`k=2`, `fov=400`, `eps=0.1`, common horizon tuned to `0.23899371...`) | `cal=200`, `null=100`, `alt=200` Modal / `300` spec | **Retrospective** planning curve (**ESTIMATE** conditional on `34` discovery cases) |

---

## 5. Cohorts, Endpoints, Identity Rules, Strata, and Exclusions

### 5.1 Charville Cohort (`core/cohorts.py`, `docs/CONTRACT.md`, `results/charville-v2.json`)

- **Source & panel:** CODEX spatial proteomics (`PIXEL_UM = 0.377` um/px), `15` cell types (`T = 7` focal tumor types, `K = 15` total types), `9` protein channels (`PD1`, `PDL1`, `Ki67`, `GranzymeB`, `HLA-DR`, `aSMA`, `CollagenIV`, `Vimentin`, `Podoplanin`) + `SIZE` (cohort-wide z-score after pre-normalization).
- **Grammar (`core/grammar.py`):** `v1` = `7 × 15 × 3 radii (25, 50, 100 um) × 2 stats (fraction, enrichment) = 630` features (hash `772f6a845266470456af96f92d0573cc1bec72432ad74d789439baff066c2156`); `v2` appends `7 × 15 × 3 × 10 = 3,150` channel means = **`3,780`** total features (hash `26a745f831ef71ec2c8ae5bf65f3668d29fd0f53ab97699cf52ac0cc13f11564`), keeping `11,880` missing (`NaN`) cells un-imputed.
- **Verified identity & aggregate exclusions (`core/cohorts.py`):** `patient_id` = first underscore-delimited field of `sample_label_visualizer`. **Aggregate exclusions:** `3` source cases are excluded (`2` lacking valid `(surgery_date, last_contact_date)` dates; `1` spanning two coverslips `c002`/`c003`), leaving **`159` usable patients**, **`286` regions**, **`619,645` cells**, and **`38` observed events** (**MEASURED**).
- **Strata:** Coverslip batch groups `c001..c004` mapped to strata `0..3` (`40`, `33`, `48`, `38` patients; `0` spanning patients).
- **Event & time definitions:** `time_days = (last_contact_date - surgery_date).days` (`%m/%d/%Y`; `NaN` if unparseable); `event = 1.0` when `charville_labels.csv` vital-status column indicates an all-cause endpoint event (`0.0` otherwise when `time_days` is finite).
- **Clinical covariates:** `clinical = None` (zero-covariate mode in `stats/screen.py`; labels contain no clinical covariates).

### 5.2 UPMC Cohort (`core/cohorts.py`, `core/heldout.py`, `core/upmc_locked.json`, `audit/rediscovery.py`)

- **Source & panel:** UPMC CODEX cohort (`0.377` um/px; Wu et al., *Nature Biomedical Engineering* 2022), `17` cell types (`UPMC_TYPES`, `T = 6` focal tumor types, `K = 17` total), **`612`** `v1` grammar features (hash `adc93e69df4280df4ebc40432d4b3ca84ffb4d8e83513ae1ce883feff981915a`). `PD1`, `PDL1`, and `GranzymeB` are absent from the UPMC antibody panel, rendering Charville `v2` signatures using those markers structurally `NOT TESTABLE`.
- **Verified identity & aggregate exclusions (`identity = "case"`):** Patient-level inference and `upmc-9` runs strictly use verified explicit case numbers (`sample_label_visualizer`, non-`NA`) whose regions agree on clinical fields (`UPMC_TUPLE`). **Aggregate exclusions:** `86` discovery-side `NA` regions, cases with conflicting tuple rows, and `1` unmapped NA-tuple group spanning coverslip groups (`6` regions across `c001, c004–c007`) are excluded from patient-level inference (`core/upmc_locked.json`).
- **Strata & covariates:** Coverslip groups `0 = c001–c003`, `1 = c004–c007`. R-15 clinical covariate matrix `C = [hpvstatus]` (`0`, `1`, or `NaN` for missing `NA`; mismatched region `hpvstatus` raises `ValueError`).
- **Event & time definitions:** `time_days = float(survival_day)`; `event = float(survival_status)` (`1` = all-cause event, `0` = censored).
- **Stratified Locked Split (`core/heldout.py`, `core/upmc_locked.json`):** `SEED = 20261121`, `FRACTION = 0.3`, stratified on `survival_status` (manifest SHA-256 `05b38abae377177a097f2db0cd42188178c275a338aa44ec79c1be8ecd3969e3`), verified against `upmc_labels.csv` SHA-256.
  - **Discovery verified cases (**MEASURED**):** `34` patients, `125` regions, `6` events (disjoint caches `upmc_{cells,channels}.discovery.npz`, `LockedPatientError` enforced).
  - **Locked verified cases (**MEASURED**):** `16` patients, `63` verified case regions (`97` total locked regions across `23` locked patients (`24` split units)), `4` verified-case events.
- **Single-opening guard (`audit/rediscovery.py:main`):** Enforces independent validation gates: (1) `--open-locked` flag, (2) `--events` matching `4`, (3) `--out` does not already exist, (4) `RULE_SHA256 == PREREG_SHA256`, (5) **separately**, clean committed git HEAD via `_code_state()` (no modified tracked files), (6) `_split_preflight()` verifying split API and approved manifest SHA-256, (7) absence of atomic marker file `opening_record(rule_sha, manifest_sha)`, (8) `MEASURED` discovery rank, and (9) atomic file reservation + marker write prior to reading locked outcomes.

---

## 6. Frozen and Pre-Specified Random Seeds

| Component | Module / Script | Seed | Stream / Derivation Rule |
|---|---|---|---|
| UPMC stratified holdout split | `core/heldout.py`, `core/upmc_locked.json` | `20261121` | `30%` stratified split over linked patient units |
| R-16 locked conditional power check | `audit/rediscovery.py` (`POWER_SEED`) | `20261121` | `400`-trial Type-II censoring power gate (`MIN_POWER = 0.50`) |
| Atlas production & new scaling runs | `atlas/__main__.py`, `atlas/runner.py` | `20261121` | `SeedSequence([20261121, int(design_id[:8], 16), stage, index]).spawn(4)` for `[image, survival, censor, permutation]` (`stage`: `101`=cal, `202`=null, `303`=alt) |
| Atlas finite acquisition bank | `atlas/bank.py`, `atlas/__main__.py` | `20261121` (`bank_draws=64`) | SHA-256 key per `(regions_per_patient, fov_um, eps)`; `reimage_ref` draw order: patient `integers` -> region `choice` -> crop `random(2)` + mis-call `random(n_cells)` |
| `vs-schoenfeld`, `power-at-scale`, `hn-power` | `bench/vs_schoenfeld.py`, `power_at_scale.py`, `hn_power.py` | `20261121` | Shared Atlas seed scheme (`design_id` omits `backend`) |
| Event-count calibration & plasmode checks | `bench/event_count_calibration.py`, `bench/real_check.py` | `20261121` | Paired arm streams via `SeedSequence([seed, n, stage, i]).spawn(3)` |
| Hard-negative confounding calibration | `stats/hard_negative.py` (`run`) | `2026` | Synthetic cohort generator (`d = 0..n_datasets-1`) + `B = 99` bootstrap `seed = d` |
| Real-cohort discovery screens (`stats-9` / R-18) | `audit/screen_cohort.py`, `audit/charville_v2.py` | `0` (`--seed 0`) | Chunk-invariant `SeedSequence(seed).spawn(B)[b]` over `B >= 999` replicates |

---

## 7. What Counts as Replication

Prospective evaluation criteria for **new runs** (`stats-9`, `upmc-9`, new scaling) are separated strictly from historical artifacts and cross-method comparisons:

1. **Historical one-time R-16 locked UPMC run vs. new reproducibility checks:**
   - **Historical one-time opening (retrospective record):** Executed once at `32a69fa` (`results/upmc-rediscovery.json`) for pre-registered candidate `"Tumor (Podo+)|Stromal / Fibroblast|25um|fraction"` (`PREREG_SHA256 = a70424447055052d19610332a478fb46da68d87a9f2f30b350cc3c14a4bada1e`), direction `HR > 1`, one-sided Wald `p < 0.05`, and power gate `>= 0.50`. Recorded **`UNMEASURED`** (`18% < 50%` conditional power at `4` events; HR `1.82` `[0.47, 6.97]`, one-sided `p = 0.193`, C-index `0.548`, discovery rank `366/612`). The once-only opening record prohibits re-opening that rule on the same manifest.
   - **New `upmc-9` / `stats-9` runs:** Any newly scheduled UPMC or Charville screen (`stats.screen` with HPV adjustment `C = [hpvstatus]` on UPMC discovery, or `C = None` on Charville, `B >= 999`, `seed = 0`) rejects the global null in the exploratory screen only if `maxT` bootstrap `p <= 0.05` (with candidate claims requiring `adjusted_p <= 0.05`). Because the cohorts were already inspected prior to this plan, this threshold cannot establish confirmatory replication. Cross-panel targets whose markers are missing on UPMC (e.g., `PD1` in Charville's retrospective exploratory top hit `Tumor 5|Macrophage|100um|PD1`, `z = 3.957`, permutation `p = 0.0981` in `results/charville-v2.json`) remain **`NOT TESTABLE`**, never failed replications.

2. **Exact within-method replay vs. cross-method decision agreement (distinct claims):**
   - **Exact within-method replay / backend parity (required for scaling & GPU checks):** Within the **same statistical procedure**, CPU/NumPy, Numba, CUDA GPU, multi-GPU fleet shards, and fresh-container reruns must produce bitwise identical re-imaged feature matrices (`X` SHA-256), `z` and `maxT` matching within `Z_TOL = 1e-9` relative (`Z_TOL`), and identical per-study accept/reject decisions (`100%` agreement within procedure).
   - **Cross-method comparison & paired method deltas (not replay equivalence):** Comparing distinct statistical procedures or approximations on shared studies measures method divergence rather than replay failure:
     - *Shared design cutoff vs. nested per-study permutation (`atlas/runner.py`):* Quantified as **`nested_shared_minus_conditional = mean(maxT_reject - conditional_reject)`** (shared minus conditional) with conservative >=95% `paired_interval` bounds; intervals containing zero indicate no detected difference on `n = 20` studies, not equivalence. On `1,000`-study retrospective arms (`bench/results/modal-design-20261121T194458Z.json`, `matched-e2e` `89/100`, `fair` `362/400` cross-method decision agreement in `bench/results/sweep-repeats.json`), shared-threshold and per-study permutation are distinct procedures with different power (`15.3%` vs `20.3%`, **ESTIMATE**).
     - *Finite acquisition bank vs. fresh direct imaging (`atlas/runner.py`):* Quantified as `bank_minus_direct_rejection` under the shared bank-derived cutoff with paired >=95% bounds.
     - *Weibull vs. Kaplan-Meier plasmode sensitivity (`bench/real_check.py`):* Paired `sim - plasmode` power differences across `{50, 100, 159}` full-tissue studies (`+1.75`, `-0.70`, `-1.30` pp; retrospective **ESTIMATE** bounds all crossing zero).

3. **Fresh-container compute & scaling replication (`bench/repeats.py`, `bench/sweep_repeats.py`):**
   - For the fair repeats and other separately specified runs, benchmarks require `3` independent fresh containers matching the declared container class (`NVIDIA A100-SXM4-40GB` where pinned), passing study-manifest completeness, bitwise feature SHA-256 parity, and `1e-9` `maxT` parity within procedure, reporting median and `min-max` spread as descriptive only. **Explicit same-host exception:** the paired lab scaling check started at `13:06Z` executes three lab one-vs-eight-worker scaling pairs inside one job on the same host under its declared paired protocol rather than across separate fresh containers (fair lab/fused repeats run separately across fresh containers).
