# IDEA v6: Can your study find it? (BUILD PLAN, for approval 1)

**The current build plan is the doc "Build plan: Can your study find it? (v6)": https://claude.ai/artifact/2DkGkU62tfth2m11KfVUWz.** It was written after Fable's v4 regrade (6/4/3/5; `blind-review/fable-v4.md`, evidence in `fable-v4-evidence/`).

What changed in v6:
- The three CPU shortcuts Fable built (nesting, region pool, analytic mis-calls) are now the reshaping we claim.
- The speedup ladder separates what the algorithm buys from what the hardware buys.
- The exact engine has two jobs: the design atlas, and a within-region label-shuffle null that answers "composition or arrangement?".
- There is a pivot gate at 11:45.

The v5 and v4 text below is history.

# IDEA v5 (superseded by v6)

## v5: changes after Atlas's v4 regrade (5/5/5/7; `blind-review/atlas-v4.md`); the lead accepted all

The v4 text below stands, except where this table overrides it. The full integrated plan is in the shared doc "Can your study find it? Full plan v5". Fable's v4 regrade is still pending; fold it in when it lands.

| # | Atlas v4 finding | v5 action |
|---|---|---|
| 1 | Event yield cannot resolve patient identity; "all 79 primary" reverses the safeguard | Charville (162 case-numbered patients, 38 deaths) is the only real-data cohort with patient-level inference. UPMC real-data results are descriptive, with sensitivity bounds for tuple-matched records. UPMC tissue is still a source for the atlas, which uses verified patients only for the regions-per-patient axis. Drop the "20-event rule" as a validity claim |
| 2 | The shared threshold is not a permutation guarantee | Adopt an explicit **unconditional design threshold**, estimated from independent noisy null studies per design point. Calibrate it on separate runs at the design extremes, with tail intervals. Check it against conditional permutations on a subset. Never call it exact per-study control. Null studies are re-imaged too, and their cost is counted |
| 3 | "Every study must be recounted" is too strong | Drop irreducibility. Build the best adequate CPU simulator: a cached bank of region acquisitions, sparse relabel updates, region-first batching. Show an accuracy–runtime frontier against direct recounting. The GPU must win against the best adequate method. One label draw per cell per study, focal cells relabelled too |
| 4 | Logical gathers are not a bandwidth bound | Measure DRAM bytes, achieved throughput, cold start and amortised times on named hardware. 50–200× is an untested estimate, not the promise |
| 5 | The "truth" is a reference measurement and assay-dependent | Call it the reference layer. Define the estimand. Separate sampling, measurement and boundary bias. Specify confusion matrices, missing focal types, region sampling without replacement (k capped at what a patient has), and field dimensions. Promote source bootstrap plus correlated-error and non-PH stress tests to P1. Effect-specific guidance only, never "regions beat patients" as a universal |
| 6 | Null and omnibus need restraint | State the joint exchangeability of the observed outcome within strata, and non-informative censoring. Planted controls draw censoring from the KM estimate of each cohort's censoring distribution. Calibrate the choice among maxT, sum of z² and higher criticism jointly. A significant omnibus is "exploratory conditional association", not diffuse biology or an effect size |
| 7 | Modest novelty | Add a prior-work capability table; remove "no tool answers". Cut the step-down and leakage-publication claims |

---

# IDEA v4 (superseded where the v5 table says so)

**Event:** SoTA Scientific Parallelism Hackathon, King's Cross. Sat 21 Nov 11:00 to Sun 22 Nov 18:00 (London time = UTC).

**Project tag / repo:** `sota-parallelism-nov21` / `atiladeokegab/sota-parallelism-nov21`, created at approval 1.

**Versions:**
- v1: panel search (`IDEA-v1-panel-search.md`)
- v2: exhaustive biomarker search (`IDEA-v2-exhaustive-search.md`)
- v3: power atlas over precomputed features (`IDEA-v3-power-atlas.md`)
- v4: this file

v4 was rebuilt after the blind regrades of v3 (`blind-review/atlas-v3.md`, `blind-review/fable-v3.md`). The v3 scores were:

| Reviewer | Correctness | Reshaping | Speedup | Science |
|---|---|---|---|---|
| Atlas | 6 | 6 | 5 | 7 |
| Fable | 4 | 3 | 3 | 6 |

Fable ran the plan on the real data. It found that the search is about 1,000 patterns, not 300,000. It also found that the nested shuffles collapse to one threshold per design, and that UPMC's coverslips do not separate patients. Once those were corrected, the v3 GPU workload fitted on a laptop.

v4 keeps the sound statistic and the atlas. It moves the GPU to where the compute really is: **re-imaging every simulated study cell by cell from real tissue**. Every add and cut the reviewers asked for is applied below.

## Every correction, applied

| # | Finding (who) | v4 action |
|---|---|---|
| 1 | C is about 1,000, not 300,000 (Fable, measured: 612 UPMC, 1,260 Charville) | **Fix.** Count C exactly at the spine and print it; never quote 300k. The GPU workload no longer depends on C |
| 2 | Nested shuffles are redundant: the max-stat threshold is stable across cohorts (3.86 ± 0.05), and nested and fixed-threshold power agree to 0.005 (Fable) | **Adopt it as reshape 1.** One permutation run per design point, then one matrix-vector product per study. Verify against the nested version on 3 design points |
| 3 | UPMC coverslips do not separate patients: 48 of 50 case numbers span 2–4 coverslips, so LOCO validates nothing. Patients nest in two coverslip groups, c001–003 and c004–007 (Fable) | **Cut LOCO.** Strata are the coverslip group. Stability is shown by leave-one-patient-out influence on top hits. No held-out claim |
| 4 | The confirmatory 48 is the worst subset: 6 deaths, skewed to survivors (Fable) | **Fix.** Primary UPMC analysis is all 79 patients, any-cause death, 27 events, one strata definition. Minimum-event rule: no confirmatory search under 20 events. Cancer-specific death is secondary |
| 5 | Controls fail: HPV is constant within HPV strata, so its loss is guaranteed; coverslip ID is vacuous (both) | **Replace.** The positive controls are semi-synthetic: a planted effect on real features with real censoring. The negative controls are simulated known-null batch artefacts. Measure control operating rates. Drop the word "batch-proof" |
| 6 | The batch-artefact toggle has no artefact (Fable, measured) | **Cut the toggle.** Report "stratification changes little here" as a finding |
| 7 | Novelty: permGPU 2010 proposed this power analysis; Matrix eQTL 2012; tensorQTL 2019; Baker et al. 2023 did spatial-omics power analysis (Fable) | **Cite all four.** Claim the application: power for *survival-outcome discovery searches* under imaging-design choices, batch and censoring. Name how each differs |
| 8 | Speedup is a library ratio; the fused max buys at most 25% (Fable). A 10 s button needs about 96 TFLOP/s in v3 (Atlas) | **Cut the custom GEMM, TF32/FP16 and the fused-max claim.** The GPU hero is the cell-level re-imaging kernel (below). The baseline is our own all-core CPU version of the same kernel |
| 9 | The simulator confuses detection with identification: at HR 5, n=160, power is 0.80 but the planted pattern tops the list only 56% of the time (Fable) | **Report both.** Power, correct-identification rate and false-identification rate, per design point |
| 10 | Region-level ICC is 0.34–0.44 and there is no regions-per-patient axis (Fable). The simulator is narrow (Atlas) | **Add imaging-design axes:** regions per patient, field-of-view size, cell-type mis-call rate. This is the new GPU workload |
| 11 | Exchangeability and strong FWER need stating (Atlas) | **State the conditional null:** censoring, including other-cause death, is independent of features within strata. Give a merge rule and informative stratum and event counts. Single-step maxT controls the global null; step-down claims only under subset pivotality, which is stated |
| 12 | Permutation variance is not Cox information (Atlas) | **Name the statistic:** randomisation-standardised score, z = U / √(Σₛ Sxxₛ·Srrₛ/(nₛ−1)). Validate U against statsmodels PHReg. Validate the variance against exact enumeration on tiny strata. Do not call it a Cox z |
| 13 | "At most 5%" from 1,000 null cohorts is ±1.35 points (Atlas) | **Run 10,000 null studies per calibration point** (cheap after reshape 1): about ±0.43 points. Report binomial intervals |
| 14 | Effects are undefined for continuous candidates, and the simulator has no event calibration (Atlas) | **Define the effect as HR per SD of the true feature.** Drop "prevalence". Tune the Weibull scale and censoring to hit a target event count, and report the achieved events |
| 15 | Pooling cohorts and resampling to 640 adds no diversity (Atlas) | **Build cohort-specific atlases**, with no pooling. Cap n at about 3× the source size, and label extrapolation. Bootstrap the source cohort for uncertainty |
| 16 | Stress tests are missing (Atlas) | **Add three:** non-proportional hazards (effect decays), heavier censoring, and an unmeasured confounder |
| 17 | Coverslip-level splits leak patients: 67–100% of held-out case numbers also appear in training (Fable) | **Add a leakage audit** as a finding. Re-derive it ourselves before any claim about published work |
| 18 | UPMC carries a diffuse signal that maxT cannot see: 20–27% of patterns beyond \|z\| 1.96 against 4% under the null, permutation p ≤ 0.006 (Fable) | **Add an omnibus permutation test** (sum of z², plus higher criticism) next to maxT |
| 19 | Both cohorts' outcomes have now been examined by a reviewer | **Disclose.** The real-data searches are reported as already-seen, so exploratory. The atlas and the controls carry the confirmatory weight |
| 20 | Cut: GPU neighbour search (cKDTree takes about 15 s in total), minus-sampling (drops 33%; r = 0.96 with uncorrected), the upper-quartile feature (r = 0.9 with the mean), Helsinki (both) | **Cut all four.** The one-off neighbour search stays on the CPU. Features are uncorrected |

## Problem

Spatial biology images dozens of proteins in every cell of a tumour. Labs then search for survival-linked spatial patterns. Before imaging, they must choose:
- how many patients;
- how many regions per patient;
- how large a field of view;
- how fine a cell typing.

Each choice trades cost against the chance of finding something real. No tool answers that trade-off for an outcome-discovery search. Baker et al. (2023) simulated tissue to size studies for detecting cell types and their spatial interactions. permGPU (2010) sketched power runs for permutation tests. Neither covers survival endpoints, censoring, batch, multiplicity and imaging error together.

## The idea, in one line

Re-image thousands of simulated studies **cell by cell from real tumour tissue**. Each study varies its patients, regions, field of view and cell-typing error. Plant a known effect on the true biology, then run the full multiplicity-controlled survival search on what each study would have seen. The result is an atlas of what a study design can and cannot find.

**Promise:** before you image your next cohort, see what a study like yours can find, and what it would get wrong.

**Spin:** not a faster search. Your study, simulated on real tissue a thousand times over, before you image a slide.

## How it works

### 1. The truth layer (CPU, once)

- Load all cells: UPMC has about 2.0M cells in 308 regions (6.5k per region, measured).
- Build a distance-sorted neighbour list per cell out to 100 µm with cKDTree, once. Measured: 24, 90 and 332 neighbours per cell at 25, 50 and 100 µm, so about 6.7×10⁸ entries for UPMC.
- Compute the **true** features from full regions and true cell types.
- The pattern grammar (frozen and hashed):
  - tumour focal types × every neighbour type × {25, 50, 100 µm} × {fraction, enrichment};
  - focal-cell-weighted patient mean;
  - C counted exactly and printed.

### 2. The re-imaging kernel (GPU, the hero workload)

Each simulated study draws:
- n patients, with replacement and by index;
- k regions per patient;
- a field-of-view crop of side s inside each region;
- a cell-type mis-call per cell from a confusion matrix at rate ε.

For every focal cell in the crop, the kernel walks its distance-sorted neighbour list once. It skips neighbours outside the crop, which a smaller field of view really cannot see, and counts the mis-called types. It reads all three radii from the same pass, using offsets into the sorted list. A segmented reduction then gives the patient feature matrix X̂ (n × C), which never leaves the device.

Cost: n = 160, k = 4 is about 4M cells × 332 neighbours ≈ 1.4×10⁹ neighbour reads per study. One design point is 1,000 studies, and one atlas is about 50 points: **≈7×10¹³ irregular memory reads**.

Precomputed features cannot be reused. The crop removes neighbours and the mis-call changes their types, so every study must be recounted.

### 3. Outcomes and the statistic

- **Simulated outcomes:** survival is drawn from the **true** features, with HR per SD on one planted pattern (variant: a correlated cluster). The Weibull baseline is tuned to a target event count, with censoring that depends on group.
- **What the study sees:** each study observes only X̂. Attenuation from field of view, regions and mis-calls is the scientific point.
- **The score:**
  - z = U / √(Σₛ Sxxₛ·Srrₛ/(nₛ−1)), where U = Xᵀr (log-rank as a matrix product, with Breslow ties).
  - **Reshape 1:** the maxT threshold is computed once per design point from one permutation run (10⁴ shuffles). Each study then needs one matrix-vector product.
  - The threshold is checked against nested permutations on 3 design points.
- **Outputs per design point:**
  - power;
  - correct-identification rate;
  - false-identification rate;
  - FWER under the null (10⁴ null studies);
  - omnibus power.

### 4. The atlas grid (cohort-specific; size fixed after one point is timed)

| Axis | Values |
|---|---|
| Patients n | 40, 80, 160, 240 (UPMC); 80, 160, 320, 480 (Charville) |
| Regions per patient k | 1, 2, 4, all |
| Field of view s | 250, 500 µm, full region |
| Mis-call rate ε | 0, 10%, 25% |
| True effect | HR per SD 1.25, 1.5, 2, 3 |
| Stress tests | non-PH, heavy censoring, unmeasured confounder (one slice each) |

### 5. Real-data audit (already-seen, so exploratory)

- **UPMC:**
  - endpoint: any-cause death in all 79 patients (27 events);
  - strata: coverslip group;
  - secondary: cancer-specific death.
- **Charville:**
  - endpoint: overall death (38 events);
  - strata: coverslip, after a check that its coverslips are patient-level.
- **Tests:** maxT and the omnibus test, with leave-one-patient-out influence on the top hits.
- **Leakage audit:** how many patients a coverslip-level split shares between train and test.
- **Atlas placement:** each real result is placed on its cohort's atlas, to show what a search at that size could have found.

### 6. Correctness

- **The statistic:** U against statsmodels PHReg (Breslow) at β=0. The variance against exact enumeration of within-stratum permutations on tiny strata. Adversarial cases: ties, constant columns, event-free strata, single-patient strata.
- **The kernel:** must match a NumPy reference cell for cell on 5 regions × 3 crops × 2 mis-call seeds. With no crop and ε=0 it must reproduce the truth layer exactly.
- **The threshold shortcut:** must agree with full nested permutations on 3 design points (power within Monte Carlo error).
- **Calibration:** FWER from 10⁴ null studies, with binomial intervals, under group-dependent censoring. Positive controls (planted effect on real features with real censoring) must be recovered at the simulated power.

## The speedup, and how we make it count

**The ladder (one design point, 1,000 studies):**

| Step | What changes | Expected |
|---|---|---|
| 0. What a lab would run | Per study: cKDTree features plus a permutation test per study | CPU-days for one atlas (context only) |
| 1. Algebra | One threshold per design, then a matrix-vector product per study, removing a factor of ~10⁴ shuffles | Reshape 1, measured on the CPU |
| 2. Same kernel, best CPU | Numba-parallel re-imaging gather on all cores; our baseline | Measured |
| 3. Same kernel, GPU | Warp per focal cell; distance-sorted lists; one pass for all radii; crop mask and relabel built on device; segmented reduction; studies batched per launch over one resident adjacency | Measured |

**Why step 3 should win honestly:** the kernel is bound by irregular memory reads.
- About 1.4×10⁹ gathers per study, roughly 7 GB of traffic.
- An H100-class GPU moves about 3 TB/s: a few ms per study.
- An 8-core CPU sustains roughly 10–40 GB/s on random gathers: about 0.2–0.7 s per study.

Expected 50–200× over 8 cores and 10–30× over a 64-core node. These are estimates until measured; we report achieved GB/s for both against their peaks (roofline).

**What makes the score:**
1. **A credible baseline:** our own numba kernel, the same algorithm, on all cores of the best CPU we can get; a 64-core cloud node if allowed.
2. **End to end:** includes adjacency upload, mis-call draws and the statistics, not just the kernel.
3. **Roofline:** achieved bandwidth against peak on both machines.
4. **Scaling:** in studies, cells per study and radius, plus multi-GPU if offered (studies are independent).
5. **The ladder slide:** algebra first (a reshaping credit no hardware gives), then hardware. Each step measured.
6. **Never claim:** a custom GEMM, the fused max, TF32, or any 300k figure.

## What we build (priority order = the critical path)

| P | Area | Build | Done when |
|---|---|---|---|
| P0 | `core/` | Loaders; patient identity (case numbers, tuple sensitivity); coverslip groups; endpoints; frozen grammar and its hash; exact C | Both cohorts load; C printed; hash recorded |
| P0 | `stats/` | Float64 reference: stratified score, permutation variance, maxT, omnibus | Matches PHReg U and exact enumeration; adversarial cases pass |
| P0 | `reimage/` | NumPy reference re-imager, then the numba CPU kernel, then the GPU kernel (CuPy RawKernel or Numba-CUDA) | All three agree cell for cell; reproduce the truth layer at no crop and ε=0 |
| P1 | `atlas/` | Simulator, threshold shortcut and its nested check, grid runner, outputs | Atlas charts per cohort, plus a calibration chart with intervals |
| P1 | `bench/` | The ladder: the same kernel on CPU and GPU, end to end; roofline; scaling | Measured charts |
| P1 | `audit/` | Real-data maxT and omnibus; leave-one-patient-out; leakage audit | Results, labelled already-seen |
| P1 | `demo/` | The Preflight demo page; saved-run replay; deck | Opens on the number; works with Wi-Fi off |
| P2 | `atlas/` | Stress-test slices; source bootstrap | Shown as bands |
| P2 | `stats/` | Step-down maxT, under the stated assumption | Shown on one slice |

**Cut:**
- custom GEMM, the fused max, TF32/FP16
- LOCO
- the naive/stratified toggle
- the HPV and coverslip-ID controls
- Helsinki
- minus-sampling
- the upper-quartile feature
- GPU neighbour search
- the marker panel, C-index, random hold-out, DFCI
- the treatment-decision story

**What we don't build:**
- segmentation
- a backend service
- clinical claims
- pooled-cohort atlases

## Three prepared outcomes

| Outcome | What we say |
|---|---|
| The atlas shows design choices matter (expected) | "Doubling regions per patient beats adding X patients; at 25% mis-calls you lose Y% power" |
| The real-data search finds nothing under maxT but the omnibus is significant | "The signal is real but diffuse; no single pattern is detectable at this size, and here is the size where one would be" |
| A real hit survives with stable influence | "A ranked hypothesis, already-seen, and the atlas says how often a study this size gets the identity right" |

## Demo

**Hero:** one design point, 1,000 re-imaged studies from 2M real cells. Our all-core CPU kernel against the GPU kernel, with the same outputs and the measured time on screen.

**One chart:** power against patients, one line per regions-per-patient setting, at a fixed effect size, with our cohort marked.

**The spark:** one big button, "Run the studies". A judge sets patients, regions per patient, field of view and mis-call rate. The GPU re-images 1,000 studies in seconds and returns power and identification rate. Cached grid fallback.

| t | On screen |
|---|---|
| 0:00–0:15 | Problem and promise |
| 0:15–0:25 | The number: one study design simulated, CPU-hours to seconds |
| 0:25–0:50 | Real tissue: a field of view shrinks and cells are mis-called, so the pattern blurs. This is what a study sees |
| 0:50–1:15 | The ladder: algebra, then best CPU, then GPU, with the roofline |
| 1:15–1:45 | The judge's button |
| 1:45–2:20 | The atlas: "regions beat patients" (or whatever it shows) |
| 2:20–2:45 | Real data: maxT empty, omnibus significant, placed on the atlas |
| 2:45–3:00 | Promise; "Bring us your study design before you image it" |

## Grade (Zeus's judgement; v3 was over-graded by 2–3 points, so this is tempered)

| Version | Correctness | Reshaping | Speedup | Science | Top 3 | 1st |
|---|---|---|---|---|---|---|
| v3 as regraded (Atlas / Fable) | 6 / 4 | 6 / 3 | 5 / 3 | 7 / 6 | ~25% | ~10% |
| v4 on paper | 7 | 6 | 6 | 7 | ~45% | ~20% |
| v4, ladder measured and kernel ≥50× over 8 cores | 7 | 7 | 7 | 7 | ~50% | ~25% |

**What still loses it:**
- **"Baker et al. already did spatial power analysis."** Answer: survival discovery, multiplicity, censoring and the imaging axes together, with the difference shown on one slide.
- **"A smart CPU person would use expected counts."** Answer: run the analytic attenuation alongside and show where it fails, at small fields of view where variance matters. If it doesn't fail, say so.
- **A rival with a GPU-native physics problem.** If the 11:00 pitch round has one with a real hours-long CPU baseline, take it; v4 is the strongest version of this idea, not a guaranteed winner.

## Risks to resolve in the first hour

1. **Adjacency memory:** 6.7×10⁸ int32 entries is about 2.7 GB for UPMC; Charville is still to count. Fits on one GPU; check the event GPU's memory.
2. **The threshold shortcut:** confirm it holds when X̂ is noisy (mis-calls change its correlation). If it drifts, compute one threshold per (design, ε) pair.
3. **Charville coverslips:** check that they are patient-level before using them as strata.
4. **Kernel route:** NumPy reference by build + 1.5 h; numba CPU by + 3 h; GPU by + 5 h. Atlas runs on numba if the GPU kernel slips.

## Areas and owners

| Area | Directories | Owner | Issues |
|---|---|---|---|
| Core contract | `core/` | — | — |
| Reference statistics | `stats/` | @atiladeokegab [Atlas] | #4 |
| Re-imaging kernel | `reimage/` | @atiladeokegab [Prometheus] | #2 |
| Atlas | `atlas/` | @atiladeokegab [Atlas] | #5 |
| Benchmarks | `bench/` | @atiladeokegab [Hermes] | #3 |
| Real-data audit | `audit/` | @atiladeokegab [Hephaestus] | #6 |
| Demo and deck | `demo/` | @atiladeokegab [Zeus] | #7 |
| Tests | `tests/` | @atiladeokegab [Prometheus] | #8 |
| Submission | `docs/film/`, `docs/SUBMISSION.md`, `README.md` | @atiladeokegab | #9, #10, #11 |

## Deadlines (PROVISIONAL, confirm at 11:00)

| Deadline | UTC |
|---|---|
| Build start | 2026-11-21T11:00Z |
| Spine: loaders, reference stats, NumPy re-imager on one region | + 2 h |
| Numba kernel and one design point timed; size the grid | + 3 h |
| GPU kernel matches the reference | + 5 h |
| P1 green or cut | 2026-11-21T20:00Z |
| Harsh rehearsal | freeze − 4 h |
| Code freeze | submit − 60 min |
| Submit | before 2026-11-22T18:00Z (ask) |

## Ask the organisers at 11:00

1. Judging format and length; are slides allowed?
2. Is our own all-core CPU version of the same kernel the baseline you expect? Can we use a 64-core node for it?
3. Does a methods result (a design atlas) score as highly as a discovery?
4. What are the rules on pre-existing code, data and specs?
5. Which GPU and how much memory? Is a multi-GPU node available?
6. When and how do we submit?
7. Are mentors also judges?

## Lessons changed in v4 (rows not listed stand as in `IDEA-v3-power-atlas.md`)

| # | Lesson | v4 |
|---|---|---|
| 11 | Test from the goal | Kernel must reproduce the truth layer; nested check of the shortcut; 10⁴-study calibration with intervals |
| 12 | A bolder approach? | Re-imaging real tissue is the bolder version; the GPU now does the hard part |
| 15 | Spine first | NumPy re-imager on one region by + 2 h |
| 23 | Critical path first | NumPy, then numba, then GPU; the atlas never waits on CUDA |
| 6 | One hero, one chart | Same kernel CPU against GPU; power against n by regions |
| 20 | One big button | Now with study-design knobs a lab actually controls |
| 43 | Prepare the null | Three outcomes, including omnibus-only |
| 37 | Judges decide in 20 s | The number by 0:25 |
| 4 | One before/after number | "One study design simulated: X CPU-hours → Y s" |
| 8 | Do less | 11 items cut, listed above |
| S | Play to win | Every reviewer attack has a row in the corrections table |
| 3 | The user does it today | Labs choose patients, regions and field of view before imaging today |

## Approval 1 checklist

- [ ] Room's pitch round done; v4 chosen or replaced
- [ ] Organiser answers recorded
- [ ] Prometheus score out of 6
- [ ] Repo published
