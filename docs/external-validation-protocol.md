# Preflight Basel external validation

Protocol `preflight-basel-external-v1`, proposed 2026-11-23. Hermes records the reviewed protocol and split hashes before anyone opens reserved outcomes.

## Question

Does the imaging plan that Preflight prefers on pilot patients preserve a prespecified spatial biomarker's survival association better on reserved patients? This is a retrospective test on archived imaging from a cohort outside Preflight's original inputs.

## Source and eligibility

Use the Basel breast tumour cohort from Jackson and Fischer. Pin the CC BY 4.0 processed release at Zenodo DOI `10.5281/zenodo.20621983`. The 100-patient `sce_basel.rds` (MD5 `c704430733435c8866438362edec4274`) is import smoke only. For the experiment, select Basel only from `sce_full.rds` (MD5 `ddbc54e6e8a604a8febabd5a8ad2cd1a`). Retain the exact source version, checksum, importer SHA, and counts for exclusions.

Require a unique patient-to-image join, annotated tumour tissue, finite coordinates, documented positive follow-up and a recognised endpoint status. Use the endpoint availability flag during cohort selection. Require the larger square to fit within verified image dimensions. Choose one eligible tumour core per patient. Choose the largest usable image area if more than one qualifies and break ties by lexical image ID. All assignment and core choices occur before opening reserved outcomes. Preserve the published map from pixels to micrometres (1 µm per pixel). Count excluded patients and reasons before assigning the split. Publish aggregate counts only.

[Original study](https://doi.org/10.1038/s41586-019-1876-x) · [Processed release](https://zenodo.org/records/20621983) · [Source preparation](https://github.com/BodenmillerGroup/imcdatasets/blob/devel/inst/scripts/make-JacksonFischer-2020-BreastCancer.Rmd)

## Endpoint and biomarker

**Primary endpoint** is disease-specific survival using `OSmonth` (processed alias `patient_OS_months`) in months. Count `Patientstatus` (processed alias `patient_status`) of `death by primary disease` as an event. Censor known other deaths and living statuses at the recorded time, following the published Basel analysis. Exclude missing, unknown, conflicting or nonpositive endpoint records. Preserve the ordered meaning of event and censor flags through the join.

**Primary biomarker** is the fraction of visible neighbours within 50 µm that are T cells, averaged equally across visible tumour epithelial focal cells for each patient. Map Basel metaclusters 14–26 to tumour epithelium, excluding myoepithelial cluster 27. Map T-cell metaclusters 3 and 5 to T cells, excluding mixed B/T cluster 2. Keep all remaining cells in the denominator. The published labels rely on the Basel IMC panel, including PanCK and CD3ε. Exclude no-neighbour ambiguity by using the existing Preflight convention of a zero fraction for an isolated focal cell. A patient with no visible focal cells has a missing biomarker. Set classification perturbation to zero to isolate acquisition sampling. Derive one mean and standard deviation from eligible pilot patients under the full observed field. Freeze and reuse both across designs and partitions.

[Basel phenotype and survival analysis](https://raw.githubusercontent.com/BodenmillerGroup/SCPathology_publication/master/R/BaselTMA_pipeline.Rmd) · [Acquisition methods](https://www.repository.cam.ac.uk/bitstream/1810/303322/1/The%20Single-Cell%20Pathology%20Landscape%20of%20Breast%20Cancer.pdf)

## Split and imaging comparison

Assign patients in ascending order of SHA-256 over UTF-8 `preflight-basel-external-v1`, NUL, `20261123`, NUL, and the canonical patient ID. Break hash ties by lexical ID. The first floor(N/3) patients are pilot. All other patients are reserved. Keep all cells and images for each patient on their assigned side. Require at least 67 eligible patients in each partition so that the 60-patient design fits after deletion of any one of ten approximately equal patient blocks. A sparse or unsuitable archive produces a feasibility report before outcome opening.

- **A** uses one 300 × 300 µm field in each of 60 patients.
- **B** uses one square field of side 300√2 µm in each of 30 patients.
- Each plan images 5.4 mm² of nominal area. Use area as the explicit budget proxy. Verify field geometry so neither design silently falls back to a smaller archived field.

For each partition draw 200 independent paired acquisitions. Use NumPy PCG64 seeded with `SeedSequence([20261123, partition_code, draw_index])` where pilot is 1 and reserved is 2, and draw indices run from 0 to 199. Draw a permutation without replacement of that partition's eligible IDs in manifest order. Use the first 60 for A and the first 30 for B. After drawing the permutation, draw one pair of uniform crop-placement values for each eligible patient in manifest order. Reuse each pair for that patient across designs and patient-block deletions. Uniformly place the square within the archived image bounds. Retain each crop's visible cells and apply the same feature definition in all fields. Report nominal area and the proportion of sampled patients whose biomarker could be measured.

## Analysis and decision

Within each partition, fit a single-predictor Cox proportional hazards model with Breslow ties on the whole observed fields for the dense reference. Use the same fit on A and B. Use the frozen pilot scale and observed clinical outcomes. Record log hazard ratio per pilot standard deviation. Use complete cases and report the number of patients and events for each design. Require ten observed events for the dense reference and five for each design fit, a finite converged coefficient and estimable variance. Count all failures and include only draws with a usable fit in both arms in the paired estimate. Require at least 140 valid pairs.

For each valid pair calculate `|logHR_A − logHR_dense| − |logHR_B − logHR_dense|`. Report its mean, the fraction of failed pairs and a percentile 95% interval from 2,000 bootstrap samples of paired differences, using `SeedSequence([20261123, partition_code, 200])`. This interval quantifies Monte Carlo uncertainty over repeat acquisition draws conditional on the given cohort.

For patient-level uncertainty, assign eligible patients in each partition to ten blocks by zero-based manifest index modulo 10. For each block, omit all its patients from the dense reference and both designs, and recalculate the paired mean. In each draw, filter out omitted patients from the original permutation, refill A and B from the next eligible indices, and reuse the precomputed crop-placement values. Keep the pilot scale fixed. With the ten leave-block-out means, estimate the delete-group jackknife standard error as the square root of 9/10 times the sum of squared deviations from their mean. Report the full-cohort mean plus or minus 2.262 times that standard error as an approximate 95% patient-level interval. If any block deletion cannot meet the fit requirements, report the patient-level interval as unavailable. Also report all ten block estimates. This assesses sensitivity to patients in this cohort.

The pilot recommends A if the mean is below zero and B if it is above zero. Exact ties or insufficient usable fits produce no pilot recommendation. The reserved comparison supports the recommendation when it has the same direction, both the patient-level and conditional intervals exclude zero in that direction, and the dense-reference association is estimable with a 95% interval excluding zero. Treat unavailable or inconclusive intervals as insufficient evidence for confirming the recommendation. Report a contrary directional ranking when the reserved point estimate favours the other design, together with its uncertainty. Keep model-based planted-outcome power results separate from this observed comparison.

## Opening and release

Create a local manifest JSON with keys `eligible`, `pilot` and `reserved`, each an ordered list of opaque patient-ID strings. A separate reviewer or gatekeeper prepares the private eligible-ID manifest and outcome-availability flags from the verified source. The gatekeeper materializes separate pilot and reserved outcome files. Give the builder pilot outcomes and outcome-free geometry and phenotype files. Keep the reserved outcome file under separate custody through the pilot checkpoint. Blinded aggregate feasibility and join checks can be computed upstream without releasing reserved rows to the builder.

Prometheus independently reviews this protocol and the split checker, and Atlas independently reviews the importer and runner. Hermes records the exact protocol, split, data and code versions and the frozen pilot result. Hermes then issues an opening receipt JSON with `protocol_sha256` and `split_sha256`, hashes of their exact file bytes. The runner calls `check_outcome_opening` before reading the reserved outcome table and checks that the joined patient IDs are exactly the reserved group. The checker cannot replace separate custody of outcome values. Keep the receipt, manifest and patient-level inputs outside Git. Make any feasibility clarification before issuing the receipt. After opening, run the single frozen analysis and report aggregate results, including a negative or inconclusive result.
