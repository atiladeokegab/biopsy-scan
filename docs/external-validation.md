# Basel external validation

**Inconclusive at the prespecified pilot feasibility check.** The pilot supplied no design recommendation. The reserved outcomes stayed with the custodian. The cohort import and patient split were verified.

## Cohort and protocol

The retrospective comparison asked whether the imaging plan recommended on pilot patients better preserved a prespecified spatial biomarker’s survival association on reserved patients. It used the Basel breast tumour cohort from [Jackson and Fischer](https://doi.org/10.1038/s41586-019-1876-x), from the [CC BY 4.0 processed release at Zenodo](https://zenodo.org/records/20621983). The verified `sce_full.rds` file was 1,248,797,988 bytes with MD5 `ddbc54e6e8a604a8febabd5a8ad2cd1a`.

The Basel tumour rows contained 767,833 cells, 289 images and 281 patients. At least one image could fit either planned field for 270 patients. Positive, known follow-up was available for 280 patients. There were 269 patients at the intersection and zero conflicting endpoint records. The [frozen protocol](external-validation-protocol.md) selected one image per patient and assigned 89 to pilot and 180 to reserved with zero overlap. The pilot contained 16 disease-specific events.

Protocol SHA-256 `7acf85e5d93508697882b6ff6ae4801a1d15a7d1254532ae1aa124115f7f955d`. The primary endpoint was disease-specific survival. The biomarker was the fraction of neighbours within 50 µm that were T cells, averaged over tumour epithelial cells using the published Basel phenotypes.

## Designs and pilot stopping rule

Both designs allocated 5.4 mm² of nominal imaged area. Plan A allocated one 300 µm square to each of 60 patients. Plan B allocated one square of side 300√2 µm to each of 30 patients. The archive supported both field sizes in the selected patient set.

The protocol required five observed events per design fit and at least 140 usable pairs out of 200 predetermined acquisitions. In each acquisition, A and B used the first 60 and 30 patients of the same permutation of pilot patients. The permutation started from frozen manifest order and used NumPy PCG64 with `SeedSequence([20261123, 1, draw_index])` for draw indices 0–199.

In **135 of 200** draws, B contained at least five events before cropping and fitting. A crop with no visible tumour epithelial cells makes that patient’s biomarker unavailable for fitting, so the event count sets an upper bound of 135 usable pairs. The lowest such ceiling across the ten prespecified patient-block deletions was 92. The reported pilot stopped at that gate, leaving dense and cropped effect estimates and uncertainty intervals unavailable. The archive therefore provides no answer about which design better preserves the survival association. A cohort with more observed disease-specific events could make the same comparison feasible.

## Verification and reproduction

Two independent runs reproduced the 135-draw ceiling. The source import took 29.7 seconds, the custodian’s verification and join took 27.9 seconds, and the pilot gate took 4.2 seconds on the existing CPU instance. Incremental cloud and software cost was £0. This report contains aggregate counts only.

The analysis takes an outcome-free geometry export, the custodian’s deterministic split manifest and a separate pilot endpoint file. The [protocol](external-validation-protocol.md#opening-and-release) specifies custody and manifests. Keep the archive and coded files outside the checkout.

To reproduce the split as custodian, read `colData` from the verified `sce_full.rds` with R. Filter `patient_cohort` to `Basel` and `patient_disease_status` to `tumor`. Intersect the patients with an image supporting both field sizes with those who have a single distinct, positive finite `patient_OS_months` and a known `patient_status`. Normalize status by trimming whitespace and lowercasing. The allowed values are `death by primary disease`, `alive`, `alive w metastases` and `death`. Exclude patients with conflicting repeated endpoint pairs. Use `split_patients` in `scripts/check_external_validation_split.py` to create the ordered `pilot` and `reserved` lists from those `eligible` IDs. Supply the manifest and a CSV with `patient_id`, `patient_status` and `patient_OS_months` for exactly the pilot IDs. Only `death by primary disease` counts as an event.

```sh
uv run python -m audit.external_basel "$BASEL_RDS" "$BASEL_GEOMETRY" --rscript "$RSCRIPT"
uv run python scripts/check_external_validation_split.py "$BASEL_SPLIT"
uv run python -m audit.external_validation \
  --geometry "$BASEL_GEOMETRY" --manifest "$BASEL_SPLIT" \
  --protocol docs/external-validation-protocol.md --outcomes "$BASEL_PILOT_ENDPOINTS"
```
