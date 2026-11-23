# METABRIC external validation

**Inconclusive under the frozen rule.** Plan A gave estimates closer to the full-image reference in both partitions. In the reserved comparison, both intervals for the error contrast favoured A. The full-image survival association's interval crossed zero, so the result did not meet the prespecified confirmation criterion.

The experiment compared two ways to allocate 8.1 mm² of imaging area in archived METABRIC breast tumour tissue. Plan A spread the budget across 90 patients with one 300 µm square each. Plan B used one 300√2 µm square in each of 45 patients. The [frozen protocol](external-validation-protocol-v2.md) defines the comparison and the outcome-access rules.

The two CC BY 4.0 archives cover 1,123,466 processed cells across 794 matched images. Of 572 patients with a fitting invasive tumour image, 555 also had valid disease-specific follow-up and known ER status. The locked split assigned 185 patients to pilot and 370 to reserved. The area uses the published nominal scale of 1 µm/pixel. [Original study](https://doi.org/10.1038/s41588-022-01041-y) · [Processed data](https://zenodo.org/records/7324285) · [Corrected masks](https://zenodo.org/records/15615709) · [Pixel scale](https://doi.org/10.1038/s41551-025-01357-0)

The biomarker is the mean fraction of visible neighbours within 50 µm that are annotated pure CD4+ or CD8+ T cells, averaged across invasive tumour epithelial focal cells. The analysis estimates one log hazard ratio per pilot standard deviation, with separate baseline hazards for ER-positive and ER-negative disease. The contrast is the absolute error of plan A relative to the full-image association minus the absolute error of plan B. A negative contrast favours A.

## Pilot and reserved comparison

The pilot selected A by the prespecified paired error contrast. The custodian released only the METABRIC reserved table after the opening receipt matched the frozen protocol, split, and pilot checkpoint. Both analysis stages reused the pilot full-image biomarker mean of 0.02443 and population SD of 0.03146.

| Measure | Pilot | Reserved |
|---|---:|---:|
| Full-image patients / disease-specific events | 185 / 65 | 370 / 109 |
| Full-image log hazard ratio per pilot SD (95% CI) | 0.0835 (−0.1230 to 0.2900) | −0.0316 (−0.1811 to 0.1179) |
| Usable pairs / attempted | 200 / 200 | 200 / 200 |
| Mean error contrast (A minus B) | −0.0926 | −0.1141 |
| 95% acquisition interval, conditional on the cohort | −0.1161 to −0.0701 | −0.1397 to −0.0881 |
| Approximate 95% patient-block interval | −0.1576 to −0.0277 | −0.1792 to −0.0490 |
| Sampled patients with a measurable biomarker, A / B | 98.21% / 99.88% | 99.20% / 99.64% |
| Complete-case events per draw, A (median, range) | 31 (24–40) | 26 (15–36) |
| Complete-case events per draw, B (median, range) | 16 (9–25) | 13 (5–21) |

Plan A had more missing biomarker measurements in both partitions. The comparison covers this cohort, phenotype, and pair of retrospective imaging designs.

### Patient-block deletion estimates

| Omitted block | Pilot error contrast | Reserved error contrast |
|---:|---:|---:|
| 0 | −0.0895 | −0.0927 |
| 1 | −0.1020 | −0.0875 |
| 2 | −0.1045 | −0.1056 |
| 3 | −0.1020 | −0.1152 |
| 4 | −0.0842 | −0.1140 |
| 5 | −0.0762 | −0.1086 |
| 6 | −0.0792 | −0.1144 |
| 7 | −0.0985 | −0.1130 |
| 8 | −0.0981 | −0.1184 |
| 9 | −0.0915 | −0.1074 |

The pilot took 62.5 seconds and the single reserved run took 74.1 seconds on the existing CPU instance. This report contains aggregate results. The Basel reserved outcomes remained under their original custody.

## Reproduction

Keep the archives and coded input tables outside the checkout. The custodian prepares the frozen split, one selected image per eligible patient, and separate pilot and reserved endpoint tables as specified in the protocol. The command below runs the pilot with the exact pinned inputs. The reserved invocation additionally takes `--pilot` and `--receipt` and uses the reserved endpoint table only after receipt verification.

```sh
uv run python -m audit.external_metabric \
  --processed "$METABRIC_PROCESSED" --masks "$METABRIC_MASKS" \
  --selection "$METABRIC_SELECTION" --manifest "$METABRIC_SPLIT" \
  --protocol docs/external-validation-protocol-v2.md \
  --endpoints "$METABRIC_PILOT_ENDPOINTS"
```
