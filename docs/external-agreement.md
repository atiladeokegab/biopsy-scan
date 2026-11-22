# Why the external baseline agrees 14/50, and agreement on a matched config

Req: A-9.2@1. Code: `bench/external_agreement.py` (test: `bench/test_external_agreement.py`).
Results: `bench/results/external-agreement-20261122T133722Z.json` (run 1) and
`bench/results/external-agreement-20261122T135211Z.json` (run 2).

## The starting number

The off-the-shelf baseline (#132, `bench/external_baseline.py`, result
`bench/results/external-baseline-20261122T043823Z.json`, `summary.top1_same`) picked the same top feature as
Preflight in **14 of 50 studies, MEASURED** (null stream 5/25, alternative 9/25; outcomes SIMULATED). In none of the
36 misses did the two pick the same (tumour type, neighbour type, radius) with the other statistic; median top-10
overlap 5 of 10 (all MEASURED, from that file's records).

## Short answer

The two pipelines are not computing the same statistic. The main cause is the "enrichment" half of the features:
squidpy's `nhood_enrichment` returns a permutation z-score, while Preflight's enrichment is a ratio to the region's
cell-type mix. The Cox test is a minor cause (Wald versus score). When squidpy's graph is used with Preflight's
per-cell definition, and lifelines' own Efron score is taken at beta = 0, the two agree to floating-point rounding:
**top-1 16/16 MEASURED** on the studies run locally.

## Definitional differences, ranked by effect on top-1

| # | What | Preflight | Baseline (#132) | Effect (MEASURED unless stated) |
|---|---|---|---|---|
| 1 | Enrichment statistic | per focal cell `fraction / p_j`, with `p_j` the share of type j among the region's visible cells: `core/truth.py:65,72`, `reimage/fused.py:62`, `core/grammar.py:5-6` | squidpy permutation z, `(count - mean(perms)) / std(perms)` over 1,000 label shuffles: `bench/external_baseline.py:97,104`; squidpy 1.8.2 `squidpy/gr/_nhood.py:220` | Feature Spearman against ours, median **0.79** (#132, 50 studies). Swapping only the enrichment half into our X and keeping our score test moved top-1 away from ours in the one null study run (picked 625, ours 359). Alt study: top-1 kept. 2 studies, too few to give a rate |
| 1a | Shuffle scope of that z | (no permutation: the ratio uses each region's own mix) | `nhood_enrichment` is called without `library_key` (`bench/external_baseline.py:97`), so labels are shuffled across both of a patient's regions (`_nhood.py:452-458`) | Part of row 1. It cannot be split out without the z |
| 1b | NaN | 0 when `p_j = 0` (`core/truth.py:72`) | z is NaN when the permutation std is 0 (`_nhood.py:220`), for example a type absent from the patient | 396 NaN only in squidpy's X over 50 studies (#132), and 3 per study in both local studies. lifelines drops those patients for that column |
| 1c | Monte Carlo noise | none (deterministic) | z depends on the permutation seed | Not measured separately |
| 2 | Fraction weighting | mean over focal cells of the per-cell share `n_j / n` (cells with no neighbours count as 0): `core/truth.py:71,76`, `core/grammar.py:79` | pooled edge share `sum n_tj / sum n_t`, so cells with many neighbours weigh more: `bench/external_baseline.py:103` | Feature Spearman **0.994** (#132). With only this swap into our X and our score, top-1 matched in **10/16** (5/8 null, 5/8 alternative); median z Spearman 0.991 |
| 3 | Cox test | stratified Cox **score** z at beta = 0, Breslow ties, analytic information (`stats/screen.py:3-11,304`); maxT permutation calibrates the call, not the ranking | lifelines `CoxPHFitter` **Wald** z at the MLE, Efron ties (`bench/external_baseline.py:123`) | On the matched X, Wald top-1 = ours in **4/4**. Median z Spearman 1.0 (rounded); \|z\| Spearman 0.9998 to 1.0. Small |
| 4 | Ties | Breslow | Efron | **None** here: 0 tied event times in the studies checked (null 0: 159 patients, 35 events, 0 tied event times). Efron equals Breslow without ties |
| 5 | Strata, NaN rows | coverslip group; per-column complete cases | same strata (`bench/external_baseline.py:123`); per-column `dropna` | None. Identical by construction |
| 6 | Neighbourhood graph | `cKDTree.query_pairs`, `d <= r`, no self (`core/truth.py:43-47`) | `spatial_neighbors(coord_type="generic", radius=r, library_key="region")`, no self | None. Same graph: X differs by at most 1e-14 (below) |
| 7 | Close races | - | - | Many studies are near-ties: our top-1 \|z\| leads the second by 0.03 to 0.58 (median 0.19) in the 16 local studies. Half of the 50 are nulls, where the top-1 is a race among noise features. Any change to the statistic reshuffles these, so items 1 and 2 cost more top-1 matches than their feature correlations suggest |

Maximum T (maxT) gives family-wise adjusted p-values that rise as |z| falls. It decides whether a study is called.
It cannot change which feature ranks first, so it plays no part in top-1.

## Matched config

- **Neighbourhood:** `squidpy.gr.spatial_neighbors(coord_type="generic", radius=r, library_key="region",
  delaunay=False, set_diag=False)` for each radius of 25, 50 and 100 um.
- **Features:** written out by hand on squidpy's graph (`graph_features`). Each cell gets its neighbour-type counts as
  `A @ onehot`, then the per-cell fraction, enrichment = fraction / region share `p_j`, and the mean over the focal
  cells of type t. This step does not reuse `core.truth`. squidpy has no function for this per-cell mean.
  `nhood_enrichment` and `co_occurrence` return only pooled quantities, and `co_occurrence` has no `library_key`.
- **Cox test:** lifelines' Efron gradient and Hessian at beta = 0, from the routine `CoxPHFitter`'s default model
  fits with (`SemiParametricPHFitter._get_efron_values_single`, a private method of lifelines 0.30.3). It is summed
  over the coverslip strata, and z = U / sqrt(I). lifelines has no public score test for Cox models.

Versions: squidpy 1.8.2, lifelines 0.30.3, anndata 0.12.19, pandas 2.3.3, numpy 2.4.6 (the Modal image's pins).

## Agreement on the matched config (all MEASURED, deterministic, one repeat; outcomes SIMULATED)

Run 1 covered the first 8 studies of each stream (16 of #132's 50). In all 16 the X sha256 equals #132's recorded
`x_sha_ours`, and our top-1 equals its recorded `ours_top`, so the local cells match the Modal run.

| Config | Top-1 = ours | Top-10 overlap (median) | z Spearman (median) | max \|ΔX\| | max \|Δz\| |
|---|---|---|---|---|---|
| matched (squidpy graph + grammar + lifelines Efron score) | **16/16** (8/8 null, 8/8 alt) | 10 | 1.0 | 9.8e-15 | 1.1e-14 |
| our X + lifelines Efron score | 16/16 | 10 | 1.0 | 0 | 2.7e-15 |
| our X with squidpy's pooled fraction, our score | 10/16 | 8.5 | 0.991 | 0.27 | 1.19 |
| #132 as run, on these 16 studies | 5/16 | (from the #132 records) | | | |

Run 2 used fewer studies because these configs are slow on this loaded VM: the first 2 studies of each stream for
the Wald fits, and the first 1 of each stream for squidpy's 1,000-permutation features.

| Config | Studies | Top-1 = ours |
|---|---|---|
| matched X + lifelines default Wald (CoxPHFitter per feature) | 4 | 4/4 |
| squidpy's #132 features (pooled share + permutation z) + our score test | 2 | 1/2 (null 0 picked 624, the same as #132's lifelines pick, so the Wald test did not drive that miss) |
| our X with squidpy's permutation z in the enrichment slots, our score | 2 | 1/2 |

Hardware: AMD EPYC 7B13, 8 cores, 2 workers. The shared VM had a load average of 21 to 34 during both runs. Wall
times were 871 s (run 1) and 843 s (run 2). They are MEASURED but not speed figures. The agreement numbers are
deterministic, so the spread is 0 by construction. squidpy's z uses #132's per-study seed.

## What still differs, and what this does not show

- **The matched config is not squidpy's statistic.** squidpy provides the graph, and the aggregation is about six lines
  of numpy. No combination of squidpy's documented parameters gives a per-cell mean of fractions, or an enrichment
  ratio to each region's own mix. A lab using `nhood_enrichment` as documented gets a different feature (row 1). That
  is a choice of statistic, not a bug in either tool.
- **The matched score test calls a private lifelines method.** It may change between lifelines versions. lifelines'
  public, default path is the Wald test, which agreed on top-1 in 4/4 but is not identical.
- **Sample sizes are small.** The matched config ran on 16 of the 50 studies, Wald on 4 and squidpy's permutation z
  on 2. The full 50 would take about 45 minutes of local CPU for the matched config alone at the measured rate
  (ESTIMATE, from run 1's 871 s for 16 studies on 2 workers). That exceeds this VM's 10-minute job budget, and
  Modal was out of scope. The 2-study rows show the mechanism, not a rate.
- **Effects were not split by item.** The ranking of rows 1, 2 and 3 comes from feature correlations (50 studies)
  and the one-at-a-time swaps above. It was not a full factorial run.
