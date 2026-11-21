# Contract

Written by the lead's agent at the start of the build. Every area builds against it; only the lead
changes it (anything else is a change-request, AGENTS.md §6). Owner of everything here: `core/`.

## Assignment

Re-image thousands of simulated spatial-biomarker studies cell by cell from real CODEX tissue (UPMC,
Charville), so a lab can see, before imaging a slide, which study designs (patients, regions per
patient, field of view, mis-call rate) can find a survival-linked spatial pattern.

There is no backend service, app factory, store or gateway: the product is a Python library plus
saved runs. `core/` is the shared core; `reimage/`, `stats/`, `atlas/`, `bench/`, `audit/` import it
and never re-implement it.

## Data

- Source: Zenodo record 13179600 (CC-BY-4.0; cite Wu et al., Nat. Biomed. Eng. 2022).
  `make data` (`scripts/fetch_data.sh`) puts the two label CSVs and two raw zips in `data/`, which
  is never committed (`data/.gitignore` is `*`). `LOCAL_SRC=<dir>` links local copies instead.
- Everything in `data/` is untrusted: parsed as CSV only. Scripts that read it run with `python -I`.
- Units: cell coordinates are pixels in the files; `core.cohorts.PIXEL_UM = 0.377` converts them to
  µm. Every distance in this contract is µm.

## Cohort tables (`core.cohorts`)

`load_regions(cohort, identity="tuple") -> DataFrame`, one row per region, label-file order:

| Column | Type | Meaning |
|---|---|---|
| `region_id` | str | e.g. `UPMC_c001_v001_r001_reg001` |
| `coverslip` | str | `c001`…, from the region id |
| `case` | str | case number from the sample label (`""` when UPMC has `NA`) |
| `patient_id` | str | patient identity (below) |
| `time_days` | float64 | follow-up in days; NaN when unknown |
| `event` | float64 | 1 death, 0 censored, NaN when unknown |

`load_patients(regions) -> DataFrame`, one row per patient sorted by `patient_id`; **the row number is
the patient index** used by every array below. Columns: `patient_id`, `time_days`, `event`,
`stratum` (modal coverslip: most regions, ties to the lowest), `n_coverslips`, `n_regions`.

Patient identity and endpoints, as measured on the data:

| | UPMC | Charville |
|---|---|---|
| Identity | one patient per clinical tuple (status, primary_outcome, recurred, hpvstatus, survival_status, survival_day) | case number = sample label before `_` |
| Patients | **79** (sensitivity `identity="tuple+case"`: **81**, splitting the 2 tuples that hold two case numbers) | **162** |
| Endpoint | any-cause death (`survival_status`), `survival_day` | death (`alive_or_deceased`), last contact − surgery in days |
| Events | **27** | **38**; 2 patients (24135, 24136) have no dates, so no endpoint |
| Coverslips | 7; **74 of 79 patients span >1 coverslip**, so coverslip is not patient-level | 4; 1 patient spans 2 |
| Cells | 2,165,215 in 308 regions | 632,280 in 292 regions |

## Cells (`core.cohorts.load_cells(cohort, region_ids=None)`)

Cells concatenated region by region (in `region_ids` order; default all, label order):

| Key | dtype, shape | Meaning |
|---|---|---|
| `xy_um` | float64 (N, 2) | position in µm |
| `cell_type` | int16 (N,) | index into `type_names` (true type) |
| `region_ptr` | int64 (R+1,) | region r's cells are `[region_ptr[r], region_ptr[r+1])` |
| `region_ids` | list[str] (R,) | |
| `type_names` | list[str] (K,) | every type in the cohort, sorted; codes are stable across calls |

Cached as `data/<cohort>_cells.npz` after the first read.

## Feature grammar (`core.grammar`), frozen

- Focal types T: the cohort's tumour types (name starts with "Tumor"). Neighbour types: all K types.
- Radii `RADII_UM = (25, 50, 100)`; statistics `("fraction", "enrichment")`.
- Feature index `c = ((t*K + j)*3 + r)*2 + s`; `C = T*K*3*2`; names from `feature_names(g)`.
- Per focal cell: `fraction = n_j(r) / n(r)` (0 when `n(r) = 0`, self excluded, distance `<= r`);
  `enrichment = fraction / p_j`, `p_j` = share of type j among the region's visible cells (0 when 0).
- Patient feature = mean over all the patient's focal cells of type t, regions pooled (focal-cell
  weighted). A patient with no focal cell of type t gets NaN for that t.
- `grammar_hash(g)` = SHA-256 of the canonical JSON of `grammar(type_names)` (includes `PIXEL_UM`).

| Cohort | K | T | C | grammar sha256 |
|---|---|---|---|---|
| UPMC | 17 | 6 | **612** | `adc93e69df4280df4ebc40432d4b3ca84ffb4d8e83513ae1ce883feff981915a` |
| Charville | 15 | 7 | **630** | `772f6a845266470456af96f92d0573cc1bec72432ad74d789439baff066c2156` |

`make summary` prints these; a change to the grammar changes the hash, and every saved run records it.

## Neighbour lists (`core.truth.neighbours(xy_um)`), per region, local indices

| Key | dtype, shape | Meaning |
|---|---|---|
| `idx` | int32 (E,) | neighbour's local index; each row sorted by (distance, index), self excluded |
| `dist` | float64 (E,) | distance in µm, `<= 100` |
| `ptr` | int64 (n+1,) | row i is `[ptr[i], ptr[i+1])` |
| `rad_end` | int64 (n, 3) | row i's neighbours within radius r are `[ptr[i], rad_end[i, r])` |

A kernel walks one row once and reads all three radii from `rad_end`.

## Region output: what `reimage/` produces and `stats/` consumes

One look at one region returns `(S, m)`:

- `S` float64 (T, K, 3, 2): over visible focal cells of observed type t, the sum of fraction (`s=0`)
  and enrichment (`s=1`) of observed neighbour type j within radius r.
- `m` int64 (T,): number of visible focal cells of observed type t.

`core.truth.region_sums(counts, types, visible, focal, K)` turns per-cell counts `(n, 3, K)` into
`(S, m)`; every re-imager uses it, so equal counts give bit-equal outputs.

Patient features: sum `S` and `m` over a patient's sampled regions to `S_p (P, T, K, 3, 2)` and
`m_p (P, T)`, then `core.grammar.features(S_p, m_p)` gives **X̂ float64 (P, C)**, rows in patient index
order (a study's resampled patients: its own row order, with the patient index kept alongside).
The truth layer is the same with full regions and true types: `core.truth.truth_region`.

## Reference re-imager (`core.reimage_ref.reimage_region`), the oracle

`reimage_region(nbr, xy_um, types, focal, K, crop=None, eps=0.0, M=None, u=None, rng=None) -> (S, m)`

- `crop = (x0, y0, s)` in µm: visible iff `x0 <= x < x0+s` and `y0 <= y < y0+s`; `None` = full region.
  `random_crop(xy, s, rng)` places one uniformly inside the region's bounding box.
- Mis-call: `M` (K, K) row-stochastic, default `confusion_matrix(K, eps)` (keep with 1−eps, else a
  uniform other type). `u` float64 (n,) uniform draws, one per cell: observed type = first k with
  `cumsum(M[true])[k] > u`. Kernels take the same `u` so they can match cell for cell.
- Focal cells and neighbours both use observed types; neighbours outside the crop are skipped.
- Guarantee (tested on the fixture and on one real region per cohort): `crop=None, eps=0` reproduces
  `truth_region` exactly (`np.array_equal`).

## Check

`make smoke` = `uv sync --dev` + `uv run pytest -q` on tiny invented fixtures, offline, no real data.
`core/test_spine.py` pins: truth reproduction at no crop and eps=0, crop = truth of the visible cells,
neighbour-list order and radius offsets, mis-call rate, grammar count and hash, both loaders.
