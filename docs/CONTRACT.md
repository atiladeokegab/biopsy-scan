# Contract

Biopsy Scan's shared shapes. Written by the lead's agent; every area builds against it; only the lead
changes it. `core/` owns the data and grammar,
`reimage/` the kernels, `stats/` the statistic.

## Assignment

Re-image thousands of simulated spatial-biomarker studies cell by cell from real CODEX tissue (UPMC,
Charville), so a lab can see, before imaging a slide, which study designs (patients, regions per
patient, field of view, mis-call rate) can find a survival-linked spatial pattern.

There is no backend service, app factory, store or gateway: Biopsy Scan is a Python library plus saved
runs. `core/` is the shared core; other areas import it and never re-implement it.

## Data

- Zenodo record 13179600 (CC-BY-4.0; cite Wu et al., Nat. Biomed. Eng. 2022). `make data`
  (`scripts/fetch_data.sh`) puts the label CSVs and raw zips in `data/`, never committed
  (`data/.gitignore` is `*`). `LOCAL_SRC=<dir>` links local copies instead.
- Everything in `data/` is untrusted: parsed as CSV only; scripts that read it run with `python -I`.
- Coordinates are pixels in the files; `core.cohorts.PIXEL_UM = 0.377` converts to µm. Every distance
  here is µm.

## Cohort (`core.cohorts.load(cohort) -> dict`)

Patient index p in `0..P-1`, region index r in `0..R-1`, cell index in `0..n-1`.

| Key | dtype, shape | Meaning |
|---|---|---|
| `cell_type` | int8 [n] | index into `type_names` (true type) |
| `x`, `y` | float32 [n] | position in µm |
| `region_id` | int32 [n] | region index of each cell; cells are grouped by region, in region order |
| `patient_of_region` | int32 [R] | patient index of each region |
| `time` | float64 [P] | follow-up, days |
| `event` | bool [P] | death |
| `strata` | int32 [P] | coverslip group (below) |
| `type_names` | list[str] [K] | every type in the cohort, sorted |
| `region_names` | list[str] [R] | e.g. `UPMC_c001_v001_r001_reg004` |
| `patient_names` | list[str] [P] | case numbers |

`cells` (the reimage input) is this dict; only `cell_type, x, y, region_id, patient_of_region` are
required (without `type_names`, K = max type + 1 and every type is focal). Only patients with a verified identity, an endpoint and an unambiguous stratum are in it
(one aligned patient mask; their regions and cells go with them).

| | UPMC | Charville |
|---|---|---|
| Verified identity | case number from the sample label; regions labelled `NA` are dropped | case number (sample label before `_`) |
| P / regions / events | **34 / 125 / 6** (discovery side, the default); full source before the lock: 50 / 188 / 10 | **159** / 286 / **38** |
| Endpoint | any-cause death, `survival_day` | death, last contact − surgery, in days |
| `strata` | coverslip group: 0 = c001–c003, 1 = c004–c007 (4 / 30 patients on the discovery side, 7 / 43 in the full source; no case crosses) | coverslip c001–c004 → 0–3 |
| Excluded | 86 discovery-side `NA` regions; the 97 locked held-out regions (`core/upmc_locked.json`) are never loaded | 24135, 24136 (no dates); 24180 (on c002 and c003) |

UPMC is read through the locked held-out split (`core/heldout.py`): every loader defaults to
`split="discovery"`. `audit/results/upmc.json` and the UPMC entry of `audit/results/summary.json` predate
the lock: they are historical full-source evidence (50 / 188 / 10), not the discovery default.

**Charville is the patient-level cohort.** UPMC tuple-matched patients (one patient per clinical
tuple: 79 patients, 27 deaths in the full source, `split="all"`; 81 when the 2 tuples holding two case numbers are split) are
**provisional and descriptive only**: `core.cohorts.load_regions("upmc", identity="tuple")`. They are
never used for patient-level inference or the regions-per-patient axis (one all-`NA` tuple spans both
coverslip groups).

## Feature grammar (`core.grammar`), frozen

- Focal types: the cohort's tumour types (name starts with `Tumor`), T of them. Neighbour types: all K.
- Radii **25, 50, 100 µm**; statistics `fraction`, `enrichment`.
- Feature c = `((t*K + j)*3 + r)*2 + s`; C = T·K·3·2. The ordered list is `feature_names(grammar(type_names))`,
  each `"<focal type>|<neighbour type>|<radius>um|<stat>"`.
- Per visible focal cell i of (observed) type t: `n(r)` = visible cells at distance `<= r`, self
  excluded; `n_j(r)` = those of observed type j; `fraction = n_j(r)/n(r)` (0 when `n(r) = 0`);
  `enrichment = fraction / p_j`, `p_j` = share of type j among the region's visible cells (0 when `p_j = 0`).
- Patient feature = mean over all the patient's sampled focal cells of type t, sampled regions pooled
  (focal-cell weighted).
- **Missing focal type: X stays NaN** where a patient has no focal cell of type t in the sampled
  regions. Never filled with zeros. `stats/` handles it per feature as complete case (patients with a
  finite value for that feature).
- `grammar_hash(g)` = SHA-256 of the canonical JSON of `grammar(type_names)` (includes `PIXEL_UM`).

| Cohort | K | T | C | grammar sha256 |
|---|---|---|---|---|
| UPMC | 17 | 6 | **612** | `adc93e69df4280df4ebc40432d4b3ca84ffb4d8e83513ae1ce883feff981915a` |
| Charville | 15 | 7 | **630** | `772f6a845266470456af96f92d0573cc1bec72432ad74d789439baff066c2156` |

`make summary` prints these; every saved run records the hash.

## Design and the random stream

`design = dict(patients, regions_per_patient, fov_um, eps, seed)`:

- `patients`: `None` = every patient once, in index order; an int n = n patients drawn with
  replacement; or an int array of patient indices.
- `regions_per_patient`: `None` = all of a patient's regions; int k = k drawn without replacement
  (all when the patient has fewer than k).
- `fov_um`: `None` = **full field**; float s = a square field of view of side s µm per sampled region.
- `eps`: mis-call rate; the confusion matrix is `confusion_matrix(K, eps)` (keep the type with 1 − eps,
  else a uniform other type).
- `seed`: the design point's seed; callers derive study seeds from it.

One study with seed `seed` uses `rng = np.random.default_rng(seed)` in exactly this order:
1. if `patients` is an int: `idx = rng.integers(0, P, patients)`;
2. for each drawn patient in order: if k is set and below its region count,
   `rng.choice(its_regions_sorted, k, replace=False)`;
3. for each (drawn patient, sampled region) in order: if `fov_um` is set, `random_crop` draws
   `rng.random(2)`; then `u = rng.random(n_cells_of_region)` drives the mis-call.

`core.reimage_ref.draw(cells, design, seed)` returns these draws, so kernels can share them.

## Reference re-imager (`core.reimage_ref`), the oracle

`reimage(cells, design, seed, region_kernel=reimage_region) -> X float64 [P, C]`: one study. The
wrapper owns patient and region sampling and the RNG order (above), and calls
`region_kernel(nbr, xy, types, focal, K, crop=crop, M=M, u=u) -> (S, m)` once per sampled region in
draw order, so numba and GPU kernels drop in with identical sampling. Rows follow the drawn patients
(`patients=None`: patient index order). With `patients=None, regions_per_patient=None, fov_um=None,
eps=0` it equals the truth layer `core.truth.truth(cells)` exactly (`np.array_equal`, NaN in the same places).

Per region it calls `reimage_region(nbr, xy, types, focal, K, crop, eps, u=u) -> (S float64 [T,K,3,2],
m int64 [T])`: sums of fraction (s=0) and enrichment (s=1) over visible focal cells, and the focal
count. A patient's row is `features(sum of S over its sampled regions, sum of m)`. Crop
`(x0, y0, s)`: visible iff `x0 <= x < x0+s` and `y0 <= y < y0+s`. Mis-call: observed type = first k
with `cumsum(M[true])[k] > u`. Focal cells and neighbours both use observed types; neighbours outside
the crop are skipped.

Neighbour lists (`core.truth.neighbours(xy)`, per region, local indices): `idx int32 [E]` sorted by
(distance, index) per row, self excluded; `dist float64 [E]` (`<= 100`); `ptr int64 [n+1]`;
`rad_end int64 [n, 3]` (row i within radius r is `[ptr[i], rad_end[i, r])`). A kernel walks a row once.

## Kernels (`reimage/`)

`reimage.run(design, seeds) -> X float64 [S, P, C]`: the cohort comes in `design["cells"]` (the
cells dict above) and the backend in optional `design["backend"]` in `{"numpy", "cpu", "gpu"}`
(default `"numpy"`). Study s equals `core.reimage_ref.reimage(design["cells"], design, seeds[s],
region_kernel=<backend's region kernel>)`: same dtype, same NaN policy, same draws, matched cell for
cell. A batched backend may run many studies per launch, but it must reproduce these draws.

## Statistics (`stats/`)

- `stats.score(X, time, event, strata) -> {"U", "V", "z"}`: X float64 [P, C] (or [S, P, C]), stratified
  log-rank score per feature at β = 0, complete case per feature; U, V, z float64 [C] (or [S, C]).
- `stats.maxT(z) -> float` (or [S]): max over features of |z|.
- `stats.threshold(design, null_studies) -> float`: the design point's maxT threshold from null studies.

## Check

`make smoke` = `uv sync --dev` + `uv run pytest -q` over `core/ stats/ reimage/ bench/ audit/ atlas/ tests/`, tiny invented
fixtures, offline, no real data.
