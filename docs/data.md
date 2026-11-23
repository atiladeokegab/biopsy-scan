# Data and licences

## CODEX cohorts

The cohorts come from Wu et al., *Nature Biomedical Engineering* (2022), Zenodo record
[13179600](https://zenodo.org/records/13179600), under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The full cell tables aren't in this
repository: `make data` downloads them (about 730 MB) into `data/`, which git ignores. Set
`DATA_DIR` to use another folder.

Some derived extracts are committed so the results and the demo can be checked without the
download. They are redistributed under the same CC BY 4.0 licence with the attribution above:

- **Cell-level extracts** (cell positions and types from Charville patients): `demo/app/cells.json`,
  `demo/data/tissue.json` and `demo/data/tma.json`.
- **Patient-level values:** some committed result files carry the cohorts' coded patient labels and
  follow-up times, as published in the Zenodo record.
- **Aggregates:** the per-design results under `atlas/results/` and the benchmark outputs under
  `bench/results/` hold summary statistics, not cells.

## Human Protein Atlas photos

The routine IHC photos in the demo (six in total; two appear in the closing frames of the race GIF
and in the Routine IHC screenshot on the Home page (README)) come from the
[Human Protein Atlas](https://v25.proteinatlas.org/) version 25, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They are cropped to the tissue core,
resized and, in the demo, shown with coloured overlays. Each image's source page is listed in
[`demo/workflow/img/hpa_manifest.json`](../demo/workflow/img/hpa_manifest.json).

Reference: Uhlén M et al., Tissue-based map of the human proteome, *Science* 347, 1260419 (2015).

## The software

The code is released under the [MIT licence](../LICENSE). The licence covers the original software
only; the data and images above keep their own licences.
