# The Preflight demo

One dark page with three linked views, driven by four sliders: patients, regions per patient,
field of view and mis-call rate. It uses plain HTML, CSS and ES-module JS, with no build step and
no network calls. three.js r170 is vendored in `vendor/three/` (MIT, licence alongside), so the
page works with Wi-Fi off.

```bash
make demo                               # checks demo/data, then serves; open http://localhost:8000
python3 -m http.server 8000 -d demo     # the same, without the data check
make tissue                             # fetches the data, then regenerates demo/data/tissue.json
```

Browsers block `fetch` and modules from `file://`. If you double-click `index.html`, the page
shows this command instead.

1. **Tissue (3D).** The first six patients in `tissue.json` are shown as thin slabs of cells, coloured
   tumour, stroma or immune, three by two. Each core turns slowly in place; drag it, or focus it and
   use the arrow keys. The scene is sized in core widths, so 720 µm real cores and 2,500 µm synthetic
   ones both draw as fine dots.
   - The field-of-view squares drop onto each slab, one per region per patient. Cells outside
     every square dim.
   - The mis-call slider recolours that fraction of cells at random, so you can see the noise.
2. **The race.** The same N studies run on the CPU (one at a time) and on the GPU (all at once).
   Each tile is one study: green if it found the pattern, grey if it missed. The share of green
   tiles is the grid point's power.
   - Timing comes only from `bench.json`. Without a MEASURED bench, both lanes still play at a
     fixed display pace that is not to scale, both say UNMEASURED, and no seconds or ratio are
     shown.
   - With one, both lanes run on one clock: the GPU lane finishes in 1.4 s on screen, and the CPU
     lane is slowed by the measured ratio. The line under the lanes names the hardware from
     `bench.json` ("GPU: <gpu>, 4.2 s · CPU: all cores of <cpu>, 1 h · 857× faster").
3. **The answer.** Power and identification rate for the grid point nearest the sliders, and the
   one chart: power against patients, one line per regions value. The chart marks the current
   design (white ring) and our cohort (green).

Check the data: `python3 -I demo/check_data.py [data_dir]` exits 0 when every file present is
valid. The repo's pytest collects only `core/` and `tests/`, so this check is not in
`make smoke` yet.

Regenerate the placeholders: `python3 -I demo/synth_data.py`. It writes the **SYNTHETIC**
`grid.json` (a toy attenuation model) and `tissue.json` (random tumour nests with an immune rim).
The page labels both SYNTHETIC until each file says `"source": "MEASURED"`.

The committed `grid.json` is Atlas's Charville grid, copied verbatim from `atlas/results/production/charville/grid.json`
at commit 79a7792 (public on PR #45; sha256 e0581681…): 48 designs, 1,000 planted-effect studies each, simulated on real tissue
with SYNTHETIC Weibull outcomes (HR 2 per SD). It is the legacy R4 Cox score power, not the PD1 pattern and not the
R15 screen, and the page says so. Its metric label is `status` / `labels.power` (ESTIMATE; outcomes SYNTHETIC), shown as "ESTIMATE simulation ·
SYNTHETIC outcomes"; its `source: MEASURED` is kept only for compatibility and never shown as the label. For a MEASURED grid, `check_data` also requires each power to equal its count over
the studies and sit inside its interval, and the note to name the synthetic outcomes, legacy R4 and not R15.

## `data/tissue.json` (`preflight-tissue/1`): written from the core loaders

```jsonc
{
  "schema": "preflight-tissue/1",
  "source": "MEASURED",                // or "SYNTHETIC": shown on screen as a warning
  "note": "free text",
  "region_um": 2500,                   // side of the square every patient's region is drawn in
  "patients": [                        // one slab each, drawn in this order (about 12 is plenty)
    { "id": "CHV-017",
      "cells": [                       // subsample to about 3k cells per patient
        { "x": 812, "y": 1450, "type": "tumour" }   // microns from the region corner; tumour | stroma | immune
      ] }
  ]
}
```

## `data/grid.json` (`preflight-grid/1`): written by the atlas (#5)

```jsonc
{
  "schema": "preflight-grid/1",
  "source": "MEASURED",               // or "SYNTHETIC"
  "note": "free text shown under the chart",
  "studies_per_point": 1000,          // simulated studies behind each point; also the race's tile count
  "effect_hr_per_sd": 2.0,            // the planted effect every point uses
  "cohort": {                         // "our cohort", marked on the chart
    "name": "Charville", "patients": 159, "regions": 2,   // regions: the whole-number line it is marked on
    "regions_mean": 1.8,              // optional: shown as "≈1.8 regions per patient"
    "fov_um": "full", "miscall": 0.0
  },
  "axes": {                           // every value a point may take, ascending
    "patients": [20, 40, 80, 160, 300],
    "regions": [1, 2, 4, 8],
    "fov_um": [250, 500, 1000, 2000, "full"],   // microns; "full" = the whole region, always last
    "miscall": [0.0, 0.1, 0.2, 0.3]             // fraction of cells mis-called
  },
  "points": [                         // one per design point; the grid may be sparse
    { "patients": 160, "regions": 4, "fov_um": 500, "miscall": 0.1,
      "power": 0.62,                  // share of studies where the maxT test rejects
      "identification_rate": 0.48 }   // share where it rejects AND the top hit is the planted pattern
  ]
}
```

The checker enforces these rules:
- Every point's values are on `axes`.
- There are no duplicate design points.
- `power` and `identification_rate` are in 0..1, and identification never exceeds power.

## `data/bench.json` (`preflight-bench/1`): written by bench (#3), optional

```jsonc
{
  "schema": "preflight-bench/1",
  "status": "MEASURED",               // or "UNMEASURED"; a missing file also means UNMEASURED
  "studies": 1000,                    // studies in the timed design point
  "cpu": "AMD EPYC 7B13, 8 vCPU",     // required: the CPU model the CPU time was measured on
  "gpu": "NVIDIA L4",                 // required: the GPU model the GPU time was measured on
  "cpu_seconds": 3600,                // end to end, same algorithm, all CPU cores
  "gpu_seconds": 4.2                  // end to end on the GPU
}
```

## `data/baseline.json` (`preflight-baseline/1`): the speed race, filled by `collect_race.py`

Four rows, all timing the same `task`: one design point, 1,000 simulated studies on real Charville, end to end
(demo-plan-v3, revision 16:58Z). Never edit a time by hand: `python3 -I demo/collect_race.py` (run by `make demo`
and by Pages before every deploy) fills rows from `bench/results/*.json`. Only build-modal's `"kind": "design_point"`
file (#75's `gpu/modal_app.py`; slots `a_lab_all_cores`, `b_ours_cpu_all_cores`, `c_ours_1_gpu`,
`d_ours_n_gpu.runs["16"]`) fills the four rows (ids `lab-cpu`, `preflight-cpu`, `gpu-1`, `gpu-16`), and only when it
says `"status": "MEASURED"`, `"studies": 1000` and carries its `scope`; a FAILED run, a 32-study trial or a generic
`race` list is refused. `seconds` is in-container compute (`gpu-16`: the measured compute span across containers);
start-up goes in `startup_s` and the threshold-and-calls step in `calls_s`, each printed apart and never counted.
GPU lanes need `e_gpu_parity.exact_all_containers`. The file's `scope` (WORKFLOW comparison, EXPLORATORY
calibration) is copied to `baseline.json` and printed above the race; `check_data` refuses timed rows without it.
A row no file feeds keeps its committed value.

```jsonc
{
  "schema": "preflight-baseline/1",
  "task": "One design point: 1,000 simulated studies on real Charville tissue, end to end",
  "rows": [{
    "id": "lab-cpu",             // unique; bench files address rows by id
    "method": "The lab way: a shuffle test in every study, all CPU cores",
    "role": "baseline",          // "baseline" (grey bar) or "screen" (Preflight, accent bar)
    "label": "PENDING",          // MEASURED, ESTIMATE, SYNTHETIC, UNMEASURED or PENDING
    "seconds": null,             // compute seconds; null when PENDING or UNMEASURED, a positive number otherwise
    "startup_s": null,           // optional, shown apart
    "hardware": null,            // required when timed
    "cores": null,               // a positive integer, or "gpu": "1x NVIDIA H100"; one is required when timed
    "gpu": null,
    "source": "build-modal",     // always required
    "note": null                 // optional
  }],
  "workflow": {                  // optional footnote: different work, seconds only, never a bar or a ratio
    "method": "Standard Cox screen of all 3,780 candidates (statsmodels PHReg)",
    "seconds": 63.84, "label": "MEASURED", "hardware": "AMD EPYC 7B13", "cores": 8, "gpu": null,
    "source": "bench/results/impl-a-charville-amd-epyc-7b13.json", "note": "298.3 s on 1 core",
    "fingerprints": {"patient_rows": "...", "X": "...", "C": "..."}  // copied from the file, to match later screens
  }
}
```

`interim` (optional): measured groups shown until the design point is timed, each racing on its own scale,
from the Modal runner's `"kind": "ladder"` file (#75): one study, NumPy against 1 GPU in the same container, and
16 studies on 1 against 4 GPU containers, compute only with start-up apart. Today it was filled by
`python3 -I demo/collect_race.py <path to #75's bench/results/modal-a100-20261121T171213Z.json>`; once #75
merges, Pages refills it from the tree.

The checker rejects a number without a label or source, a PENDING or UNMEASURED row with a number, and checks
every field on its own (`seconds` and `startup_s` positive finite numbers, `cores` a positive integer, text
fields non-empty strings). The page escapes every string it inserts. On screen each timed bar fills at its
rate (the slowest timed row takes 10 s) and prints its real seconds at the finish; a PENDING row has no bar and
never moves. No ratio is printed anywhere: the R-14 Cox screen produces different output from the race's
unit, so it is only the workflow footnote, in seconds.

## `data/claims.json` (`preflight-claims/1`): written by `collect_race.py`

The power result at the demo's 200 patients (simulated, its 95% interval, and the Schoenfeld formula) with
`power.model`, which names its simulated design (a planted HR 2 per SD among 630 features) and says it is not
the PD1 pattern's power; the false-alarm count pooled over `false_alarms.designs` null designs (a pooled
summary, not per-design error control); and the maxT result, each with its `label` and `source`
(`bench/results/vs-schoenfeld-300.json`, `results/charville-v2.json`). `claims.js` checks every number it
renders and throws on a bad file; `app.js` draws each panel on its own, so one bad file leaves only its panel
on the placeholder.

## `data/biomarker.json` (`preflight-biomarker/1`): build-8's file from #82, copied verbatim

The KM split at the cohort median (`km.arms[].curve`, `n`, `events`, `p_value: null`) and one real region per
arm (the arm's median patient; columnar `cells`, `counts`). The page leads with the curves and the death counts,
shows `km.label` ("exploratory, cutoff chosen on the same cohort, no p-value") beside them, and states each
region's counts. `check_data.py` refuses a file without `exploratory: true`, with a p-value, or without "no
p-value" in its label. The hit cannot be tested in our second cohort (UPMC has no PD1 stain), and the page says so; it never says "not replicated".

## How the page picks a grid point

The slider ranges are:
- patients 20 to 300 in steps of 1, starting at the cohort (159);
- regions 1 to 8;
- field of view 200 µm up to the core width (`tissue.region_um`) in 100 µm steps, plus full core;
- mis-call 0 to 30%.

The page picks the grid point nearest the sliders, measuring each axis in steps along its own
`axes` list, so going from 1 to 2 regions counts the same as going from 4 to 8. It says when the
point it shows differs from what was asked. The chart uses that point's field of view and
mis-call rate.

The page opens at our cohort's patients and regions, with two 400 µm fields and 10% mis-called, so
the fields of view and the noise are visible from the first second.

## Open: 60 fps on the demo laptop (lesson 27)

The 3D view draws about 30,000 points in 12 draw calls. It has only been rendered with software
WebGL (SwiftShader) on a VM with no GPU, at about 2 fps. Its frame rate on the demo laptop's real
GPU is not yet measured; check it there before the demo.
