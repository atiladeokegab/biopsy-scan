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
and by Pages before every deploy) fills rows from `bench/results/*.json`. A bench file feeds a row with a
top-level `race` list, e.g.
`"race": [{"id": "gpu-1", "seconds": 4.2, "startup_s": 90.0, "label": "MEASURED", "hardware": "Modal", "gpu": "1x NVIDIA H100"}]`
(ids: `lab-cpu`, `preflight-cpu`, `gpu-1`, `gpu-16`). `seconds` is compute time; container start-up goes in
`startup_s`, printed apart and never counted. The GPU model is whatever the file says. A row no file feeds
keeps its committed value.

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

The checker rejects a number without a label or source, a PENDING or UNMEASURED row with a number, and checks
every field on its own (`seconds` and `startup_s` positive finite numbers, `cores` a positive integer, text
fields non-empty strings). The page escapes every string it inserts. On screen each timed bar fills at its
rate (the slowest timed row takes 10 s) and prints its real seconds at the finish; a PENDING row has no bar and
never moves. No ratio is printed anywhere: the R-14 Cox screen produces different output from the race's
unit, so it is only the workflow footnote, in seconds.

## `data/claims.json` (`preflight-claims/1`): written by `collect_race.py`

The power result at the demo's 200 patients (simulated, its 95% interval, and the Schoenfeld formula), the
false-alarm rate pooled over every null study, and the maxT result, each with its `label` and `source`
(`bench/results/vs-schoenfeld-300.json`, `results/charville-v2.json`).

## `data/patch.json` (`preflight-patch/1`): written by `scripts/export_patch.py`

One real Charville 500 µm patch (`cells`: `x`, `y` in µm, `k` one of `t5`, `tumour`, `mac`, `mac_pd1`,
`immune`, `other`), chosen by `window_rule` for the most PD1+ macrophages near Tumor 5, the 100 µm
`radius_um`, how a macrophage is drawn PD1+ (`pd1_rule`), and `km`: the cohort's Kaplan-Meier steps split at
the median of the per-patient feature value (or null), with `note`, shown beside the curves: "exploratory,
cutoff chosen on the same cohort, no p-value". Needs the raw data:
`DATA_DIR=... uv run python -I scripts/export_patch.py`. `data/biomarker.json` (build-8, #82) is a separate
file with two whole regions, one per arm; this page does not read it.

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
