# Demo: "Can my study find it?"

One dark page with three linked views, driven by four sliders: patients, regions per patient,
field of view and mis-call rate. It uses plain HTML, CSS and ES-module JS, with no build step and
no network calls. three.js r170 is vendored in `vendor/three/` (MIT, licence alongside), so the
page works with Wi-Fi off.

```bash
make demo                               # checks demo/data, then serves; open http://localhost:8000
python3 -m http.server 8000 -d demo     # the same, without the data check
make tissue                             # regenerates demo/data/tissue.json (needs `make data`)
```

Browsers block `fetch` and modules from `file://`. If you double-click `index.html`, the page
shows this command instead.

1. **Tissue (3D).** Each patient is one thin slab of cells, coloured tumour, stroma or immune, and
   the slabs are stacked. The view orbits slowly; drag it, or focus it and use the arrow keys.
   The patients slider sets how many slabs show (one per 25 patients, up to the patients in
   `tissue.json`).
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
