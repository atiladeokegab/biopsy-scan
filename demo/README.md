# Demo: "Can my study find it?"

A static page: plain HTML, CSS and JS, no build step, no network calls. It reads one file,
`demo/data/grid.json`, and works with Wi-Fi off.

```bash
python3 -m http.server 8000 -d demo     # then open http://localhost:8000
```

Browsers block `fetch` from `file://`, so double-clicking `index.html` shows an error card with
this command instead of the page.

The grid shipped here is **SYNTHETIC** (`demo/synth_grid.py`, a toy attenuation model). The page
labels it SYNTHETIC on screen until the atlas (#5) replaces the file with `"source": "MEASURED"`.

Check a grid file: `python3 -I demo/check_grid.py [path]` (exit 0 when valid). The repo's pytest
collects only `core/` and `tests/`, so this check is not in `make smoke` yet.

## `grid.json` schema (`preflight-grid/1`)

One file, written by the atlas (#5). The headline comes from bench (#3); until it is measured,
leave `status` as `UNMEASURED` and the page shows a placeholder.

```jsonc
{
  "schema": "preflight-grid/1",
  "source": "MEASURED",               // or "SYNTHETIC": shown on screen as a warning
  "note": "free text shown under the chart",
  "studies_per_point": 1000,          // re-imaged studies behind each point
  "effect_hr_per_sd": 2.0,            // the planted effect every point uses
  "cohort": {                         // "our cohort", marked on the chart
    "name": "Charville", "patients": 162, "regions": 2, "fov_um": "full", "miscall": 0.0
  },
  "headline": {
    "status": "MEASURED",             // or "UNMEASURED" (then the seconds may be null)
    "studies": 1000,
    "gpu_seconds": 4.2,               // end to end, one design point, on the GPU
    "cpu_seconds": 3600               // same algorithm, all CPU cores
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

Rules the checker enforces: every point's values are on `axes`; no duplicate design points;
`power` and `identification_rate` are in 0..1 and identification never exceeds power.

## How the page picks a point

The sliders cover patients 20 to 300, regions 1 to 8, field of view 250 to 2000 µm plus full,
and mis-call 0 to 30%. The button finds the grid point nearest to the sliders, measuring each axis
in steps along its own `axes` list (so 1 to 2 regions counts the same as 4 to 8), and says when the
point it shows differs from what was asked. The chart draws power against patients, one line per
`regions` value, at the chosen point's field of view and mis-call rate, with the cohort marked.
