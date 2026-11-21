// "Can my study find it?": three linked views (tissue, race, answer) driven by four sliders.
// Data: data/grid.json, data/tissue.json, optional data/bench.json (schemas in README.md).
import { createTissue } from "./tissue.js";
import { createRace } from "./race.js";
import { drawChart, pct, fovLabel } from "./chart.js";

const $ = (id) => document.getElementById(id);
const FOV_STEP = 100;
let FULL_SLIDER; // the fov slider's last stop means "full core"; set from tissue.region_um
const AXES = ["patients", "regions", "fov_um", "miscall"];
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
let grid, tissue, bench = null, view3d = null, race, shown = null, raceTimer = 0;

// Position of value v along an axis list, in steps (fractional between neighbours); "full" sits at its own index.
function axisPos(list, v) {
  if (v === "full") return list.indexOf("full") >= 0 ? list.indexOf("full") : list.length;
  const nums = list.filter((x) => x !== "full");
  if (v <= nums[0]) return 0;
  for (let i = 1; i < nums.length; i++) {
    if (v <= nums[i]) return i - 1 + (v - nums[i - 1]) / (nums[i] - nums[i - 1]);
  }
  return nums.length - 1;
}

function nearest(want) {
  let best = null, bestD = Infinity;
  for (const p of grid.points) {
    const d = AXES.reduce((s, k) => s + (axisPos(grid.axes[k], want[k]) - axisPos(grid.axes[k], p[k])) ** 2, 0);
    if (d < bestD) { bestD = d; best = p; }
  }
  return best;
}

function readSliders() {
  const fov = +$("fov").value;
  return {
    patients: +$("patients").value,
    regions: +$("regions").value,
    fov_um: fov >= FULL_SLIDER ? "full" : fov,
    miscall: +$("miscall").value / 100,
  };
}

function setSliders(c) {
  $("patients").value = c.patients;
  $("regions").value = c.regions;
  $("fov").value = c.fov_um === "full" ? FULL_SLIDER : c.fov_um;
  $("miscall").value = Math.round(c.miscall * 100);
}

function verdict(power) {
  if (power >= 0.8) return "Yes. Most studies like yours find the pattern.";
  if (power >= 0.5) return "Maybe. Many studies like yours miss it.";
  return "Unlikely. Most studies like yours miss the pattern.";
}

// "≈1.8 regions per patient" when the cohort gives a mean; the chart still marks it on the whole-number line.
const cohortRegions = (c) =>
  c.regions_mean != null && c.regions_mean !== c.regions
    ? `≈${c.regions_mean} regions per patient, marked on the ${c.regions}-region line`
    : `${c.regions} region${c.regions > 1 ? "s" : ""} per patient`;

const describe = (p) =>
  `${p.patients} patients, ${p.regions} region${p.regions > 1 ? "s" : ""}, ${fovLabel(p.fov_um)}, ${pct(p.miscall)} mis-called`;

// Every slider move: tissue and answer at once; the race restarts once the slider settles.
function update({ restartRace }) {
  const want = readSliders();
  $("patients-out").value = want.patients;
  $("regions-out").value = want.regions;
  $("fov-out").value = fovLabel(want.fov_um);
  $("miscall-out").value = pct(want.miscall);

  if (view3d) {
    const slabs = view3d.update(want);
    $("tissue-caption").textContent =
      `${slabs} of ${want.patients} patients shown · ` +
      (want.fov_um === "full" ? "whole core imaged" : `${want.regions} × ${want.fov_um} µm field${want.regions > 1 ? "s" : ""} per patient`) +
      ` · ${pct(want.miscall)} mis-called`;
  }

  const p = nearest(want);
  const changed = p !== shown;
  shown = p;
  $("power").textContent = pct(p.power);
  $("ident").textContent = pct(p.identification_rate);
  $("verdict").textContent = verdict(p.power);
  $("snap").textContent = AXES.every((k) => want[k] === p[k]) ? "" : `Nearest grid point: ${describe(p)}.`;
  if (changed) {
    const a = $("answer");
    a.classList.remove("flash"); void a.offsetWidth; a.classList.add("flash");
  }
  $("chart-title").textContent = drawChart($("chart"), grid, p);

  if (restartRace || changed) {
    clearTimeout(raceTimer);
    raceTimer = setTimeout(runRace, restartRace ? 0 : 350);
  }
}

function runRace() {
  race.run({ studies: grid.studies_per_point, power: shown.power, bench, reducedMotion });
}

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

async function main() {
  try {
    [grid, tissue] = await Promise.all([getJSON("data/grid.json"), getJSON("data/tissue.json")]);
  } catch (err) {
    $("error").hidden = false;
    $("error").innerHTML =
      "Could not read the data files. Browsers block them from a double-clicked file; " +
      "from the repo root run <code>python3 -m http.server 8000 -d demo</code> and open " +
      "<code>http://localhost:8000</code>.";
    $("run").disabled = true;
    return;
  }
  bench = await getJSON("data/bench.json").catch(() => null); // absent until bench (#3) runs

  const synthetic = [tissue.source !== "MEASURED" && "tissue", grid.source !== "MEASURED" && "power grid"].filter(Boolean);
  if (synthetic.length) {
    $("synthetic").hidden = false;
    $("synthetic").textContent = `SYNTHETIC ${synthetic.join(" and ")} · placeholders until the real export lands`;
  }
  $("race-n").textContent = grid.studies_per_point.toLocaleString("en-GB");
  $("note").textContent =
    `HR ${grid.effect_hr_per_sd} per SD planted · ${grid.studies_per_point.toLocaleString("en-GB")} studies per point. ` +
    `White ring: your design. Green: our cohort (${cohortRegions(grid.cohort)}).` +
    (grid.note ? ` ${grid.note}` : "");

  try {
    view3d = createTissue($("tissue"), tissue, { reducedMotion });
    $("spin").setAttribute("aria-pressed", String(!view3d.spinning));
    $("spin").textContent = view3d.spinning ? "Pause rotation" : "Rotate";
    $("spin").addEventListener("click", () => {
      view3d.setSpin(!view3d.spinning);
      $("spin").setAttribute("aria-pressed", String(!view3d.spinning));
      $("spin").textContent = view3d.spinning ? "Pause rotation" : "Rotate";
    });
  } catch (err) {
    $("tissue-caption").textContent = "This browser could not start WebGL, so the 3D tissue is off. Everything else works.";
  }
  race = createRace($("cpu"), $("gpu"), { cpu: $("cpu-count"), gpu: $("gpu-count"), speed: $("race-speed") });

  for (const id of ["patients", "regions", "fov", "miscall"]) $(id).addEventListener("input", () => update({}));
  $("design").addEventListener("submit", (e) => { e.preventDefault(); update({ restartRace: true }); });
  let t;
  addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => update({ restartRace: true }), 200); });

  // Open on a typical imaging design (two 400 µm fields, 10% mis-called) so the fields of view and the noise show at once.
  // Field of view: 200 µm up to the core's width in 100 µm steps, then one more stop for "full core".
  FULL_SLIDER = Math.floor(tissue.region_um / FOV_STEP) * FOV_STEP + FOV_STEP;
  Object.assign($("fov"), { min: 200, step: FOV_STEP, max: FULL_SLIDER });
  setSliders({ ...grid.cohort, fov_um: 400, miscall: 0.1 });
  update({ restartRace: true });
}

main();
