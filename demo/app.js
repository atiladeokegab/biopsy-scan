// The Preflight demo. Hero: the pattern (biomarker.json), "Run the studies" and the speed race (baseline.json),
// then the power and honesty results (claims.json). Below: explore a design (grid.json, tissue.json, four sliders).
// Every file is local, so a served checkout replays the committed results with no network (schemas in README.md).
import { createTissue } from "./tissue.js";
import { createSpeedRace } from "./race.js";
import { drawPatch, drawKM } from "./bio.js";
import { drawChart, pct, fovLabel } from "./chart.js";

const $ = (id) => document.getElementById(id);
const FOV_STEP = 100;
let FULL_SLIDER; // the fov slider's last stop means "full core"; set from tissue.region_um
const AXES = ["patients", "regions", "fov_um", "miscall"];
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
let grid, tissue, view3d = null, shown = null, biomarker = null;

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

// Every slider move: tissue and answer at once.
function update() {
  const want = readSliders();
  $("patients-out").value = want.patients;
  $("regions-out").value = want.regions;
  $("fov-out").value = fovLabel(want.fov_um);
  $("miscall-out").value = pct(want.miscall);
  $("miscall-key").textContent = `${pct(want.miscall)} of cells mis-called, drawn in another class's colour`;

  if (view3d) {
    const slabs = view3d.update(want);
    $("tissue-caption").textContent =
      `${slabs} of ${want.patients} patients, ` +
      (want.fov_um === "full" ? "whole core imaged" : `${want.regions} × ${want.fov_um} µm field${want.regions > 1 ? "s" : ""} each`);
  }

  const p = nearest(want);
  shown = p;
  $("power").textContent = pct(p.power);
  $("ident").textContent = pct(p.identification_rate);
  $("planted").textContent = `Planted pattern: HR ${grid.effect_hr_per_sd} per SD`;
  $("verdict").textContent = verdict(p.power);
  $("snap").textContent = AXES.every((k) => want[k] === p[k]) ? "" : `Nearest grid point: ${describe(p)}.`;
  $("chart-title").textContent = drawChart($("chart"), grid, p);

}

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

const pct1 = (v) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;
const tagOf = (el, label) => { el.textContent = label; el.classList.toggle("ok", label === "MEASURED"); };

// The hero and the results: each file on its own, so one missing file never blanks the others.
async function hero() {
  const [bl, cl, bm] = await Promise.allSettled(
    ["data/baseline.json", "data/claims.json", "data/biomarker.json"].map(getJSON));
  if (bl.status === "fulfilled") {
    const race = createSpeedRace($("speed-race"), bl.value, { reducedMotion });
    $("run").addEventListener("click", () => race.run());
  } else $("run").disabled = true;

  if (bm.status === "fulfilled") {
    biomarker = bm.value;
    drawPatch($("bio"), biomarker);
    tagOf($("bio-tag"), biomarker.source);
    if (biomarker.km) {
      drawKM($("km"), biomarker.km);
      $("km-hi").textContent = `${biomarker.km.high.events}/${biomarker.km.high.n} died`;
      $("km-lo").textContent = `${biomarker.km.low.events}/${biomarker.km.low.n} died`;
    } else $("km").closest(".km").hidden = true;
  }

  if (cl.status === "fulfilled") {
    const { power: pw, false_alarms: fa, maxT: mt } = cl.value;
    $("pw-n").textContent = pw.patients;
    $("pw-sim").textContent = pct(pw.sim);
    tagOf($("pw-sim-tag"), pw.label);
    $("pw-ci").textContent = `95% interval ${Math.round(pw.lo * 100)}–${Math.round(pw.hi * 100)}%, ${pw.n} simulated studies`;
    $("pw-formula").textContent = pct(pw.formula);
    $("pw-src").textContent = pw.source;
    $("fa-rate").textContent = pct1(fa.hits / fa.n);
    tagOf($("fa-tag"), fa.label);
    $("fa-detail").textContent = `${fa.hits} of ${fa.n} null studies, against a ${pct(fa.target)} target`;
    $("mt-p").textContent = `p = ${mt.p.toFixed(3)}`;
    $("mt-detail").textContent = `for the pattern after ${mt.tests.toLocaleString("en-GB")} tests (maxT): ${mt.verdict}.`;
    tagOf($("mt-tag"), mt.label);
    $("fa-src").textContent = `${fa.source} · ${mt.source}`;
    $("bio-cap").textContent =
      `Real Charville tissue (patient ${biomarker ? biomarker.patient : "—"}, ${biomarker ? biomarker.window_um : 500} µm square). ` +
      `The top of ${mt.tests.toLocaleString("en-GB")} patterns: z = ${mt.z.toFixed(2)}, maxT p = ${mt.p.toFixed(3)} after ${mt.tests.toLocaleString("en-GB")} tests. Exploratory, not a discovery.`;
  }
}

async function main() {
  hero();
  const [gr, ti] = await Promise.allSettled([getJSON("data/grid.json"), getJSON("data/tissue.json")]);
  if (gr.status === "rejected" || ti.status === "rejected") {
    // Opened as a file, both fail: the serve-over-HTTP message is the one that helps.
    $("error").hidden = false;
    $("error").innerHTML = ti.status === "rejected"
      ? "Cannot load tissue data. Serve over HTTP (<code>python3 -m http.server 8000 -d demo</code>) and reload."
      : "Cannot read grid.json. Run <code>python3 -I demo/check_data.py</code> and reload.";
    $("tissue-caption").textContent = "";
    $("verdict").textContent = "";
    return;
  }
  [grid, tissue] = [gr.value, ti.value];

  const prov = (d) => (d.source === "MEASURED" ? "MEASURED" : "SYNTHETIC");
  $("provenance").textContent = `${prov(tissue)} tissue · ${prov(grid)} grid`;
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
  for (const id of ["patients", "regions", "fov", "miscall"]) $(id).addEventListener("input", () => update());
  $("design").addEventListener("submit", (e) => e.preventDefault());

  // Open on a typical imaging design (two 400 µm fields, 10% mis-called) so the fields of view and the noise show at once.
  // Field of view: 200 µm up to the core's width in 100 µm steps, then one more stop for "full core".
  FULL_SLIDER = Math.floor(tissue.region_um / FOV_STEP) * FOV_STEP + FOV_STEP;
  Object.assign($("fov"), { min: 200, step: FOV_STEP, max: FULL_SLIDER });
  setSliders({ ...grid.cohort, fov_um: 400, miscall: 0.1 });
  update();
}

main();
let resizeT;
addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => biomarker && drawPatch($("bio"), biomarker), 200); });
