// The Preflight demo. Hero: the pattern (biomarker.json, #82), "Run the studies" and the speed race (baseline.json),
// then the power and honesty results (claims.json). Below: explore a design (grid.json, tissue.json, four sliders).
// Every file is local, so a served checkout replays the committed results with no network (schemas in README.md).
import { createTissue } from "./tissue.js";
import { mountReimage } from "./reimage3d.js";
import { createSpeedRace, workflowHTML } from "./race.js";
import { drawRegion, drawKM } from "./bio.js";
import { claimTexts, bioTexts, gridTexts, dialRanges } from "./claims.js";
import { drawChart, pct, fovLabel } from "./chart.js";
import { panelTexts, chartSVG, scenarioText } from "./preflight.js";

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
  if (power >= 0.8) return "Yes. Most studies like yours find the planted effect.";
  if (power >= 0.5) return "Maybe. Many studies like yours miss it.";
  return "Unlikely. Most studies like yours miss the planted effect.";
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
  $("verdict").textContent = verdict(p.power);
  $("snap").textContent = AXES.every((k) => want[k] === p[k]) ? "" : `Nearest grid point: ${describe(p)}.`;
  $("chart-title").textContent = drawChart($("chart"), grid, p);

}

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

const tagOf = (el, label) => { el.textContent = label; el.classList.toggle("ok", label === "MEASURED"); };

const drawRegions = () => biomarker?.regions.forEach((r) => drawRegion($(`reg-${r.arm}`), r));

// One panel's bad file never stops the others: it stays on its placeholder and the reason goes to the console.
function panel(name, draw) {
  try { draw(); return true; } catch (e) { console.warn(`${name} panel not drawn:`, e.message); return false; }
}

// The hero and the results: each file on its own, so one missing file never blanks the others.
async function hero() {
  const [bl, cl, bm] = await Promise.allSettled(
    ["data/baseline.json", "data/claims.json", "data/biomarker.json"].map(getJSON));
  if (bl.status === "fulfilled") panel("race", () => {
    const d = bl.value, races = [];
    // Until the design point is timed, the measured pairs (#75's ladder file) race, each on its own scale.
    if (!d.rows.some((r) => r.seconds > 0 && !["PENDING", "UNMEASURED"].includes(r.label)) && d.interim?.length) {
      $("speed-title").textContent = "How fast, measured so far";
      $("speed-dp").hidden = false;
      for (const g of d.interim) {
        const head = document.createElement("p"), group = document.createElement("div"), lanes = document.createElement("div");
        head.className = "sr-head"; head.title = g.unit; head.textContent = g.title;
        group.className = "sr-group"; group.append(lanes);
        if (g.scope_line) { // Atlas's cleared scope line, beside its bars (numbers from the file, via collect_race.py)
          const scope = document.createElement("p");
          scope.className = "sr-scope"; scope.textContent = g.scope_line; group.append(scope);
        }
        $("speed-interim").append(head, group);
        races.push(createSpeedRace(lanes, g, { reducedMotion }));
      }
    } else $("speed-task").textContent = d.task;
    races.push(createSpeedRace($("speed-race"), d, { reducedMotion }));
    $("workflow").innerHTML = workflowHTML(d.workflow);
    $("run").addEventListener("click", () => races.forEach((r) => r.run()));
  }) || ($("run").disabled = true);
  else $("run").disabled = true;

  if (bm.status === "fulfilled") panel("biomarker", () => {
    const b = bm.value, texts = bioTexts(b);
    const arm = (n) => b.km.arms.find((a) => a.name === n);
    drawKM($("km"), { high: { steps: arm("high").curve }, low: { steps: arm("low").curve } });
    $("km-note").textContent = b.km.label.replace(/^exploratory,\s*/, "");
    for (const n of ["high", "low"]) $(`km-${n.slice(0, 2)}`).textContent = `${arm(n).events} of ${arm(n).n} died`;
    for (const [id, v] of Object.entries(texts)) $(id).textContent = v;
    tagOf($("bio-tag"), b.source);
    biomarker = b;
    drawRegions();
  });


  if (cl.status === "fulfilled") panel("claims", () => {
    const { power: pw, false_alarms: fa, maxT: mt } = cl.value;
    for (const [id, v] of Object.entries(claimTexts(cl.value))) $(id).textContent = v;
    tagOf($("pw-sim-tag"), pw.label); tagOf($("fa-tag"), fa.label); tagOf($("mt-tag"), mt.label);
  });

}

// "Preflight your study": every control redraws the answer from data/preflight.json at once; no network after load.
async function preflight() {
  const data = await getJSON("data/preflight.json");
  const form = $("pf-form");
  $("pf-scenario").textContent = scenarioText(data);
  const read = () => ({ patients: +$("pf-patients").value, window: form.elements["pf-window"].value,
    miscall: +form.elements["pf-miscall"].value });
  const draw = () => panel("preflight", () => {
    const s = read(), t = panelTexts(data, s);
    $("pf-patients-out").value = s.patients;
    for (const k of ["sim", "formula", "gap", "src"]) $(`pf-${k}`).textContent = t[k];
    $("pf-sim-note").textContent = t.simNote;
    $("pf-caveat").textContent = t.caveat; $("pf-caveat").hidden = !t.caveat;
    $("pf-key-sim").hidden = !t.showSimKey; $("pf-chart").setAttribute("aria-label", t.chartLabel);
    tagOf($("pf-sim-tag"), t.simTag); tagOf($("pf-formula-tag"), t.formulaTag);
    // Drawn at its on-screen width, so the axis text stays 12 px on a phone.
    const W = Math.max(300, Math.round($("pf-chart").getBoundingClientRect().width) || 560);
    $("pf-chart").setAttribute("viewBox", `0 0 ${W} 220`);
    $("pf-chart").innerHTML = chartSVG(data, s, W);
  });
  form.addEventListener("input", draw);
  addEventListener("resize", draw);
  form.addEventListener("submit", (e) => e.preventDefault());
  draw();
}

async function main() {
  hero();
  preflight().catch((e) => console.warn("preflight panel not drawn:", e.message));
  const [gr, ti] = await Promise.allSettled([getJSON("data/grid.json"), getJSON("data/tissue.json")]);
  if (gr.status === "rejected" || ti.status === "rejected") {
    // Opened as a file, both fail: the serve-over-HTTP message is the one that helps.
    $("error").hidden = false;
    const fix = ti.status === "rejected"
      ? "Serve over HTTP (<code>python3 -m http.server 8000 -d demo</code>) and reload."
      : "Run <code>python3 -I demo/check_data.py</code> and reload.";
    $("error").innerHTML = (ti.status === "rejected" ? "Cannot load tissue data. " : "Cannot read grid.json. ") + fix;
    $("tissue-caption").textContent = "";
    $("verdict").textContent = "";
    $("reimage").hidden = true;
    $("reimage-still").hidden = false;
    $("reimage-fallback").innerHTML = `<span>The 3D view is off: the page could not load its data. ${fix} Below: the last frame, rendered on a GPU.</span>`; // the fix in place, not a pointer to it
    return;
  }
  [grid, tissue] = [gr.value, ti.value];

  for (const [id, v] of Object.entries(gridTexts(grid))) $(id).textContent = v;
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
  panel("reimage", () => mountReimage($, tissue, { reducedMotion, search: location.search }));
  for (const id of ["patients", "regions", "fov", "miscall"]) $(id).addEventListener("input", () => update());
  $("design").addEventListener("submit", (e) => e.preventDefault());

  // Open on a typical imaging design (two 400 µm fields, 10% mis-called) so the fields of view and the noise show at once.
  // Field of view: 200 µm up to the core's width in 100 µm steps, then one more stop for "full core".
  FULL_SLIDER = Math.floor(tissue.region_um / FOV_STEP) * FOV_STEP + FOV_STEP;
  Object.assign($("fov"), { min: 200, step: FOV_STEP, max: FULL_SLIDER });
  for (const [id, range] of Object.entries(dialRanges(grid))) Object.assign($(id), range);
  setSliders({ ...grid.cohort, fov_um: 400, miscall: 0.1 });
  update();
}

main();
let resizeT;
addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(drawRegions, 200); });
