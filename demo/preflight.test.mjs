// The "Preflight your study" panel (node demo/preflight.test.mjs; exit 1 on failure): every number it prints is a
// stored point of the two committed power files, a labelled interpolation between two of them, or nothing.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { design, estimate, panelTexts } from "./preflight.js";

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const data = read("./data/preflight.json");
const curve = read("../bench/results/power-at-scale.json");
const vs = read("../bench/results/vs-schoenfeld-300.json");
const crop = design(data, "400um", 0.1), core = design(data, "core", 0);

// The design point the pitch quotes: 27.0% [22.1, 32.4] simulated against the formula's 80.1%, straight from the files.
const at200 = vs.points.find((p) => p.patients === 200);
const t = panelTexts(data, { patients: 200, window: "400um", miscall: 0.1 });
assert.equal(t.sim, "27%");
assert.equal(t.simTag, "ESTIMATE: simulated");
assert.equal(t.formula, "80%");
assert.equal(t.formulaTag, "ANALYTIC (formula)");
assert.equal(t.simNote, "95% interval 22%–32%, 300 simulated studies at 200 patients");
assert.equal(t.gap, "The formula overstates power by 53 points here.");
assert.equal(data.formula.power["200"], at200.analytic_power);

// Every simulated point is shown exactly as stored, with its file's label.
for (const p of curve.points.filter((q) => q.patients >= 50 && q.patients <= 600)) {
  const e = estimate(crop, p.patients);
  assert.deepEqual([e.power, e.lo, e.hi, e.label], [p.power, p.power_lo, p.power_hi, curve.label]);
}
for (const p of vs.control.points.filter((q) => q.patients >= 50)) {
  const e = estimate(core, p.patients);
  assert.deepEqual([e.power, e.lo, e.hi, e.label], [p.sim_power, p.sim_lo, p.sim_hi, vs.control.label]);
}
// Every formula value the files store is reproduced.
for (const p of vs.points) if (data.formula.power[p.patients] !== undefined) {
  assert.ok(Math.abs(data.formula.power[p.patients] - p.analytic_power) < 1e-9, `formula at ${p.patients}`);
}

// Between points: linear, labelled, never outside its neighbours.
const e250 = estimate(crop, 250);
assert.equal(e250.label, "ESTIMATE (interpolated)");
assert.ok(Math.abs(e250.power - (0.31 + (0.41 - 0.31) / 6)) < 1e-12);
assert.match(e250.note, /between the simulated points at 240 and 300 patients/);
for (let n = 50; n <= 600; n += 10) {
  const e = estimate(crop, n), pts = crop.points;
  const i = pts.findIndex((p) => p.patients >= n), lo = pts[Math.max(i - 1, 0)], hi = pts[i];
  assert.ok(e.power >= Math.min(lo.power, hi.power) - 1e-12 && e.power <= Math.max(lo.power, hi.power) + 1e-12, `${n}`);
  assert.ok(e.lo <= e.power && e.power <= e.hi, `interval at ${n}`);
  assert.equal(e.label === "ESTIMATE (interpolated)", !pts.some((p) => p.patients === n), `label at ${n}`);
}

// No number where nothing was simulated: the control stops at 300 patients, and the off-diagonal pairs were never run.
assert.equal(estimate(core, 400).missing, "Not simulated: this design was run from 40 to 300 patients.");
const off = panelTexts(data, { patients: 200, window: "400um", miscall: 0 });
assert.equal(off.sim, "—");
assert.equal(off.simTag, "NOT SIMULATED");
assert.equal(off.formula, "80%");
assert.equal(design(data, "core", 0.1), null);

// The control design at 200: 60% [54, 66], the formula still 80%.
const c200 = panelTexts(data, { patients: 200, window: "core", miscall: 0 });
assert.deepEqual([c200.sim, c200.simTag, c200.gap],
  ["60%", "ESTIMATE (sim), control design", "The formula overstates power by 20 points here."]);

// A slider stop with no formula value is a data error, not a blank.
assert.throws(() => panelTexts(data, { patients: 55, window: "400um", miscall: 0.1 }), /no formula value/);
console.log("preflight.test.mjs: ok");
