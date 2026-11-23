// The "Plan your study" panel (node demo/preflight.test.mjs; exit 1 on failure): every number it prints is a
// stored point of the two committed power files, a labelled interpolation between two of them, or nothing.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { chartSVG, design, estimate, panelTexts, scenarioText } from "./preflight.js";

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
assert.equal(t.simNote, "Simulated point: pointwise 95% Monte Carlo interval 22%–32%, 300 studies at 200 patients.");
// Prometheus #130 (1): the difference is stated, not judged; the formula's 80% lies outside [22%, 32%].
assert.equal(t.gap, "Formula 80% vs simulated 27% (gap 53 points).");
assert.equal(t.caveat, "");
assert.equal(t.src, "bench/results/power-at-scale.json · bench/results/vs-schoenfeld-300.json");

// The fixed scenario every number belongs to, from the files' own metadata.
assert.equal(scenarioText(data), "Fixed scenario: simulated on the empirical Charville source (159 patients, 38 events) " +
  "with synthetic survival outcomes; one planted effect of HR 2 per SD among 630 features; family-wise alpha 0.05. " +
  "Above 159 patients, studies resample the source patients. No clinical claim.");
assert.deepEqual([data.scenario.hr_per_sd, data.scenario.features_m, data.scenario.cohort], [curve.hr_per_sd, vs.features_m, "charville"]);
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
assert.equal(e250.note, "Interpolated: no study was run at 250 patients. Value and bounds (27%–38%) are joined " +
  "linearly between the simulated points at 240 and 300 patients; not a validated 95% interval at 250.");
assert.equal(e250.simulated, false);
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
assert.equal(off.showSimKey, false);
assert.match(off.chartLabel, /not simulated, so there is no simulated curve/);
assert.equal(off.caveat, "");
// Out of range for the control keeps its curve and its key, and says why there is no dot.
const c400 = panelTexts(data, { patients: 400, window: "core", miscall: 0 });
assert.equal(c400.showSimKey, true);
assert.match(c400.chartLabel, /400 patients is outside its simulated range/);

// The control design at 200: 60% [54, 66], the formula still 80%.
const c200 = panelTexts(data, { patients: 200, window: "core", miscall: 0 });
assert.deepEqual([c200.sim, c200.simTag, c200.gap],
  ["60%", "ESTIMATE (sim), control design", "Formula 80% vs simulated 60% (gap 20 points)."]);
assert.equal(c200.src, "bench/results/vs-schoenfeld-300.json");

// Prometheus #130 (1): at 80 patients the formula's 18% lies inside the control's [16%, 26%], so no direction is
// claimed, and the control's false-alarm rate there (11 of 100 null studies) is carried whenever it is chosen.
const c80 = panelTexts(data, { patients: 80, window: "core", miscall: 0 });
assert.equal(c80.gap, "Formula 18% vs simulated 21% (gap 3 points). The formula's value lies inside the simulated interval.");
const caveat80 = "Caveat: at 80 patients this design rejected a true null in 11 of 100 null studies (11.0%, 95% interval " +
  "5.6%–18.8%) against a 5% target, so error control was not established there.";
assert.equal(c80.caveat, caveat80);
assert.equal(c200.caveat, caveat80);
assert.equal(vs.control.points.find((p) => p.patients === 80).sim_fwer, 0.11);
// No cropped point's false-alarm interval sits above 5%, so the cropped design carries none.
assert.ok(crop.points.every((p) => p.fwer_lo <= 0.05));

// Near-certainty is never rounded up to 100%.
assert.equal(panelTexts(data, { patients: 600, window: "400um", miscall: 0.1 }).formula, ">99%");

// A slider stop with no formula value is a data error, not a blank.
assert.throws(() => panelTexts(data, { patients: 55, window: "400um", miscall: 0.1 }), /no formula value/);
console.log("preflight.test.mjs: ok");

// The chart carries both dots while a design is simulated at n, and only the formula's where it is not.
const svg = chartSVG(data, { patients: 200, window: "400um", miscall: 0.1 });
assert.equal((svg.match(/class="sdot"/g) ?? []).length, 1);
// An interpolated value is a ring, never the filled dot of a simulated point.
const svg250 = chartSVG(data, { patients: 250, window: "400um", miscall: 0.1 });
assert.equal((svg250.match(/class="sdot interp"/g) ?? []).length, 1);
assert.equal((svg250.match(/class="sdot"/g) ?? []).length, 0);
// No design, no simulated band, line or points.
assert.ok(!/class="(band|sim|pt)"/.test(chartSVG(data, { patients: 200, window: "400um", miscall: 0 })));
assert.equal((svg.match(/class="pt"/g) ?? []).length, crop.points.filter((p) => p.patients >= 50 && p.patients <= 600).length);
assert.ok(!/NaN|undefined/.test(svg));
assert.equal((chartSVG(data, { patients: 400, window: "core", miscall: 0 }).match(/class="sdot"/g) ?? []).length, 0);
console.log("preflight.test.mjs: chart ok");
