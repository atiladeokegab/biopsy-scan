// node demo/plan/plan.test.mjs (exit 1 on failure): every power the planner prints is a stored grid point, a labelled
// straight line between two of them, or "Not measured".
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { plan, pct, scenario } from "./plan.js";

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const data = read("./data.json");
const grid = read("../../atlas/results/production/charville/grid.json");

// A stored point is copied exactly from grid.json.
const g = grid.points.find((p) => p.patients === 159 && p.regions === 2 && p.fov_um === 400);
const r = plan(data, { cohort: "charville", patients: 159, regions: 2, fov: 400, miscall: 0.1 });
assert.equal(r.sim.power, g.power);
assert.deepEqual([r.sim.lo, r.sim.hi], [g.power_lo, g.power_hi]);
assert.match(r.sim.label, /^SIMULATED/);
assert.match(r.sim.source, /^atlas\/results\/production\/charville\/grid\.json @ [0-9a-f]{7}$/);
assert.match(r.formula.label, /^ESTIMATE/);
// Atlas #153: rejection and identification are told apart, and the interval carries its Monte Carlo scope.
assert.equal(r.sim.identification, g.identification_rate);
assert.match(r.sim.note, /rejects in 19\.0% \(95% Monte Carlo interval, conditional on the source cohort, acquisition bank and realized calibration: 17%–22%\) and names the planted feature in 8\.7%\.$/);
assert.match(plan(data, { cohort: "charville", patients: 318, regions: 2, fov: 400, miscall: 0.1 }).sim.note, /Above 159 source patients, simulated cohorts resample them\.$/);

// Halfway between two simulated patient counts: the mean of the two stored values, labelled ESTIMATE.
const a = grid.points.find((p) => p.patients === 80 && p.regions === 2 && p.fov_um === "full");
const b = grid.points.find((p) => p.patients === 159 && p.regions === 2 && p.fov_um === "full");
const mid = plan(data, { cohort: "charville", patients: 119.5, regions: 2, fov: "full", miscall: 0.1 });
assert.ok(Math.abs(mid.sim.power - (a.power + b.power) / 2) < 1e-12);
assert.match(mid.sim.label, /^ESTIMATE/);
assert.match(mid.formula.missing, /^Not computed/); // formula only at whole patient counts

// Off the grid in any axis: no number at all.
for (const off of [{ regions: 3 }, { fov: 350 }, { miscall: 0.05 }, { patients: 500 }, { patients: 40 }]) {
  const x = plan(data, { cohort: "charville", patients: 159, regions: 2, fov: 400, miscall: 0.1, ...off });
  assert.equal(x.sim.power, undefined, JSON.stringify(off));
  assert.match(x.sim.missing, /^Not measured: nobody simulated/);
}
// UPMC has its own axes.
assert.equal(plan(data, { cohort: "upmc", patients: 50, regions: 8, fov: 1000, miscall: 0.1 }).sim.label, data.labels.sim);

// Formula matches the committed Schoenfeld comparison at the same patient count and m.
const vs = read("../../bench/results/vs-schoenfeld.json");
const p80 = vs.points.find((p) => p.patients === 80);
assert.ok(Math.abs(plan(data, { cohort: "charville", patients: 80, regions: 2, fov: 400, miscall: 0.1 }).formula.power - p80.analytic_power) < 1e-12);

// Stains are not measured yet and say why.
assert.match(r.stains.missing, /^No panel recommendation\. The search .*ran on the UPMC discovery split \(34 patients, 6 events\).*UNMEASURED: C_full <= 0.5 \(0.2941\)/);
assert.equal(pct(0.0004), "<1%");
assert.equal(pct(0.996), ">99%");
assert.match(scenario(data, "charville"), /159 patients, 38 events.*630 features.*No clinical claim\.$/);
console.log("plan.test: ok");
