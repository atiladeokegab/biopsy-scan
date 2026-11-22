// Steps 3 and 4: the numbers come from preflight.json, the simulated look mis-calls about 10% of the cells in view and
// only those, and step 4's pair is a real tma.json marker (node demo/workflow/steps.test.mjs; exit 1 on failure).
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
const { powerAt, look, EPS } = await import("./step3.js");
const { PICKED } = await import("./step4.js");
const pf = JSON.parse(readFileSync(new URL("../data/preflight.json", import.meta.url)));
const p = powerAt(pf);
assert.equal(Math.round(p.analytic * 100), 80);
assert.equal(Math.round(p.sim * 100), 27);
assert.equal(p.simLabel, "ESTIMATE: simulated");
assert.ok(p.analyticLabel.startsWith("ANALYTIC"));
// A synthetic core, so the check runs before tma.json is merged: 30x30 cells on a 900 µm square, types 0..14.
const fake = { size_um: [900, 900], cells: Array.from({ length: 900 }, (_, i) => [(i % 30) * 30, Math.floor(i / 30) * 30, i % 15]) };
const tmaUrl = new URL("../data/tma.json", import.meta.url);
const tma = existsSync(tmaUrl) ? JSON.parse(readFileSync(tmaUrl)) : null;
for (const core of [fake, tma?.cores[tma.pick]].filter(Boolean)) {
  const L = look(core), seen = L.inside.filter(Boolean).length;
  const wrong = core.cells.filter(([, , t], j) => L.called[j] !== t).length;
  assert.ok(core.cells.every(([, , t], j) => L.inside[j] || L.called[j] === t), "mis-calls only inside a field");
  assert.ok(seen > 0 && Math.abs(wrong / seen - EPS) < 0.05, `mis-call rate ${wrong}/${seen}`);
}
if (tma) assert.ok(PICKED.markers.every((m) => tma.markers.includes(m)));
console.log("steps 3-4 ok");
