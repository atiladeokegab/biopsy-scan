// Step 3's two numbers come from the merged preflight.json, not the code (node demo/workflow/step3.test.mjs).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
const { powerAt } = await import("./step3.js");
const pf = JSON.parse(readFileSync(new URL("../data/preflight.json", import.meta.url)));
const p = powerAt(pf);
assert.equal(Math.round(p.analytic * 100), 80);
assert.equal(Math.round(p.sim * 100), 27);
assert.equal(p.simLabel, "ESTIMATE: simulated");
assert.ok(p.analyticLabel.startsWith("ANALYTIC"));
console.log("step3 ok");
