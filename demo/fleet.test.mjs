// The race's fleet pair (lead, 19:00Z): 1 A100 against 8, with Atlas's cleared line built from the rows it shows.
// node demo/fleet.test.mjs; exit 1 on failure.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fleetPair } from "./fleet.js";

const bl = JSON.parse(readFileSync(new URL("./data/baseline.json", import.meta.url), "utf8"));
const fleet = bl.interim.find((g) => g.rows.some((r) => r.id === "fleet-1"));
const p = fleetPair(fleet);
assert.deepEqual(p.rows.map((r) => r.id), ["fleet-1", "fleet-8"], "1 A100 against 8: no 2-, 4- or 16-GPU bar");
// Atlas's cleared wording, exactly as in #75's body
assert.equal(p.line, "MEASURED, observed 6.6× compute-span reduction 843.7→128.2 s for 1,000 Charville " +
  "re-imaging/feature-generation studies, 8 A100-SXM4-40GB vs 1, one sweep; outputs bit-identical; inference outside timing");
const without = (id) => ({ ...fleet, rows: fleet.rows.filter((r) => r.id !== id) });
assert.equal(fleetPair(without("fleet-8")), null, "no 8-GPU run: no pair, no line");
assert.equal(fleetPair(without("fleet-1")), null);
const swap = (id, o) => ({ ...fleet, rows: fleet.rows.map((r) => (r.id === id ? { ...r, ...o } : r)) });
assert.equal(fleetPair(swap("fleet-8", { gpu: "8x NVIDIA A100-SXM4-80GB" })), null, "one card model on both sides");
assert.equal(fleetPair(swap("fleet-8", { label: "ESTIMATE" })), null, "both MEASURED");
assert.equal(fleetPair({ ...fleet, title: "1000 studies on a fleet" }), null, "bit-identical outputs stated by the data");
assert.equal(fleetPair(undefined), null);
console.log("fleet.test.mjs: ok");
