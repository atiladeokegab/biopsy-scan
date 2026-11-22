// The 3D re-imaging view's model: Preflight's mis-call and crop rules, the scene's timeline, and the page
// it lives on (node demo/reimage3d.test.mjs; exit 1 on failure).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const { miscall, randomCrop, inCrop, heightField, boundaryPoint, timeline, END, FOV_UM, EPS, WINDOWS } = await import("./reimage3d.js");

// miscall follows core/reimage_ref.py: keep the type with prob 1 - eps, else a uniform other type,
// chosen as the first k with cumsum(M[type])[k] > u. Rows for eps 0.1, K 3: [.9 .05 .05], [.05 .9 .05].
assert.deepEqual([0.5, 0.92, 0.97].map((u) => miscall(0, u, 0.1, 3)), [0, 1, 2]);
assert.deepEqual([0.01, 0.5, 0.96].map((u) => miscall(1, u, 0.1, 3)), [0, 1, 2]);
assert.deepEqual([0.01, 0.06, 0.5].map((u) => miscall(2, u, 0.1, 3)), [0, 1, 2]);
assert.equal(miscall(2, 0.999, 0, 3), 2); // eps 0: nothing changes
let changed = 0;
for (let i = 0; i < 100000; i++) changed += miscall(i % 3, (i * 0.6180339887) % 1, 0.1, 3) !== i % 3;
assert.ok(Math.abs(changed / 100000 - 0.1) < 0.002, `mis-call rate ${changed / 100000}`);

// randomCrop follows random_crop: placed uniformly inside the bounding box, null when the side covers it.
assert.deepEqual(randomCrop([0, 0], [720, 700], 400, [0, 1]), [0, 300, 400]);
assert.deepEqual(randomCrop([10, 20], [730, 720], 400, [0.5, 0.5]), [170, 170, 400]);
assert.equal(randomCrop([0, 0], [720, 720], 720, [0.3, 0.3]), null);
assert.ok(inCrop(0, 300, [0, 300, 400]) && !inCrop(400, 300, [0, 300, 400]) && inCrop(5, 5, null)); // x0 <= x < x0 + s

// Height is local cell density: highest over the dense cluster, zero far from every cell.
const cells = [...Array(200)].map((_, i) => ({ x: 100 + (i % 10), y: 100 + Math.floor(i / 10) % 10, type: "tumour" }))
  .concat([{ x: 600, y: 600, type: "stroma" }]);
const h = heightField(cells, [0, 0], [720, 720], 48, 20);
assert.equal(h.max, 1);
assert.ok(h.at(105, 105) > 0.9 && h.at(600, 600) < 0.1 && h.at(400, 400) === 0);

// The camera's cell-level stop is where tumour meets the rest: between the two clusters, not inside either.
const split = [...Array(400)].map((_, i) => ({ x: (i % 20) * 18, y: Math.floor(i / 20) * 18, type: (i % 20) < 10 ? "tumour" : "stroma" }));
const [bx] = boundaryPoint(split, [0, 0], [342, 342], 24, 20);
assert.ok(bx > 120 && bx < 220, `boundary at x ${bx}`);

// Timeline: fly in, hold at cell level, back out, windows land, mis-calls flicker, then rest.
const at = (t) => timeline(t);
assert.equal(at(0).step, 0);
assert.equal(at(5).step, 1);
assert.equal(at(9.5).step, 2);
assert.equal(at(12).step, 3);
assert.equal(at(END).step, 3);
assert.equal(at(0).windows, 0);
assert.equal(at(END).windows, 1);
assert.equal(at(END).dim, 1);
assert.equal(at(END).settled, true);
assert.equal(at(1e9).camera.r, at(END).camera.r); // clamps after the end
assert.ok(at(5).camera.r < at(0).camera.r / 5, "cell level is at least 5x closer than the whole core");
// Flicker stays under 3 flashes per second (WCAG 2.3.1): count colour changes over the flicker.
let flips = 0, prev = at(0).called, t0 = null, t1 = null;
for (let t = 0; t <= END; t += 0.01) {
  const c = at(t).called;
  if (c !== prev) { flips++; t0 ??= t; t1 = t; }
  prev = c;
}
assert.ok(flips >= 4, `flicker changes ${flips}`);
assert.ok(flips / 2 / (t1 - t0) <= 3, `flash rate ${flips / 2 / (t1 - t0)} per s`);

// The page: the section and its labels exist, and the proof take's anchors are untouched.
const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
for (const id of ["speed-race", "speed-interim", "run", "verdict", "tissue-caption", "reimage", "reimage-steps"])
  assert.equal(html.split(`id="${id}"`).length, 2, `exactly one #${id}`);
assert.ok(html.indexOf('id="speed-race"') < html.indexOf('id="reimage"'), "the 3D view sits below the race");
const label = html.match(/id="reimage-label"[^>]*>([\s\S]*?)<\/p>/)[1].replace(/<[^>]+>/g, "").replace(/\s+/g, " ");
assert.match(label, /Real cell positions and types/);
assert.match(label, /Charville \(colorectal, not head and neck\)/);
assert.match(label, /imaging simulated by Preflight/);
// The steps name the design the code draws.
const steps = html.match(/id="reimage-steps"[^>]*>([\s\S]*?)<\/ol>/)[1];
assert.ok(steps.includes(`${["", "one", "two"][WINDOWS]} ${FOV_UM} µm fields`) && steps.includes(`${EPS * 100}% of the cells`), "steps match the design");
console.log("reimage3d.test.mjs: ok");
