// The 3D re-imaging view's model: Preflight's mis-call and crop rules, the scene's timeline, and the page
// it lives on (node demo/reimage3d.test.mjs; exit 1 on failure).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const { miscall, randomCrop, inCrop, heightField, boundaryPoint, timeline, END, FOV_UM, EPS, WINDOWS, TYPES,
  clock, inView, captureTime, mountReimage } = await import("./reimage3d.js");

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
assert.ok(steps.includes(`${["", "one", "two"][WINDOWS]} ${FOV_UM} µm fields`) && steps.includes(`${EPS * 100}% chance`), "steps match the design");

// The display model is on screen: 15 measured types drawn as 3 colour groups, errors drawn only between groups.
const tissueJSON = JSON.parse(readFileSync(new URL("./data/tissue.json", import.meta.url), "utf8"));
const nTypes = Object.keys(tissueJSON.meta.display_classes).length;
const flat = (x) => x.replace(/<[^>]+>/g, "").replace(/\s+/g, " ");
assert.ok(flat(steps).includes(`${nTypes} measured cell types`) && flat(steps).includes(`${TYPES.length} colour groups`), "steps say types are grouped");
assert.match(flat(steps), /mis-call inside one group does not show here/);
assert.doesNotMatch(flat(steps), /mis-calls 10% of the cells/, "no exact achieved fraction");
const aria = html.match(/<canvas id="reimage"[^>]*aria-label="([^"]+)"/)[1];
assert.match(aria, /10% chance/); assert.match(aria, /colour groups/);
assert.match(html, /id="reimage-replay"[^>]*hidden/, "controls start hidden"); assert.match(html, /id="reimage-pause"[^>]*hidden/);

// Playback: time stops while the scene is off screen and resumes where it left off.
let ms = 0; const c = clock(() => ms);
assert.equal(c.t(), 0, "not started before it is seen");
c.show(true); ms = 5000; assert.equal(c.t(), 5);
c.show(false); ms = 25000; assert.equal(c.t(), 5, "off screen: held");
c.show(true); assert.equal(c.t(), 5, "back on screen: resumes, no jump"); ms = 26000; assert.equal(c.t(), 6);
assert.equal(c.toggle(), false); ms = 30000; assert.equal(c.t(), 6, "paused");
c.show(false); c.show(true); ms = 31000; assert.equal(c.t(), 6, "paused stays paused across a scroll");
assert.equal(c.toggle(), true); ms = 32000; assert.equal(c.t(), 7);
c.replay(); assert.equal(c.t(), 0); ms = 33000; assert.equal(c.t(), 1);
const still = clock(() => ms, { held: END, playing: false }); still.show(true); ms = 99000; assert.equal(still.t(), END, "reduced motion: last frame");
assert.equal(inView({ isIntersecting: true, intersectionRatio: 0.1 }), false, "a sliver is not in view");
assert.equal(inView({ isIntersecting: true, intersectionRatio: 0.5 }), true);

// Capture time: only a finite number holds the scene.
for (const bad of [null, "", "nope", "NaN", "Infinity", " "]) assert.equal(captureTime(bad), null, `t3d=${bad}`);
assert.equal(captureTime("9.8"), 9.8); assert.equal(captureTime("0"), 0);

// Setup: provenance first, controls only after WebGL starts, an honest fallback without it.
const fakeEl = () => { const e = { textContent: "", hidden: false, attrs: {}, cls: new Set(["ok"]), on: {}, firstChild: { textContent: "" },
  setAttribute(k, v) { this.attrs[k] = v; }, removeAttribute(k) { delete this.attrs[k]; }, addEventListener(t, f) { this.on[t] = f; } };
  e.classList = { remove: (k) => e.cls.delete(k), add: (k) => e.cls.add(k) }; return e; };
const page = () => { const els = {}; const $ = (id) => (els[id] ??= fakeEl());
  $("reimage-replay").hidden = $("reimage-pause").hidden = true; $("reimage-steps").children = [1, 2, 3, 4].map(fakeEl); return $; };
const synth = { source: "SYNTHETIC", patients: [{ id: "S-1", region: "r1", cells_in_region: 2, cells: [{ x: 0, y: 0, type: "tumour" }, { x: 9, y: 9, type: "stroma" }] }] };
let $ = page();
mountReimage($, synth, { reducedMotion: false, search: "", create: () => { throw new Error("no WebGL"); } });
assert.equal($("reimage-tag").textContent, "SYNTHETIC cells"); assert.ok(!$("reimage-tag").cls.has("ok"), "no MEASURED styling");
assert.ok($("reimage-replay").hidden && $("reimage-pause").hidden, "no dead controls without WebGL");
assert.ok($("reimage").hidden, "canvas hidden without WebGL");
assert.match($("reimage-credit").textContent, /WebGL/); assert.match($("reimage-fallback").textContent, /SYNTHETIC/);
let got; $ = page();
mountReimage($, tissueJSON, { reducedMotion: false, search: "?t3d=nope", create: (cv, t, o) => { got = o; return { replay() {}, toggle: () => false }; } });
assert.equal(got.freezeAt, null, "malformed t3d plays normally");
assert.equal($("reimage-tag").textContent, "MEASURED cells"); assert.match($("reimage-credit").textContent, /Zenodo 13179600/);
assert.ok(!$("reimage-replay").hidden && !$("reimage-pause").hidden, "controls revealed once WebGL runs");
$ = page(); mountReimage($, tissueJSON, { reducedMotion: true, search: "", create: () => ({}) });
assert.ok($("reimage-replay").hidden, "reduced motion: no controls");
console.log("reimage3d.test.mjs: ok");
