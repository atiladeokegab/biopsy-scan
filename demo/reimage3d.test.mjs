// The 3D re-imaging view's model: Preflight's mis-call and crop rules, the scene's timeline, and the page
// it lives on (node demo/reimage3d.test.mjs; exit 1 on failure).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const { miscall, randomCrop, inCrop, heightField, boundaryPoint, timeline, END, FOV_UM, EPS, WINDOWS, TYPES,
  clock, inView, captureTime, mountReimage, groupMatrix, callFrom, fitView, drawStudy } = await import("./reimage3d.js");
const THREE = await import("./vendor/three/three.module.min.js");

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
assert.match(label, /one patient in the Charville cohort \(colorectal, not head and neck\)\./, "the cohort is not the patient");
assert.match(label, /Imaging simulated by Preflight\./);
// The steps name the design the code draws.
const steps = html.match(/id="reimage-steps"[^>]*>([\s\S]*?)<\/ol>/)[1];
assert.ok(steps.includes(`${["", "one", "two"][WINDOWS]} ${FOV_UM} µm fields`) && steps.includes(`${EPS * 100}% chance`), "steps match the design");

// The display model is on screen: 15 measured types drawn as 3 colour groups, errors drawn only between groups.
const tissueJSON = JSON.parse(readFileSync(new URL("./data/tissue.json", import.meta.url), "utf8"));
const nTypes = Object.keys(tissueJSON.meta.display_classes).length;
const flat = (x) => x.replace(/<[^>]+>/g, "").replace(/\s+/g, " ");
assert.ok(flat(steps).includes(`${nTypes} measured cell types`) && flat(steps).includes(`${TYPES.length} colour groups`), "steps say types are grouped");
assert.ok(flat(steps).includes(`10% chance of a mis-call among the ${nTypes} types`), "one rate, the study's");
assert.match(flat(steps), /Only a mis-call into another colour group shows here\./);
// One idea per sentence: no semicolons or colon-joined clauses in the visible copy; one word per idea.
const section = flat(html.match(/<section class="reimage"[\s\S]*?<\/section>/)[0]);
assert.doesNotMatch(section, /;/, "no semicolon-joined ideas");
assert.doesNotMatch(section, /\blook\b|\bcore\b|SIMULATED/, "glossary: region, imaging, SYNTHETIC");
assert.match(section, /SYNTHETIC imaging/);
assert.match(html, /<span class="tag measured" id="reimage-tag">MEASURED cells<\/span>/, "MEASURED tag neither accent nor warning");
assert.doesNotMatch(section, /\bsmall\b/, "no size word the picture can contradict");
assert.match(html, /<p class="reimage-fallback" id="reimage-fallback" role="status">Loading the region's cells\.<\/p>/, "a loading line until the view starts");
assert.match(flat(steps), /each on a different region\. Both are drawn here on this one\. In this draw they see [^.]+\./, "says why the fields share a region"); assert.match(flat(steps), /In this draw that is [^.]+\./);
assert.match(html, /<h2 id="reimage-title">[^<]*fields[^<]*mis-call/, "the title states the answer");
assert.match(html, />Replay the fly-through</); assert.match(html, />Pause the fly-through</);
assert.match(html, /<img id="reimage-still"[^>]*src="img\/reimage-end\.jpg"[^>]*loading="lazy"[^>]*hidden/, "a GPU still waits for the no-WebGL case");
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
const fakeEl = () => { const e = { textContent: "", hidden: false, attrs: {}, cls: new Set(["measured"]), on: {}, firstChild: { textContent: "" },
  setAttribute(k, v) { this.attrs[k] = v; }, removeAttribute(k) { delete this.attrs[k]; }, addEventListener(t, f) { this.on[t] = f; } };
  e.classList = { remove: (k) => e.cls.delete(k), add: (k) => e.cls.add(k) }; return e; };
const page = () => { const els = {}; const $ = (id) => (els[id] ??= fakeEl());
  $("reimage-replay").hidden = $("reimage-pause").hidden = true; $("reimage-steps").children = [1, 2, 3, 4].map(fakeEl); return $; };
const synth = { source: "SYNTHETIC", patients: [{ id: "S-1", region: "r1", cells_in_region: 2, cells: [{ x: 0, y: 0, type: "tumour" }, { x: 9, y: 9, type: "stroma" }] }] };
let $ = page();
mountReimage($, synth, { reducedMotion: false, search: "", create: () => { throw new Error("no WebGL"); } });
assert.equal($("reimage-tag").textContent, "SYNTHETIC cells"); assert.ok(!$("reimage-tag").cls.has("measured"), "SYNTHETIC wears the warning border");
assert.ok($("reimage-replay").hidden && $("reimage-pause").hidden, "no dead controls without WebGL");
assert.ok($("reimage").hidden, "canvas hidden without WebGL");
assert.doesNotMatch($("reimage-credit").textContent, /WebGL/, "the credit does not repeat the fallback");
assert.match($("reimage-fallback").textContent, /SYNTHETIC/); assert.match($("reimage-fallback").textContent, /hardware acceleration|another browser/, "says how to recover");
assert.equal($("reimage-still").hidden, true, "no still of real cells over synthetic input");
$ = page(); $("reimage-still").hidden = true;
mountReimage($, tissueJSON, { reducedMotion: false, search: "", create: () => { throw new Error("no WebGL"); } });
assert.equal($("reimage-still").hidden, false, "measured input without WebGL: the GPU still shows in the well");
assert.match($("reimage-fallback").textContent, /rendered on a GPU/);
const study = drawStudy(tissueJSON);
assert.ok(study.seen > 0 && study.crossed > 0 && study.crossed < study.seen, "the draw sees cells and crosses some groups");
assert.equal($("reimage-seen").textContent, `${study.perField.map((v) => v.toLocaleString("en-GB")).join(" and ")} of its ${study.n.toLocaleString("en-GB")} cells`, "each field's count, shown without WebGL");
assert.equal($("reimage-crossed").textContent, `${study.crossed.toLocaleString("en-GB")} cells`);
let got; $ = page();
mountReimage($, tissueJSON, { reducedMotion: false, search: "?t3d=nope", create: (cv, t, o) => { got = o; return { replay() {}, toggle: () => false }; } });
assert.equal(got.freezeAt, null, "malformed t3d plays normally");
assert.match($("reimage-credit").textContent, /Zenodo 13179600/);
assert.ok(!$("reimage-replay").hidden && !$("reimage-pause").hidden, "controls revealed once WebGL runs");
assert.ok($("reimage-fallback").hidden, "the loading line goes once WebGL runs");
assert.ok($("reimage-steps").cls.has("playing"), "while it plays, the phone shows the current step by the canvas");
assert.match(html, /<li value="1">[\s\S]*<li value="4">/, "steps keep their numbers when others are hidden");
$ = page(); mountReimage($, tissueJSON, { reducedMotion: true, search: "", create: () => ({}) });
assert.ok($("reimage-replay").hidden, "reduced motion: no controls"); assert.ok(!$("reimage-steps").cls.has("playing"), "reduced motion: every step shows");

// The mis-calls drawn are Preflight's 15-type model seen through the 3 colour groups: a type in group g is called
// as another type with prob eps, uniform over the other 14, so it lands in group h != g with prob eps * n_h / 14.
const sizes = TYPES.map((g) => Object.values(tissueJSON.meta.display_classes).filter((v) => v === g).length);
assert.deepEqual(sizes, [7, 3, 5], "tumour 7, stroma 3, immune 5 source types");
const M = groupMatrix(sizes, EPS);
M.forEach((row) => assert.ok(Math.abs(row.reduce((a, b) => a + b) - 1) < 1e-12, "rows sum to 1"));
assert.ok(Math.abs(M[0][1] - 0.1 * 3 / 14) < 1e-12 && Math.abs(M[0][0] - (1 - 0.1 * 8 / 14)) < 1e-12, "tumour row");
assert.ok(Math.abs(M[1][1] - (1 - 0.1 * 12 / 14)) < 1e-12, "stroma row");
// Brute force: draw the 15-type model directly and compare the group-crossing rate.
const types15 = Object.values(tissueJSON.meta.display_classes);
for (let g = 0; g < 3; g++) {
  const src = types15.findIndex((v) => v === TYPES[g]); let cross = 0, cross15 = 0; const r2 = (() => { let a = 99 + g; return () => { // mulberry32: the float LCG it replaces lost bits past 2^53
    a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();
  for (let i = 0; i < 200000; i++) {
    if (callFrom(M[g], r2()) !== g) cross++;
    const u = r2(); if (u < EPS) { const others = types15.filter((_, k) => k !== src); if (others[Math.floor(r2() * 14)] !== TYPES[g]) cross15++; }
  }
  assert.ok(Math.abs(cross - cross15) / 200000 < 0.004, `group ${TYPES[g]}: projected ${cross} vs 15-type ${cross15}`);
}

// Each field's count is that field alone (the study's fields are on different regions, so never pooled).
study.crops.forEach((cr, i) => assert.equal(study.perField[i], tissueJSON.patients.length && study.cells.filter((c) => c.x >= cr[0] && c.x < cr[0] + cr[2] && c.y >= cr[1] && c.y < cr[1] + cr[2]).length));
assert.equal(study.seen, study.inside.reduce((a, b) => a + b), "seen counts the cells inside");

// Framing: each shot is fitted to its subject at the canvas's own aspect, so the 390 view is redrawn, not cropped.
const pts = [...Array(32)].map((_, i) => new THREE.Vector3(300 * Math.cos(i / 5), 20, 300 * Math.sin(i / 5)));
for (const aspect of [1280 / 600, 900 / 600, 390 / 340]) {
  const { target, dist } = fitView(pts, 1.0, -0.55, 34, aspect, 0.9);
  const cam = new THREE.PerspectiveCamera(34, aspect, 1, 1e5);
  cam.position.set(target.x + dist * Math.cos(1.0) * Math.sin(-0.55), target.y + dist * Math.sin(1.0), target.z + dist * Math.cos(1.0) * Math.cos(-0.55));
  cam.lookAt(target); cam.updateMatrixWorld(); cam.updateProjectionMatrix();
  const ndc = pts.map((p) => p.clone().project(cam));
  const edge = Math.max(...ndc.map((q) => Math.max(Math.abs(q.x), Math.abs(q.y))));
  assert.ok(edge <= 0.9 + 1e-6 && edge > 0.85, `aspect ${aspect.toFixed(2)}: subject fills the frame inside the margin (${edge.toFixed(3)})`);
}
console.log("reimage3d.test.mjs: ok");
