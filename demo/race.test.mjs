// Regression: the race's speed line and when it plays (node demo/race.test.mjs; exit 1 on failure).
import assert from "node:assert/strict";

// Just enough browser for createRace: tokens resolve to their own names, frames run only when pumped.
globalThis.document = { documentElement: {} };
globalThis.getComputedStyle = () => ({ getPropertyValue: (name) => name });
globalThis.devicePixelRatio = 1;
let frames = [];
globalThis.requestAnimationFrame = (f) => frames.push(f);
globalThis.cancelAnimationFrame = () => { frames = []; };
const pump = (now) => { const f = frames; frames = []; f.forEach((cb) => cb(now)); };

const { speedWord, raceAction, createRace } = await import("./race.js");

assert.equal(speedWord(1, 2), "GPU 2.0× slower"); // the bug: this used to print "1× faster"
assert.equal(speedWord(2, 2), "about the same");
assert.equal(speedWord(2.04, 2), "about the same");
assert.equal(speedWord(8.4, 2), "GPU 4.2× faster");
assert.equal(speedWord(3600, 4.2), "GPU 857× faster");

// No run before the click: page load and slider moves never play the race; only the button does.
assert.equal(raceAction({ pressed: false, changed: true }), "clear"); // load, or a slider moved to a new design
assert.equal(raceAction({ pressed: false, changed: false }), null); // a slider moved within the same design
assert.equal(raceAction({ pressed: true, changed: false }), "run");
assert.equal(raceAction({ pressed: true, changed: true }), "run");

// Lanes on stub canvases: the last colour painted at each tile position.
const canvas = () => {
  const tiles = new Map();
  const g = { setTransform() {}, fillRect(x, y) { tiles.set(`${x},${y}`, this.fillStyle); } };
  return { clientWidth: 200, clientHeight: 100, getContext: () => g, tiles };
};
const colours = (c) => new Set(c.tiles.values());
const cpu = canvas(), gpu = canvas(), out = { cpu: {}, gpu: {}, speed: {} };
const race = createRace(cpu, gpu, out);
const design = { studies: 100, power: 0.5, bench: null, reducedMotion: false };

race.clear(100); // what the page does on load
pump(1e9);
assert.deepEqual(colours(gpu), new Set(["--surface-elevated"]), "before the click the lanes are empty");
assert.equal(out.gpu.innerHTML, "<b>0</b> of 100 studies");

race.run(design); // the button
pump(0); pump(1e9);
assert.ok(colours(gpu).has("--accent") && colours(cpu).has("--accent"), "the click plays the race");

race.run(design);
pump(0); pump(700); // mid-race, then a slider moves to a new design
race.clear(100);
pump(1e9);
assert.deepEqual(colours(gpu), new Set(["--surface-elevated"]), "a slider change clears the lanes");
assert.deepEqual(colours(cpu), new Set(["--surface-elevated"]));
assert.equal(out.cpu.innerHTML, "<b>0</b> of 100 studies");

race.redraw(); // a resize after a clear keeps the lanes empty
pump(1e9);
assert.deepEqual(colours(gpu), new Set(["--surface-elevated"]), "a resize never plays the race");
// The speed race: bars fill at their measured rate; PENDING rows never move and never show a time.
const { raceState, RACE_MS } = await import("./race.js");
const { readFileSync } = await import("node:fs");
const bl = JSON.parse(readFileSync(new URL("./data/baseline.json", import.meta.url), "utf8"));
const rows = [{ seconds: 60 }, { seconds: 15 }, { seconds: null }];
assert.deepEqual(raceState(rows, 0).map((r) => r.frac), [0, 0, 0]);
const mid = raceState(rows, RACE_MS / 4);
assert.equal(mid[0].frac, 0.25); assert.equal(mid[0].clock, 15); // the slowest row: a quarter done at 15 real s
assert.ok(mid[1].done && mid[1].clock === 15, "a 4x faster row finishes at a quarter of the race, showing its real 15 s");
assert.ok(raceState(rows, 1e9).every((r, i) => (rows[i].seconds == null ? r.frac === 0 && r.clock === null : r.done)));
bl.rows.forEach((r) => assert.ok(r.seconds == null || (r.label && r.source), `${r.id}: a time needs a label and a source`));
// #79 review (Atlas): no ratio anywhere in the race or its workflow footnote, every string escaped,
// and a row whose seconds are not a positive finite number is drawn as untimed.
const { speedRaceHTML, workflowHTML, pairRatio } = await import("./race.js");
const row = (o) => ({ id: "x", method: "m", role: "screen", label: "MEASURED", hardware: "h", cores: 8, gpu: null, source: "s", ...o });
const both = speedRaceHTML({ rows: [row({ role: "baseline", seconds: 2 }), row({ seconds: 1 }), row({ seconds: 2 })] });
assert.doesNotMatch(both, /×|faster|slower|speed-?up/i, "two MEASURED rows never print a ratio, slower or equal");
const wf = workflowHTML({ method: "Cox screen", seconds: 63.84, label: "MEASURED", hardware: "EPYC", cores: 8, source: "f.json" });
assert.match(wf, /^Workflow comparison, not part of the race: .*63\.8 s/);
assert.doesNotMatch(wf, /×|faster|slower|speed-?up/i, "the workflow footnote is seconds only");
assert.equal(workflowHTML(null), "");
const evil = speedRaceHTML({ rows: [row({ seconds: 1, gpu: "<x-g>", hardware: "<x-h>", note: "<x-n>", method: "<x-m>", label: "<x-l>", source: "<x-s>" })] });
assert.doesNotMatch(evil, /<x-/, "every dynamic string is escaped");
assert.doesNotMatch(speedRaceHTML({ rows: [row({ seconds: 1, gpu: null, cores: "<x-c>" })] }), /<x-/, "cores escaped too");
assert.doesNotMatch(workflowHTML({ method: "<x-m>", seconds: 1, label: "<x-l>", hardware: "<x-h>", cores: "<x-c>", note: "<x-n>", source: "<x-s>" }), /<x-/);
for (const bad of [0, -1, NaN, Infinity, "5", true])
  assert.match(speedRaceHTML({ rows: [row({ seconds: bad })] }), /class="pending"/, `seconds ${String(bad)} is never a bar`);
assert.ok(raceState([row({ seconds: "5" }), row({ seconds: 4 })], 1e9)[0].clock === null);
assert.match(speedRaceHTML({ rows: [row({ seconds: 4, startup_s: 90 })] }), /\+ 90\.0 s start-up, not counted/);
assert.equal(bl.rows.length, 4, "the race is the four rows of one design point");
assert.ok(!bl.rows.some((r) => /3,780|Cox/.test(r.method)), "the Cox screen is the workflow footnote, never a bar");
// The lead's bug: bars must be proportional. Final width = seconds / slowest, on one shared scale per pair.
const pair = [row({ seconds: 2.88 }), row({ seconds: 0.873 })];
const end = raceState(pair, 1e9);
assert.equal(end[0].frac, 1);
assert.ok(Math.abs(end[1].frac - 0.873 / 2.88) < 1e-12, "the faster run's bar is ~30% of the slower one's");
assert.ok(end.every((s, i) => s.done && s.clock === pair[i].seconds), "each prints its real seconds at its finish");
const early = raceState(pair, RACE_MS * 0.2);
assert.ok(!early[0].done && !early[1].done && early[0].frac === early[1].frac, "both grow at one pace");
const half = raceState(pair, RACE_MS / 2);
assert.ok(half[1].done && !half[0].done && half[1].frac < half[0].frac, "the faster stops first, shorter");
// A ratio only for a same-scope MEASURED pair, faster row named, direction-aware.
const g = (rows, same_scope = true) => ({ same_scope, rows });
assert.equal(pairRatio(g([row({ method: "NumPy", seconds: 2.88 }), row({ method: "GPU", seconds: 0.873 })])), "GPU: ×3.3 faster");
assert.equal(pairRatio(g([row({ method: "1", seconds: 3.8 }), row({ method: "4", seconds: 11.44 })])), "1: ×3.0 faster");
assert.equal(pairRatio(g([row({ seconds: 2 }), row({ seconds: 2 })])), null, "equal: no ratio");
assert.equal(pairRatio(g([row({ seconds: 2 }), row({ seconds: 1 })], false)), null, "not same-scope: no ratio");
assert.equal(pairRatio(g([row({ seconds: 2 }), row({ seconds: 1, label: "ESTIMATE" })])), null, "both MEASURED or no ratio");
assert.equal(pairRatio(g([row({ seconds: 2 }), row({ seconds: null, label: "PENDING" })])), null);
assert.equal(pairRatio(bl), null, "the design-point race never prints a ratio");
assert.equal(pairRatio(g([row({ seconds: 3 }), row({ seconds: 2 }), row({ seconds: 1 })])), null);
for (const grp of bl.interim ?? []) assert.match(pairRatio(grp), /×\d\.\d faster$/, grp.title);
console.log("race.test.mjs: ok");
