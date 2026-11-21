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
console.log("race.test.mjs: ok");
