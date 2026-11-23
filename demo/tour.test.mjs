// The guided tour: every stop points at something on the page, and the moves stay in range (node demo/tour.test.mjs).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { STEPS, step, layout } from "./tour.js";

// The workflow's own markup arrives from workflow/main.js at run time.
const html = readFileSync(new URL("./index.html", import.meta.url), "utf8") + readFileSync(new URL("./workflow/main.js", import.meta.url), "utf8");
for (const { target } of STEPS) {
  const name = target.slice(1);
  const re = target[0] === "#" ? new RegExp(`id="${name}"`) : new RegExp(`class="[^"]*\\b${name}\\b`);
  assert.match(html, re, `tour target ${target} is on the page`);
}
assert.equal(STEPS[0].target, "#workflow-app", "the tour opens on the Biopsy Scan workflow");
assert.match(html, /id="tour-btn"[^>]*aria-label="Start tour"/, "a Start tour button is always in the bar");
assert.equal(step(0, "back"), 0);
assert.equal(step(0, "next"), 1);
assert.equal(step(STEPS.length - 1, "next"), -1);
assert.equal(step(3, "skip"), -1);

// The card never covers the visible spotlight.
const hits = (a, b) => a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
const box = (c, cw, ch) => ({ ...c, right: c.left + cw, bottom: c.top + ch });
for (const [r, vw, vh] of [
  [{ top: 60, bottom: 1100, left: 16, right: 590 }, 1280, 633],   // tall, narrow: beside it
  [{ top: 60, bottom: 300, left: 16, right: 1264 }, 1280, 633],   // short, wide: below it
  [{ top: 400, bottom: 600, left: 16, right: 1264 }, 1280, 633],  // low on screen: above it
  [{ top: 100, bottom: 1900, left: 16, right: 374 }, 390, 757],   // phone, tall: pinned, spotlight cut short
]) {
  const { spot, card } = layout(r, 358, 230, vw, vh, 48);
  assert.ok(!hits(spot, box(card, 358, 230)), JSON.stringify({ r, spot, card }));
  assert.ok(card.top >= 0 && card.top + 230 <= vh && card.left >= 0 && card.left + 358 <= vw, "card on screen");
}
// A screen shorter than the card: the card still starts on screen.
for (const [vw, vh] of [[390, 260], [667, 320]]) {
  const { card } = layout({ top: 60, bottom: 900, left: 16, right: vw - 16 }, 358, 300, vw, vh, 48);
  assert.ok(card.top >= 0 && card.left >= 0, JSON.stringify({ vw, vh, card }));
}
console.log("tour ok");
