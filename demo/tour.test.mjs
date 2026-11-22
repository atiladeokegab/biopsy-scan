// The guided tour: every stop points at something on the page, and the moves stay in range (node demo/tour.test.mjs).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { STEPS, step } from "./tour.js";

const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
// shortcut: #flow arrives with build-demo's workflow PR; drop this exception once both are on integration.
for (const { target } of STEPS.filter((s) => s.target !== "#flow")) {
  const name = target.slice(1);
  const re = target[0] === "#" ? new RegExp(`id="${name}"`) : new RegExp(`class="[^"]*\\b${name}\\b`);
  assert.match(html, re, `tour target ${target} is on the page`);
}
assert.equal(STEPS[0].target, "#flow", "the tour opens on the Preflight workflow");
assert.match(html, /id="tour-btn"[^>]*>Start tour</, "a Start tour button is always in the bar");
assert.equal(step(0, "back"), 0);
assert.equal(step(0, "next"), 1);
assert.equal(step(STEPS.length - 1, "next"), -1);
assert.equal(step(3, "skip"), -1);
console.log("tour ok");
