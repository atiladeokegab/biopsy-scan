// Regression: the measured speed line points the right way (node demo/race.test.mjs; exit 1 on failure).
import assert from "node:assert/strict";
import { speedWord } from "./race.js";

assert.equal(speedWord(1, 2), "GPU 2.0× slower"); // the bug: this used to print "1× faster"
assert.equal(speedWord(2, 2), "about the same");
assert.equal(speedWord(2.04, 2), "about the same");
assert.equal(speedWord(8.4, 2), "GPU 4.2× faster");
assert.equal(speedWord(3600, 4.2), "GPU 857× faster");
console.log("race.test.mjs: ok");
