// A-9.2@1: the page's copy (node demo/copy.test.mjs; exit 1 on failure).
// The h1 is the tagline, the button says what it does, and the old question is gone from the repo.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";

const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
const text = (re) => (html.match(re)?.[1] ?? "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();

assert.equal(text(/<h1[^>]*>([\s\S]*?)<\/h1>/), "Thousands of studies, simulated in seconds, before you image a slide.");
assert.equal(text(/<button[^>]*id="run"[^>]*>([\s\S]*?)<\/button>/), "Run the studies");

// The old question in any form (my, your, our study, ...), in every tracked file.
// docs/PRD.md is excluded: it is the spec that names the banned phrase.
let hits = "";
try {
  hits = execFileSync("git", ["grep", "-n", "-i", "-E", "can [a-z]+ stud(y|ies) find it", "--", ".", ":!docs/PRD.md"],
    { cwd: new URL("..", import.meta.url), encoding: "utf8" });
} catch (e) {
  if (e.status !== 1) throw e; // git grep exits 1 when nothing matches
}
assert.equal(hits, "", `the old question is still in the repo:\n${hits}`);
console.log("copy.test.mjs: ok");
