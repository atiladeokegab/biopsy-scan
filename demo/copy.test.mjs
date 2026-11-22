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

// "For judges": four criteria rows below the results cards, plus the honest-limits line. Each figure is the
// one in the merged file it links to (GitHub blob at integration), so the panel cannot drift from the results.
const repo = new URL("..", import.meta.url);
const json = (path) => JSON.parse(readFileSync(new URL(path, repo), "utf8"));
const judges = html.match(/<section[^>]*id="judges"[^>]*>([\s\S]*?)<\/section>/)?.[1] ?? "";
assert.ok(judges, "no For judges section");
assert.ok(html.indexOf('id="judges"') > html.indexOf('class="results"'), "For judges sits below the results cards");
assert.equal(text(/<section[^>]*id="judges"[^>]*>[\s\S]*?<h2[^>]*>([\s\S]*?)<\/h2>/), "For judges");
const rows = [...judges.matchAll(/<p class="pf-scenario"[^>]*>([\s\S]*?)<\/p>/g)].map((m) => m[1]);
const plain = (s) => s.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
assert.deepEqual(rows.map((r) => plain(r).split(".")[0]),
  ["Correctness and assumptions", "Fundamental reshaping", "Meaningful performance", "Scientific value", "Honest limits"]);
const BLOB = "https://github.com/atiladeokegab/sota-parallelism-nov21/blob/integration/";
for (const r of rows) {
  const links = [...r.matchAll(/<a class="link" href="([^"]+)"/g)].map((m) => m[1]);
  assert.ok(links.length > 0, `row without a link: ${plain(r)}`);
  for (const l of links) {
    assert.ok(l.startsWith(BLOB), `link not a blob at integration: ${l}`);
    readFileSync(new URL(l.slice(BLOB.length), repo)); // the linked file is in the repo
  }
  assert.ok(/class="tag[^"]*">(MEASURED|ESTIMATE|SYNTHETIC|ANALYTIC|SIMULATED|UNMEASURED)</.test(r), `row without a label: ${plain(r)}`);
}
const has = (i, s) => assert.ok(plain(rows[i]).includes(s), `row ${i} lacks "${s}": ${plain(rows[i])}`);
const pct = (x) => `${(100 * x).toFixed(1)}%`;
const fw = json("stats/results/hard-negative-f458ad1.json").rates.bootstrap.rate;
has(0, fw.toFixed(3));
has(0, "within chance"); has(0, `${Math.round(100 * json("results/upmc-rediscovery.json").locked.power.value)}%`);
const arms = json("bench/results/modal-design-20261121T205536Z.json").arms;
assert.equal(arms.lab.workers, 8); assert.equal(arms.fused.workers, 8);
has(1, `${Math.round(arms.lab.stages.wall_per_study_s / arms.fused.stages.wall_per_study_s)}×`); has(1, "8 workers against 8");
const sweep = json("bench/results/sweep-repeats.json");
has(2, `${sweep.scaling.spread.speedup_64_cores.median.toFixed(1)}×`); has(2, `${sweep.scaling.repeats.length} repeats`);
has(2, `${sweep.fleet.spread.speedup_compute_span.median.toFixed(1)}×`);
const at200 = json("bench/results/vs-schoenfeld-300.json").points.find((p) => p.patients === 200);
has(3, pct(at200.analytic_power)); has(3, pct(at200.sim_power));
const lim = json("bench/results/modal-design-20261121T194458Z.json").arms;
has(4, pct(lim.numba.summary.power.rate)); has(4, pct(lim.lab.summary.power.rate));
assert.ok(!/discover/i.test(plain(judges)), "For judges must not claim a discovery");
console.log("copy.test.mjs: For judges ok");
