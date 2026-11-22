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
const SRC = {
  fa: "stats/results/hard-negative-f458ad1.json", cv: "results/charville-v2.json", upmc: "results/upmc-rediscovery.json",
  fused: "bench/results/modal-design-20261121T205536Z.json", sweep: "bench/results/sweep-repeats.json",
  vs: "bench/results/vs-schoenfeld-300.json", race: "bench/results/modal-design-20261121T194458Z.json",
  ext: "bench/results/same-host-20261122T140908Z.json", agree: "docs/external-agreement.md",
};
const fa = json(SRC.fa), cv = json(SRC.cv), upmc = json(SRC.upmc), fusedRun = json(SRC.fused);
const sweep = json(SRC.sweep), vs = json(SRC.vs), race = json(SRC.race), ext = json(SRC.ext);
const first = (label) => label.split(/[ :]/)[0]; // "ESTIMATE (sim)" -> "ESTIMATE"
// Each row's evidence links and labels, in order, as the source files state them; a swapped badge or link goes red.
const expected = [
  { links: [SRC.fa, SRC.cv, SRC.upmc], tags: [fa.label, cv.labels.cohort_counts, first(upmc.locked.power.label)] },
  { links: [SRC.fused], tags: [fusedRun.labels.compute] },
  { links: [SRC.agree, SRC.ext, SRC.sweep], tags: [first(ext.labels.timing), sweep.labels.timings] },
  { links: [SRC.vs], tags: [first(vs.simulated.label), first(vs.analytic.label)] },
  { links: [SRC.race], tags: [first(race.labels.rates)] },
];
const LABELS = /^(MEASURED|ESTIMATE|SYNTHETIC|ANALYTIC|SIMULATED|DERIVED)$/;
rows.forEach((r, i) => {
  const links = [...r.matchAll(/<a class="link" href="([^"]+)"/g)].map((m) => m[1]);
  for (const l of links) assert.ok(l.startsWith(BLOB), `link not a blob at integration: ${l}`);
  assert.deepEqual(links.map((l) => l.slice(BLOB.length)), expected[i].links, `row ${i} evidence links`);
  for (const l of links) readFileSync(new URL(l.slice(BLOB.length), repo)); // the linked file is in the repo
  const tags = [...r.matchAll(/class="tag[^"]*">([^<]*)</g)].map((m) => m[1]);
  for (const t of expected[i].tags) assert.match(t, LABELS, `source label not in the label set: ${t}`);
  assert.deepEqual(tags, expected[i].tags, `row ${i} labels: ${plain(r)}`);
});
const legend = plain(judges.match(/<p class="pf-legend">([\s\S]*?)<\/p>/)?.[1] ?? "");
for (const t of new Set(expected.flatMap((e) => e.tags))) assert.ok(legend.includes(`${t},`), `legend lacks ${t}`);
assert.ok(!/<\/span>\s+[.,;]/.test(judges), "space before punctuation after a tag pill");
const has = (i, s) => assert.ok(plain(rows[i]).includes(s), `row ${i} lacks "${s}": ${plain(rows[i])}`);
const pct = (x) => `${(100 * x).toFixed(1)}%`;
const n = (x) => x.toLocaleString("en-US");
// Row 0: false-alarm rate with its denominator and interval; Charville not significant, not "no signal".
const boot = fa.rates.bootstrap;
has(0, pct(boot.rate)); has(0, `${boot.rejections}/${n(fa.n_datasets)}`);
has(0, `${pct(boot.clopper_pearson_95[0])}–${pct(boot.clopper_pearson_95[1])}`);
has(0, "not significant"); has(0, `adjusted p = ${cv.maxT.p}`);
has(0, `${Math.round(100 * upmc.locked.power.value)}%`);
// Row 1: per evaluated study, both workloads, calibration nulls, setup excluded, different procedures.
const { lab, fused } = fusedRun.arms;
assert.equal(lab.workers, 8); assert.equal(fused.workers, 8);
has(1, `${Math.round(lab.stages.wall_per_study_s / fused.stages.wall_per_study_s)}×`); has(1, "8 workers against 8");
has(1, "per evaluated study"); has(1, `${n(lab.stages.evaluated_studies)} lab studies`);
has(1, `${n(fused.stages.evaluated_studies)} Preflight studies`);
has(1, `${n(fused.summary.studies.calibration)} calibration nulls`); has(1, "setup excluded"); has(1, "different procedures");
has(1, "cross-run"); has(1, `${(lab.stages.compute_wall_s / fused.stages.compute_wall_s).toFixed(1)}× per whole design point`);
has(1, "re-images each study"); has(1, "the cutoff and calls are outside the timing");
// Row 2: 534x off the shelf, same host, pipeline only, with its range; 5.5x is the fair 8 v 8 median; 6.3x is compute span, not Modal wall.
const exr = ext.summary.ratios;
has(2, `${Math.round(exr.external_pipeline_vs_fused_full.median)}×`); has(2, `${ext.summary.external_pipeline_s_per_study.median.toFixed(2)} s`); has(2, "same host");
has(2, `range ${Math.round(exr.external_pipeline_vs_fused_full.min)}–${Math.round(exr.external_pipeline_vs_fused_full.max)}×`);
has(2, `${Math.round(exr.external_with_checks_vs_fused_full.median)}× with them`); has(2, `${Math.round(exr.external_pipeline_vs_fused_matched.median)}× on the identical 48 studies`);
assert.ok(!plain(rows[2]).includes("cross-run"), "row 2 still says cross-run");
has(2, `${sweep.fair.spread.ratio.median.toFixed(1)}×`); has(2, "8 workers against 8");
has(2, "reference recount and agreement checks timed separately");
// Scope beside 534x and 312x (basis keys in the same-host file); 547x is the closest to the old combined-checks basis.
has(2, "the whole off-the-shelf call per study"); has(2, "each result returned to the parent");
has(2, "re-imaging and scoring only, the calibrated threshold and calls a separate step, excluded");
has(2, "the closest scope to the old 583× combined-checks basis"); has(2, "top pick the same in 16 of 16 in a separate matched-feature, matched-score experiment"); // docs/external-agreement.md, #174
// Both denominators beside 534x: the external arm's studies and the fused arm's evaluations (3 streams of full_studies_per_stream).
has(2, "median of the paired per-repeat ratios"); has(2, `over ${ext.repeats[0].arms.external.studies} external studies`);
has(2, `over ${n(3 * ext.design.full_studies_per_stream)} fused evaluations`);
has(2, "per decided study with calibration charged");
has(2, `${sweep.scaling.spread.speedup_64_cores.median.toFixed(1)}×`); has(2, `${sweep.scaling.repeats.length} repeats`);
has(2, `${sweep.fleet.spread.speedup_compute_span.median.toFixed(1)}× less compute span`);
has(2, `Modal wall ${sweep.fleet.spread.speedup_modal_wall.median.toFixed(1)}×`); has(2, `${sweep.fleet.repeats.length} repeats`);
// Row 3: the simulated power's denominator and its synthetic scenario.
const at200 = vs.points.find((p) => p.patients === 200);
has(3, pct(at200.analytic_power)); has(3, pct(at200.sim_power));
has(3, `${Math.round(at200.sim_power * at200.sim_n)}/${at200.sim_n}`); has(3, "synthetic outcome model");
has(3, `HR ${vs.hr_per_sd.toFixed(0)} per SD`); has(3, `resampled from ${vs.patients_source}`);
// Row 4: an observed power comparison between different procedures, with counts.
for (const a of [race.arms.numba, race.arms.lab]) { has(4, pct(a.summary.power.rate)); has(4, `${a.summary.power.k}/${n(a.summary.power.n)}`); }
has(4, "different procedures");
assert.ok(!/discover|threshold costs/i.test(plain(judges)), "For judges must not claim a discovery or a threshold-only cause");
console.log("copy.test.mjs: For judges ok");
