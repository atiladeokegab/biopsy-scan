// The "Preflight workflow" section (node demo/workflow.test.mjs; exit 1 on failure): first on the page, four
// clickable steps in our own words, a label and the Charville credit on every visual, and the screen result
// equal to the merged audit/results/charville.json.
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";

const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
const plain = (s) => s.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
// The page holds one mount point; the section itself is workflow/main.js's TEMPLATE, rendered into #wf-root.
assert.match(html, /<section id="workflow-app"[^>]*><div id="wf-root"><\/div><\/section>/, "no #wf-root mount point");
assert.match(html, /<script type="module" src="workflow\/main\.js"><\/script>/);
assert.ok(html.indexOf('id="workflow-app"') < html.indexOf('class="hero"'), "the workflow is the page's top section");
const { TEMPLATE: wf, start } = await import("./workflow/main.js");
assert.equal(typeof start, "function");
assert.match(wf, /^<div class="wf" id="flow"/);
assert.match(plain(wf), /Preflight workflow/);

const steps = [...wf.matchAll(/<button type="button" class="wf-node"[^>]*data-step="(\d)"[^>]*>([\s\S]*?)<\/button>/g)];
assert.deepEqual(steps.map((m) => [m[1], plain(m[2]).replace(/^0\d /, "")]),
  [["0", "Tissue context"], ["1", "Decode the tumour"], ["2", "Stress-test"], ["3", "Routine stains"]]);

const vis = [...wf.matchAll(/<div class="wf-vis" data-step="(\d)"[^>]*>([\s\S]*?)<\/div>\s*<!-- \/vis -->/g)];
assert.equal(vis.length, 4, "four visuals");
const LABELS = /MEASURED (cells|channels)|SYNTHETIC imaging|SIMULATED|REAL IHC/;
for (const [, i, body] of vis) assert.match(body, LABELS, `visual ${i} has no label`);
// The cells' credit is the one tma.json carries; step 3 names every kind of number it shows; step 4 credits HPA.
const tma = JSON.parse(readFileSync(new URL("./data/tma.json", import.meta.url), "utf8"));
const credit = tma.note.split(";")[0];
assert.equal(credit, "Charville CODEX, Wu et al. 2022, Zenodo 13179600, CC-BY-4.0");
assert.ok(plain(wf).includes(credit), "the Charville credit");
for (const t of ["SIMULATED fields and mis-calls", "ANALYTIC formula", "ESTIMATE simulation"]) assert.ok(plain(vis[2][2]).includes(t), `step 3 lacks ${t}`);
assert.match(plain(vis[3][2]), /ILLUSTRATION/);
const { PICKED } = await import("./workflow/step4.js"); // step 4 draws the HPA credit in its frame
assert.match(PICKED.credit, /Human Protein Atlas.*CC BY-SA 3\.0/);
assert.doesNotMatch(plain(wf), /clinical practice|ready for (pathology|the clinic)/i);
const texts = [...wf.matchAll(/<div class="wf-text" data-step="(\d)"[^>]*>([\s\S]*?)<\/div>/g)];
assert.equal(texts.length, 4, "four step texts");

// Our own words: none of the reference video's names, phrases or widgets.
for (const banned of [/multivision/i, /capture the full tissue context/i, /risk[- ]score/i, /from discovery to deployment/i, /complete lifecycle/i, /cloud/i])
  assert.doesNotMatch(plain(wf), banned);

// The screen result is the merged file's.
const ch = JSON.parse(readFileSync(new URL("../audit/results/charville.json", import.meta.url), "utf8"));
const s2 = plain(texts[1][2]) + " " + plain(vis[1][2]);
assert.ok(s2.includes(`z ${ch.maxT.statistic.toFixed(2)}`), `step 2 lacks z ${ch.maxT.statistic.toFixed(2)}: ${s2}`);
assert.ok(s2.includes(`maxT p ${ch.maxT.p}`), `step 2 lacks maxT p ${ch.maxT.p}`);
assert.match(s2, /within chance/i);
assert.ok(ch.exploratory && /exploratory/i.test(s2), "step 2 says exploratory");

// Step 3's promise against reality is the merged comparison at 200 patients, two 400 µm fields, 10% mis-calls.
const vs = JSON.parse(readFileSync(new URL("../bench/results/vs-schoenfeld-300.json", import.meta.url), "utf8"));
assert.deepEqual([vs.design.regions_per_patient, vs.design.fov_um, vs.design.eps], [2, 400, 0.1]);
const at200 = vs.points.find((p) => p.patients === 200), s3 = plain(texts[2][2]);
for (const s of ["200 patients", "two 400 µm fields", "one cell in ten", `${Math.round(100 * at200.analytic_power)}% power`, `${Math.round(100 * at200.sim_power)}%`])
  assert.ok(s3.includes(s), `step 3 lacks "${s}": ${s3}`);

// Each step's module exists and exports mount.
for (const f of ["step1.js", "step2.js", "step3.js", "step4.js", "main.js"]) {
  const u = new URL(`./workflow/${f}`, import.meta.url);
  assert.ok(existsSync(u), `missing workflow/${f}`);
}
for (const f of ["step1.js", "step2.js", "step3.js", "step4.js"]) assert.match(readFileSync(new URL(`./workflow/${f}`, import.meta.url), "utf8"), /export function mount\(el, data\)/);
console.log("workflow.test.mjs: ok");

// Cell colours sit on the token families: tumour in coral, immune in amber/gold, stroma/other in slate; emerald is kept for highlights.
{
  const { TYPE_COLORS } = await import("./workflow/palette.js");
  const hue = (hex) => { const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255), mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    if (!d) return null; const h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4; return (h * 60 + 360) % 360; };
  const sat = (hex) => { const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)); return Math.max(r, g, b) - Math.min(r, g, b); };
  const fam = { tumour: [345, 25], immune: [30, 55], stroma: [190, 230] };
  tma.type_groups.forEach((g, i) => {
    const c = TYPE_COLORS[i], h = hue(c), [lo, hi] = fam[g];
    assert.ok(h !== null && (lo < hi ? h >= lo && h <= hi : h >= lo || h <= hi), `${tma.type_names[i]} (${g}) ${c} hue ${h} off its family`);
    if (g === "stroma") assert.ok(sat(c) < 90, `${tma.type_names[i]} ${c} is not slate`);
  });
  assert.equal(new Set(TYPE_COLORS.map((c) => c.toLowerCase())).size, TYPE_COLORS.length, "every type has its own colour");
  assert.ok(!TYPE_COLORS.some((c) => c.toLowerCase() === "#2bd4a4"), "emerald is for highlights only");
}

// Type: Instrument Serif for the headline and the big numbers, JetBrains Mono for chips and kicker, self-hosted with their OFL.
{
  const css = readFileSync(new URL("./style.css", import.meta.url), "utf8");
  for (const [fam, file, ofl] of [["Instrument Serif", "InstrumentSerif-400-latin.woff2", "OFL-instrumentserif.txt"], ["JetBrains Mono", "JetBrainsMono-400-latin.woff2", "OFL-jetbrainsmono.txt"]]) {
    assert.match(css, new RegExp(`@font-face\\s*\\{[^}]*"${fam}"[^}]*fonts/${file}`), `no @font-face for ${fam}`);
    for (const f of [file, ofl]) assert.ok(existsSync(new URL(`./fonts/${f}`, import.meta.url)), `demo/fonts/${f} missing`);
  }
  assert.match(css, /\.wf-head\s*\{[^}]*var\(--font-serif\)/, "headline not in the serif");
  assert.match(css, /\.wf-eyebrow\s*\{[^}]*var\(--font-mono\)/, "kicker not in the mono");
}
