// Text for the results cards (claims.json) and the pattern panel (biomarker.json): pure, so race.test.mjs checks
// every number before it reaches the page. Each throws on a bad file, and app.js renders each panel on its own.
const fin = (v) => typeof v === "number" && Number.isFinite(v);
const int = (v) => Number.isInteger(v);
const pct = (v) => `${Math.round(v * 100)}%`;
const pct1 = (v) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;
const need = (ok, what) => { if (!ok) throw new Error(`bad data: ${what}`); };

export function claimTexts({ power: pw, false_alarms: fa, maxT: mt }) {
  need(int(pw?.patients) && int(pw.n) && [pw.sim, pw.lo, pw.hi, pw.formula].every(fin) &&
    0 <= pw.lo && pw.lo <= pw.sim && pw.sim <= pw.hi && pw.hi <= 1 && pw.formula >= 0 && pw.formula <= 1 &&
    typeof pw.model === "string", "power");
  need(int(fa?.hits) && int(fa.n) && fa.n > 0 && fa.hits >= 0 && fa.hits <= fa.n && fin(fa.target) &&
    int(fa.designs) && fa.designs > 0, "false_alarms");
  need(fin(mt?.p) && mt.p > 0 && mt.p <= 1 && int(mt.tests) && mt.tests > 0, "maxT");
  return {
    "pw-n": String(pw.patients),
    "pw-sim": pct(pw.sim),
    "pw-ci": `95% interval ${Math.round(pw.lo * 100)}–${Math.round(pw.hi * 100)}%, ${pw.n} simulated studies`,
    "pw-formula": pct(pw.formula),
    "pw-model": `Simulated design: ${pw.model}.`,
    "pw-src": pw.source,
    "fa-rate": pct1(fa.hits / fa.n),
    "fa-detail": `${fa.hits} of ${fa.n} null studies, pooled over ${fa.designs} designs, against a ${pct(fa.target)} target: ` +
      "a pooled summary, not per-design error control",
    "mt-p": `p = ${mt.p.toFixed(3)}`,
    "mt-detail": `for the pattern after ${mt.tests.toLocaleString("en-GB")} tests (maxT): ${mt.verdict}.`,
    "fa-src": `${fa.source} · ${mt.source}`,
  };
}

export function bioTexts(b) {
  need(fin(b?.feature?.z) && typeof b.feature.definition === "string" && typeof b.pd1_positive_rule === "string", "feature");
  const out = {
    "bio-def": `Tested: ${b.feature.definition}, a continuous score with no threshold. ` +
      `Drawn bright: ${b.pd1_positive_rule}.`,
    "bio-cap": `The top-ranked pattern, z = ${b.feature.z.toFixed(2)}. ${b.note} ` +
      "Can't be tested in our second cohort: it has no PD1 stain.",
  };
  for (const r of b.regions) {
    const c = r.counts;
    need(["tumour", "macrophages", "pd1_pos_macrophages", "pd1_pos_near_tumour"].every((k) => int(c?.[k])), `region ${r.arm}`);
    out[`reg-${r.arm}-cap`] = `${r.arm === "high" ? "High" : "Low"} arm, median patient: ${c.tumour} Tumor 5 ` +
      `cell${c.tumour === 1 ? "" : "s"}, ${c.macrophages} macrophages, ${c.pd1_pos_macrophages} drawn PD1+, ` +
      `${c.pd1_pos_near_tumour} of them within 100 µm`;
  }
  return out;
}

// The dials' grid: Atlas's grid is simulated on real tissue (source) with synthetic outcomes (meta.simulation_label).
export function gridTexts(g) {
  const outcomes = g.meta?.simulation_label ?? g.source;
  need(["MEASURED", "SYNTHETIC"].includes(g.source) && ["MEASURED", "SYNTHETIC"].includes(outcomes) && fin(g.effect_hr_per_sd), "grid");
  const legacy = /legacy R4/.test(g.note ?? "");
  return {
    provenance: `MEASURED tissue · ${outcomes} outcomes`,
    "grid-tag": g.source === "MEASURED" ? `MEASURED simulation · ${outcomes} outcomes` : "SYNTHETIC grid",
    planted: `Planted ${outcomes === "SYNTHETIC" ? "synthetic " : ""}effect: HR ${g.effect_hr_per_sd} per SD` +
      (legacy ? ", legacy Cox score test (R4); not the PD1 pattern, not the R15 screen" : ""),
  };
}
