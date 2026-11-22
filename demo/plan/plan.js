// Study-planning calculator (data.json, built by build_data.py from the committed power-atlas production runs).
// Pure, so plan.test.mjs checks it. A simulated design point is shown as stored; between two simulated patient counts
// of the same design it is a straight line labelled ESTIMATE; anything else (an unsimulated field count, field size or
// mis-call rate, or a patient count outside the run) is "Not measured". Nothing is extrapolated.
export const pct = (v) => (v >= 0.995 && v < 1 ? ">99%" : v > 0 && v < 0.005 ? "<1%" : `${Math.round(v * 100)}%`);

const sameFov = (a, b) => String(a) === String(b);

// The lab's inputs -> {sim, formula, stains}; sim is {power, lo, hi, label, source, note} or {missing, source}.
export function plan(data, { cohort, patients, regions, fov, miscall }) {
  const c = data.cohorts[cohort], src = `${c.source} @ ${c.source_commit.slice(0, 7)}`;
  const ax = c.axes, ps = ax.patients, lo = ps[0], hi = ps[ps.length - 1];
  const why = [];
  if (!ax.regions.includes(regions)) why.push(`${regions} fields per patient (simulated: ${ax.regions.join(", ")})`);
  if (!ax.fov_um.some((f) => sameFov(f, fov))) why.push(`a ${fov} µm field (simulated: ${ax.fov_um.join(", ")} µm)`);
  if (!ax.miscall.includes(miscall)) why.push(`a ${Math.round(miscall * 100)}% mis-call rate (simulated: ${ax.miscall.map((m) => `${m * 100}%`).join(", ")})`);
  if (!(patients >= lo && patients <= hi)) why.push(`${patients} patients (simulated: ${lo} to ${hi})`);
  let sim;
  if (why.length) {
    sim = { missing: `Not measured: nobody simulated ${why.join("; ")}.`, source: src };
  } else {
    const row = ps.map((n) => c.points.find((p) => p.patients === n && p.regions === regions && sameFov(p.fov_um, fov) && p.miscall === miscall));
    const j = ps.findIndex((n) => n >= patients), b = row[j];
    if (ps[j] === patients) {
      sim = { power: b.power, lo: b.lo, hi: b.hi, label: data.labels.sim, source: src,
        note: `Stored point ${b.design_id}: ${b.n.toLocaleString("en")} simulated studies, 95% interval ${pct(b.lo)}–${pct(b.hi)}.` };
    } else {
      const a = row[j - 1], t = (patients - ps[j - 1]) / (ps[j] - ps[j - 1]), mix = (k) => a[k] + t * (b[k] - a[k]);
      sim = { power: mix("power"), lo: mix("lo"), hi: mix("hi"), label: data.labels.interp, source: src,
        note: `No study was run at ${patients} patients: value and band joined linearly between ${ps[j - 1]} and ${ps[j]}; not a validated interval.` };
    }
  }
  const inRange = Number.isInteger(patients) && patients >= lo && patients <= hi;
  const formula = inRange
    ? { power: c.formula[patients - lo], label: data.labels.formula, source: `bench/vs_schoenfeld.py · ${c.features_source} (${c.features_m} features)` }
    : { missing: `Not computed: whole patient counts from ${lo} to ${hi} only.` };
  const s = data.stains; // shortcut: only the UNMEASURED case is drawn; add a stains table when translate/panels.py has a real run
  const stains = { missing: `Not measured: the routine-stain panel search (${s.code}) has not run on real data yet. ${s.source} says ${s.status}: ${s.reason}.`, source: s.source };
  return { sim, formula, stains };
}

export function scenario(data, cohort) {
  const c = data.cohorts[cohort];
  return `${c.cohort} cell tables (${c.source_patients} patients, ${c.source_events} events), synthetic outcomes, one planted ` +
    `effect of HR ${c.hr_per_sd} per SD among ${c.features_m} features, family-wise alpha ${c.alpha}. ${c.studies_per_point.toLocaleString("en")} ` +
    `studies per point; atlas code ${c.atlas_code_sha.slice(0, 7)}; ${c.hardware}. No clinical claim.`;
}
