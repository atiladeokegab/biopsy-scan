// "Preflight your study" (data/preflight.json, built by collect_preflight.py from two committed power files).
// Pure, so preflight.test.mjs checks every number before it reaches the page. No network, no simulation: a simulated
// point is shown as stored; between two points it is linear in patients and labelled ESTIMATE (interpolated);
// outside a design's simulated range, or for a window and mis-call pair nobody simulated, there is no number.
const pct = (v) => `${Math.round(v * 100)}%`;

export function design(data, window, miscall) {
  return data.designs.find((d) => d.window === window && d.miscall === miscall) ?? null;
}

// Simulated power at n patients: {power, lo, hi, label, note} or {missing: reason}.
export function estimate(d, n) {
  if (!d) return { missing: "Not simulated: only two imaging designs were run, two 400 µm fields with 10% mis-called and the whole core with none." };
  const pts = d.points, last = pts[pts.length - 1];
  if (n < pts[0].patients || n > last.patients) {
    return { missing: `Not simulated: this design was run from ${pts[0].patients} to ${last.patients} patients.` };
  }
  const i = pts.findIndex((p) => p.patients >= n), b = pts[i];
  if (b.patients === n) {
    return { power: b.power, lo: b.lo, hi: b.hi, label: d.label,
      note: `95% interval ${pct(b.lo)}–${pct(b.hi)}, ${b.n} simulated studies at ${n} patients` };
  }
  const a = pts[i - 1], t = (n - a.patients) / (b.patients - a.patients), mix = (k) => a[k] + t * (b[k] - a[k]);
  return { power: mix("power"), lo: mix("lo"), hi: mix("hi"), label: "ESTIMATE (interpolated)",
    note: `interval ${pct(mix("lo"))}–${pct(mix("hi"))}, drawn straight between the simulated points at ` +
      `${a.patients} and ${b.patients} patients (${a.n} studies each)` };
}

// Everything the panel prints for one setting of the three controls.
export function panelTexts(data, { patients, window, miscall }) {
  const f = data.formula.power[String(patients)];
  if (typeof f !== "number") throw new Error(`bad data: no formula value at ${patients} patients`);
  const d = design(data, window, miscall), e = estimate(d, patients);
  const gap = e.missing ? null : Math.round(f * 100) - Math.round(e.power * 100);
  return {
    sim: e.missing ? "—" : pct(e.power),
    simTag: e.missing ? "NOT SIMULATED" : e.label,
    simNote: e.missing ?? e.note,
    formula: pct(f),
    formulaTag: data.formula.label,
    gap: gap === null ? "The formula gives the same number whatever the window or mis-call rate."
      : gap > 0 ? `The formula overstates power by ${gap} points here.`
      : gap < 0 ? `The formula understates power by ${-gap} points here.`
      : "The formula and the simulation agree here.",
    src: d ? `${d.source} · ${data.formula.source}` : data.formula.source,
  };
}
