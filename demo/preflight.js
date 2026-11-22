// "Preflight your study" (data/preflight.json, built by collect_preflight.py from two committed power files).
// Pure, so preflight.test.mjs checks every number before it reaches the page. No network, no simulation: a simulated
// point is shown as stored; between two points it is linear in patients and labelled ESTIMATE (interpolated);
// outside a design's simulated range, or for a window and mis-call pair nobody simulated, there is no number.
const pct = (v) => (v >= 0.995 && v < 1 ? ">99%" : `${Math.round(v * 100)}%`); // never round a near-certainty up to 100%

export function design(data, window, miscall) {
  return data.designs.find((d) => d.window === window && d.miscall === miscall) ?? null;
}

const pct1 = (v) => `${(v * 100).toFixed(1)}%`;

// Simulated power at n patients: {power, lo, hi, label, note, simulated} or {missing: reason}.
// A stored point keeps its pointwise 95% Monte Carlo interval; between two points the value and both bounds are
// joined linearly, which is a display, not a validated 95% interval at that size.
export function estimate(d, n) {
  if (!d) return { missing: "Not simulated: only two imaging designs were run, two 400 µm fields with 10% mis-called and the whole core with none." };
  const pts = d.points, last = pts[pts.length - 1];
  if (n < pts[0].patients || n > last.patients) {
    return { missing: `Not simulated: this design was run from ${pts[0].patients} to ${last.patients} patients.` };
  }
  const i = pts.findIndex((p) => p.patients >= n), b = pts[i];
  if (b.patients === n) {
    return { power: b.power, lo: b.lo, hi: b.hi, label: d.label, simulated: true,
      note: `Simulated point: pointwise 95% Monte Carlo interval ${pct(b.lo)}–${pct(b.hi)}, ${b.n} studies at ${n} patients.` };
  }
  const a = pts[i - 1], t = (n - a.patients) / (b.patients - a.patients), mix = (k) => a[k] + t * (b[k] - a[k]);
  return { power: mix("power"), lo: mix("lo"), hi: mix("hi"), label: "ESTIMATE (interpolated)", simulated: false,
    note: `Interpolated: no study was run at ${n} patients. Value and bounds (${pct(mix("lo"))}–${pct(mix("hi"))}) are ` +
      `joined linearly between the simulated points at ${a.patients} and ${b.patients} patients; ` +
      `not a validated 95% interval at ${n}.` };
}

// The fixed scenario every number on the panel belongs to (scenario block of preflight.json).
export function scenarioText({ scenario: s }) {
  const cohort = s.cohort[0].toUpperCase() + s.cohort.slice(1);
  return `Fixed scenario: simulated on the empirical ${cohort} source (${s.patients_source} patients, ` +
    `${s.events_source} events) with synthetic survival outcomes; one planted effect of HR ${s.hr_per_sd} per SD among ` +
    `${s.features_m} features; family-wise alpha ${s.alpha}. Above ${s.patients_source} patients, studies resample the ` +
    `source patients. No clinical claim.`;
}

// A design whose false-alarm rate was above target at any simulated size carries that caveat whenever it is chosen.
function caveat(d, { alpha, null_studies: m }) {
  const p = d?.points.find((q) => q.fwer_lo > alpha);
  return p ? `Caveat: at ${p.patients} patients this design rejected a true null in ${Math.round(p.fwer * m)} of ${m} ` +
    `null studies (${pct1(p.fwer)}, 95% interval ${pct1(p.fwer_lo)}–${pct1(p.fwer_hi)}) against a ${alpha * 100}% target, so error control was not ` +
    `established there.` : "";
}

// Everything the panel prints for one setting of the three controls.
export function panelTexts(data, { patients, window, miscall }) {
  const f = data.formula.power[String(patients)];
  if (typeof f !== "number") throw new Error(`bad data: no formula value at ${patients} patients`);
  const d = design(data, window, miscall), e = estimate(d, patients);
  const inside = !e.missing && e.simulated && e.lo <= f && f <= e.hi;
  return {
    sim: e.missing ? "—" : pct(e.power),
    simTag: e.missing ? "NOT SIMULATED" : e.label,
    simNote: e.missing ?? e.note,
    formula: pct(f),
    formulaTag: data.formula.label,
    gap: e.missing ? `Formula ${pct(f)}; no simulated value for this setting. The formula gives the same number whatever the window or mis-call rate.`
      : `Formula ${pct(f)} vs simulated ${pct(e.power)} (gap ${Math.abs(Math.round((f - e.power) * 100))} points).` +
        (inside ? " The formula's value lies inside the simulated interval." : ""),
    caveat: caveat(d, data.scenario),
    src: [...new Set([d?.source, data.formula.source].filter(Boolean))].join(" · "),
    chartLabel: `Power against patients, 50 to 600: the formula's curve (ANALYTIC)` +
      (d ? `, and the simulated curve for this design with dots at the simulated points and a band joining their pointwise 95% intervals` +
        (e.missing ? `; ${patients} patients is outside its simulated range` : "") : `; this window and mis-call pair was not simulated, so there is no simulated curve`) + ".",
    showSimKey: Boolean(d),
  };
}

// The panel's chart as SVG markup, numbers only: the formula's curve, the chosen design's simulated points (dots)
// joined linearly (the interpolation the panel uses), a band joining their pointwise 95% intervals, and a cursor.
export function chartSVG(data, { patients, window, miscall }, W = 560, H = 220) {
  const L = 40, R = 12, T = 12, B = 34, X0 = 50, X1 = 600;
  const x = (n) => L + ((n - X0) / (X1 - X0)) * (W - L - R), y = (p) => T + (1 - p) * (H - T - B);
  const f = (v) => v.toFixed(1), line = (pts) => pts.map(([a, b], i) => `${i ? "L" : "M"}${f(x(a))},${f(y(b))}`).join("");
  const out = [];
  for (const p of [0, 0.5, 0.8, 1]) {
    out.push(`<line class="grid${p === 0.8 ? " target" : ""}" x1="${L}" x2="${W - R}" y1="${f(y(p))}" y2="${f(y(p))}"/>`,
      `<text x="${L - 6}" y="${f(y(p) + 4)}" text-anchor="end">${p * 100}%</text>`);
  }
  for (const n of [50, 200, 400, 600]) out.push(`<text x="${f(x(n))}" y="${H - 14}" text-anchor="middle">${n}</text>`);
  out.push(`<text x="${f((L + W - R) / 2)}" y="${H - 1}" text-anchor="middle">patients</text>`);
  const fx = Object.entries(data.formula.power).map(([n, p]) => [+n, p]);
  out.push(`<path class="formula" d="${line(fx)}"/>`);
  const d = design(data, window, miscall);
  if (d) {
    // Clip to the slider's 50 to 600 with the same straight lines the panel interpolates on.
    const inside = d.points.filter((p) => p.patients > X0 && p.patients < X1).map((p) => p.patients);
    const ends = [X0, X1].filter((n) => n >= d.points[0].patients && n <= d.points.at(-1).patients);
    const ns = [...new Set([...ends, ...inside])].sort((a, b) => a - b), es = ns.map((n) => [n, estimate(d, n)]);
    out.push(`<path class="band" d="${line(es.map(([n, e]) => [n, e.hi]))}${line(es.map(([n, e]) => [n, e.lo]).reverse()).replace("M", "L")}Z"/>`,
      `<path class="sim" d="${line(es.map(([n, e]) => [n, e.power]))}"/>`);
    for (const p of d.points.filter((q) => q.patients >= X0 && q.patients <= X1)) {
      out.push(`<circle class="pt" cx="${f(x(p.patients))}" cy="${f(y(p.power))}" r="3"/>`);
    }
  }
  const e = estimate(d, patients), fp = data.formula.power[String(patients)];
  out.push(`<line class="cursor" x1="${f(x(patients))}" x2="${f(x(patients))}" y1="${T}" y2="${H - B}"/>`,
    `<circle class="fdot" cx="${f(x(patients))}" cy="${f(y(fp))}" r="5"/>`);
  // A simulated point is a filled dot; an interpolated value is a ring, so the two never look alike.
  if (!e.missing) out.push(`<circle class="sdot${e.simulated ? "" : " interp"}" cx="${f(x(patients))}" cy="${f(y(e.power))}" r="5"/>`);
  return out.join("");
}
