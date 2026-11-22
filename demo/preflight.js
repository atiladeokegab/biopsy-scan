// "Preflight your study" (data/preflight.json, built by collect_preflight.py from two committed power files).
// Pure, so preflight.test.mjs checks every number before it reaches the page. No network, no simulation: a simulated
// point is shown as stored; between two points it is linear in patients and labelled ESTIMATE (interpolated);
// outside a design's simulated range, or for a window and mis-call pair nobody simulated, there is no number.
const pct = (v) => (v >= 0.995 && v < 1 ? ">99%" : `${Math.round(v * 100)}%`); // never round a near-certainty up to 100%

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

// The panel's chart as SVG markup, numbers only: the formula's curve, the chosen design's simulated points joined
// straight (the interpolation the panel uses) inside their interval band, and a cursor at the chosen patients.
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
  if (!e.missing) out.push(`<circle class="sdot" cx="${f(x(patients))}" cy="${f(y(e.power))}" r="5"/>`);
  return out.join("");
}
