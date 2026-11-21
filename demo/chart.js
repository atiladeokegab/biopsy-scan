// The one chart: power against patients, one line per regions value, at the chosen point's
// field of view and mis-call rate. White ring: the current design. Green: our cohort.
// Canvas and WebGL cannot read var(); they take colours from the same CSS tokens through this.
export const tok = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const pct = (x) => `${Math.round(x * 100)}%`;
export const fovLabel = (v) => (v === "full" ? "full region" : `${v} µm`);
const plural = (k) => `${k} region${k > 1 ? "s" : ""}`;

export function drawChart(box, grid, sel) {
  const W = Math.max(300, box.clientWidth), H = W < 560 ? 280 : 340;
  const m = { l: 44, r: W < 560 ? 64 : 90, t: 18, b: 40 };
  const xs = grid.axes.patients, x1 = Math.max(...xs);
  const X = (v) => m.l + (v / x1) * (W - m.l - m.r);
  const Y = (v) => H - m.b - v * (H - m.t - m.b);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  const el = (tag, attrs, text) => {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    svg.appendChild(e);
    return e;
  };

  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), style: "stroke:var(--line)" });
    el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, pct(t));
  }
  el("line", { x1: m.l, x2: W - m.r, y1: Y(0.8), y2: Y(0.8), style: "stroke:var(--muted)", "stroke-dasharray": "3 4" });
  el("text", { x: m.l + 4, y: Y(0.8) - 6 }, "80% power");
  for (const t of xs.filter((v, i) => W >= 560 || i % 2 === 1 || v === x1)) {
    el("text", { x: X(t), y: H - m.b + 18, "text-anchor": "middle" }, t);
  }
  el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, "patients");

  const slice = (k) =>
    grid.points
      .filter((p) => p.regions === k && p.fov_um === sel.fov_um && p.miscall === sel.miscall)
      .sort((a, b) => a.patients - b.patients);
  const ks = grid.axes.regions, cohort = grid.cohort, lines = [];
  ks.forEach((k, i) => {
    const pts = slice(k);
    if (pts.length < 2) return;
    const on = k === sel.regions;
    const shade = on ? "var(--head)" : "var(--body)";
    const alpha = on ? 1 : 0.45 + (0.35 * i) / Math.max(1, ks.length - 1);
    el("polyline", {
      points: pts.map((p) => `${X(p.patients)},${Y(p.power)}`).join(" "),
      fill: "none", style: `stroke:${shade}`, "stroke-opacity": alpha, "stroke-width": on ? 3 : 1.5, "stroke-linejoin": "round",
    });
    lines.push({ k, y: Y(pts[pts.length - 1].power), shade, alpha, on, pts });
  });
  // Direct labels at the line ends, nudged apart so they never overlap.
  lines.sort((a, b) => a.y - b.y);
  for (let i = 1; i < lines.length; i++) lines[i].y = Math.max(lines[i].y, lines[i - 1].y + 14);
  for (const l of lines) {
    const t = el("text", { x: W - m.r + 8, y: l.y + 4 }, plural(l.k));
    t.style.fill = l.shade;
    t.style.fillOpacity = l.alpha;
    if (l.on) t.style.fontWeight = 700;
  }

  // The current design first, so the cohort dot stays visible when the two coincide.
  el("circle", { cx: X(sel.patients), cy: Y(sel.power), r: 7, style: "fill:var(--bg);stroke:var(--head)", "stroke-width": 2.5 });
  // Our cohort: a vertical marker at its patient count and a dot on its regions line (interpolated).
  const cl = lines.find((l) => l.k === cohort.regions);
  if (cohort.patients <= x1) {
    el("line", { x1: X(cohort.patients), x2: X(cohort.patients), y1: Y(1), y2: Y(0), style: "stroke:var(--accent)", "stroke-dasharray": "2 4" });
    el("text", { x: X(cohort.patients), y: Y(1) - 6, "text-anchor": "middle", style: "fill:var(--accent);font-weight:600" },
      `our cohort · ${cohort.name}, ${cohort.patients}`);
    if (cl) {
      const pts = cl.pts, j = Math.max(0, pts.findIndex((p) => p.patients >= cohort.patients));
      const a = pts[Math.max(0, j - 1)], b = pts[j];
      const yv = b.patients === a.patients ? a.power : a.power + ((cohort.patients - a.patients) / (b.patients - a.patients)) * (b.power - a.power);
      el("circle", { cx: X(cohort.patients), cy: Y(yv), r: 4.5, style: "fill:var(--accent)" });
    }
  }

  svg.setAttribute("aria-label",
    `Power against patients at ${fovLabel(sel.fov_um)} and ${pct(sel.miscall)} mis-called. ` +
    lines.map((l) => `${plural(l.k)}: ${pct(l.pts[l.pts.length - 1].power)} at ${l.pts[l.pts.length - 1].patients} patients`).join("; ") +
    `. Your design: ${pct(sel.power)}.`);
  box.replaceChildren(svg);
  return `Power against patients · ${fovLabel(sel.fov_um)} field of view · ${pct(sel.miscall)} mis-called`;
}
