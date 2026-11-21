// THE PATTERN (data/biomarker.json, build-8's #82): one real Charville region per KM arm, PD1+ macrophages
// lit up inside the 100 µm reach of Tumor 5 cells, and the cohort's Kaplan-Meier split on the pattern score.
import { tok } from "./chart.js";

// #82's columnar region: type 0 Tumor 5, 1 macrophage, 2 other; pd1_pos per cell (display rule in the file).
export function drawRegion(canvas, r) {
  const { x, y, type, pd1_pos } = r.cells;
  const k = (i) => (type[i] === 0 ? "t5" : type[i] === 1 ? (pd1_pos[i] ? "mac_pd1" : "mac") : "other");
  drawPatch(canvas, { window_um: Math.max(r.width_um, r.height_um), radius_um: 100,
    cells: x.map((xi, i) => ({ x: xi, y: y[i], k: k(i) })) });
}

function drawPatch(canvas, b) {
  const dpr = Math.min(devicePixelRatio, 2);
  const w = canvas.clientWidth, h = canvas.clientHeight, side = Math.min(w, h);
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, (w - side) / 2 * dpr, (h - side) / 2 * dpr);
  const s = side / b.window_um, R = b.radius_um * s;
  const cells = (k) => b.cells.filter((c) => c.k === k);
  // The 100 µm reach: one union of rings, filled once so overlaps do not stack.
  g.beginPath();
  for (const c of cells("t5")) { g.moveTo(c.x * s + R, c.y * s); g.arc(c.x * s, c.y * s, R, 0, 2 * Math.PI); }
  g.fillStyle = "rgba(43, 212, 164, 0.10)"; g.fill();
  g.strokeStyle = "rgba(43, 212, 164, 0.35)"; g.lineWidth = 1; g.stroke();
  const dot = (k, r, colour, alpha = 1) => {
    g.globalAlpha = alpha; g.fillStyle = colour;
    for (const c of cells(k)) { g.beginPath(); g.arc(c.x * s, c.y * s, r, 0, 2 * Math.PI); g.fill(); }
    g.globalAlpha = 1;
  };
  dot("other", 1.4, tok("--stroma"), 0.7);
  dot("immune", 1.4, tok("--stroma"), 0.9);
  dot("tumour", 1.6, tok("--tumour"), 0.45);
  dot("mac", 2.2, tok("--immune"), 0.55);
  dot("t5", 3.4, tok("--tumour"));
  g.shadowColor = tok("--accent"); g.shadowBlur = 8;
  dot("mac_pd1", 3, tok("--accent"));
  g.shadowBlur = 0;
}

// Kaplan-Meier steps as an SVG path; time on x (years), survival 0..1 on y.
export function drawKM(svg, km) {
  const W = 220, H = 150, L = 30, B = 22, T = 6, Rt = 8;
  const tmax = Math.max(...[km.high, km.low].flatMap((a) => a.steps.map((p) => p[0])));
  const x = (t) => L + (t / tmax) * (W - L - Rt), y = (v) => T + (1 - v) * (H - T - B);
  const path = (steps) => steps.reduce((d, [t, v], i) => d + (i ? `H${x(t).toFixed(1)}V${y(v).toFixed(1)}` : `M${x(t).toFixed(1)},${y(v).toFixed(1)}`), "");
  const years = [...Array(Math.floor(tmax / 365.25 / 5) + 1).keys()].map((i) => i * 5);
  svg.innerHTML =
    `<line class="axis" x1="${L}" y1="${H - B}" x2="${W - Rt}" y2="${H - B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${H - B}"/>` +
    [0, 0.5, 1].map((v) => `<text x="${L - 4}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join("") +
    years.map((yr) => `<text x="${x(yr * 365.25)}" y="${H - 6}" text-anchor="middle">${yr}y</text>`).join("") +
    `<path class="lo" d="${path(km.low.steps)}"/><path class="hi" d="${path(km.high.steps)}"/>`;
}
