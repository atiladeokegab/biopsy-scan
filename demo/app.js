// "Can my study find it?": reads data/grid.json (schema in README.md) and looks up the nearest design point.
const $ = (id) => document.getElementById(id);
const FULL_SLIDER = 2250; // the fov slider's last stop means "full region"
const AXES = ["patients", "regions", "fov_um", "miscall"];
let grid = null;
let shown = null; // the grid point on screen, redrawn on resize

const pct = (x) => `${Math.round(x * 100)}%`;
const fovLabel = (v) => (v === "full" ? "full region" : `${v} µm`);
const duration = (s) =>
  s < 60 ? `${+s.toPrecision(2)} s` : s < 3600 ? `${Math.round(s / 60)} min` : `${+(s / 3600).toPrecision(2)} h`;

// Position of value v along an axis list, in steps (fractional between neighbours); "full" sits at its own index.
function axisPos(list, v) {
  if (v === "full") return list.indexOf("full") >= 0 ? list.indexOf("full") : list.length;
  const nums = list.filter((x) => x !== "full");
  if (v <= nums[0]) return 0;
  for (let i = 1; i < nums.length; i++) {
    if (v <= nums[i]) return i - 1 + (v - nums[i - 1]) / (nums[i] - nums[i - 1]);
  }
  return nums.length - 1;
}

function nearest(want) {
  let best = null, bestD = Infinity;
  for (const p of grid.points) {
    const d = AXES.reduce((s, k) => s + (axisPos(grid.axes[k], want[k]) - axisPos(grid.axes[k], p[k])) ** 2, 0);
    if (d < bestD) { bestD = d; best = p; }
  }
  return best;
}

function readSliders() {
  const fov = +$("fov").value;
  return {
    patients: +$("patients").value,
    regions: +$("regions").value,
    fov_um: fov >= FULL_SLIDER ? "full" : fov,
    miscall: +$("miscall").value / 100,
  };
}

function showSliders() {
  const w = readSliders();
  $("patients-out").value = w.patients;
  $("regions-out").value = w.regions;
  $("fov-out").value = fovLabel(w.fov_um);
  $("miscall-out").value = pct(w.miscall);
}

function setSliders(c) {
  $("patients").value = c.patients;
  $("regions").value = c.regions;
  $("fov").value = c.fov_um === "full" ? FULL_SLIDER : c.fov_um;
  $("miscall").value = Math.round(c.miscall * 100);
  showSliders();
}

function verdict(power) {
  if (power >= 0.8) return "Yes. This design finds the pattern in most studies.";
  if (power >= 0.5) return "Maybe. This design finds it in some studies, misses it in many.";
  return "Unlikely. Most studies with this design miss the pattern.";
}

function run() {
  const want = readSliders();
  const p = nearest(want);
  $("power").textContent = pct(p.power);
  $("ident").textContent = pct(p.identification_rate);
  $("verdict").textContent = verdict(p.power);
  const same = AXES.every((k) => want[k] === p[k]);
  $("snap").textContent = same
    ? `Grid point: ${describe(p)}.`
    : `Nearest grid point to what you asked: ${describe(p)}.`;
  const a = $("answer");
  a.classList.remove("flash"); void a.offsetWidth; a.classList.add("flash");
  shown = p;
  drawChart(p);
}

const describe = (p) =>
  `${p.patients} patients · ${p.regions} region${p.regions > 1 ? "s" : ""} · ${fovLabel(p.fov_um)} · ${pct(p.miscall)} mis-called`;

function headline() {
  const h = grid.headline || { status: "UNMEASURED" };
  if (h.studies) $("hl-studies").textContent = h.studies.toLocaleString("en-GB");
  if (h.status === "MEASURED") {
    $("hl-gpu").textContent = duration(h.gpu_seconds);
    $("hl-cpu").textContent = `${duration(h.cpu_seconds)} (${Math.round(h.cpu_seconds / h.gpu_seconds)}× faster)`;
  } else {
    $("hl-badge").hidden = false;
  }
}

// One chart: power vs patients, one line per regions value, at the chosen point's fov and mis-call.
function drawChart(sel) {
  const box = $("chart");
  const W = Math.max(300, box.clientWidth), H = W < 560 ? 280 : 360;
  const m = { l: 44, r: W < 560 ? 64 : 90, t: 16, b: 40 };
  const xs = grid.axes.patients, x0 = 0, x1 = Math.max(...xs);
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
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
    el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: "#22303F" });
    el("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, pct(t));
  }
  el("line", { x1: m.l, x2: W - m.r, y1: Y(0.8), y2: Y(0.8), stroke: "#7D8B9B", "stroke-dasharray": "3 4" });
  el("text", { x: m.l + 4, y: Y(0.8) - 6 }, "80% power");
  for (const t of xs.filter((v, i) => W >= 560 || i % 2 === 1 || v === x1)) {
    el("text", { x: X(t), y: H - m.b + 18, "text-anchor": "middle" }, t);
  }
  el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, "patients");

  const slice = (k) =>
    grid.points
      .filter((p) => p.regions === k && p.fov_um === sel.fov_um && p.miscall === sel.miscall)
      .sort((a, b) => a.patients - b.patients);
  const ks = grid.axes.regions;
  const cohort = grid.cohort;
  const lines = [];
  ks.forEach((k, i) => {
    const pts = slice(k);
    if (pts.length < 2) return;
    const on = k === sel.regions;
    const shade = on ? "#F4F7FA" : `rgba(185,198,212,${0.45 + (0.35 * i) / Math.max(1, ks.length - 1)})`;
    el("polyline", {
      points: pts.map((p) => `${X(p.patients)},${Y(p.power)}`).join(" "),
      fill: "none", stroke: shade, "stroke-width": on ? 3 : 1.5, "stroke-linejoin": "round",
    });
    const last = pts[pts.length - 1];
    lines.push({ k, y: Y(last.power), shade, on, pts });
  });
  // Direct labels at the line ends, nudged apart so they never overlap.
  lines.sort((a, b) => a.y - b.y);
  for (let i = 1; i < lines.length; i++) lines[i].y = Math.max(lines[i].y, lines[i - 1].y + 14);
  for (const l of lines) {
    const t = el("text", { x: W - m.r + 8, y: l.y + 4 }, `${l.k} region${l.k > 1 ? "s" : ""}`);
    t.style.fill = l.shade;
    if (l.on) t.style.fontWeight = 700;
  }

  // Our cohort: a vertical marker at its patient count and a dot on its regions line (interpolated).
  const cl = lines.find((l) => l.k === cohort.regions);
  if (cohort.patients <= x1) {
    el("line", { x1: X(cohort.patients), x2: X(cohort.patients), y1: Y(1), y2: Y(0), stroke: "#2BD4A4", "stroke-dasharray": "2 4" });
    el("text", { x: X(cohort.patients), y: Y(1) - 4, "text-anchor": "middle", style: "fill:#2BD4A4;font-weight:600" },
      `our cohort · ${cohort.name}, ${cohort.patients}`);
    if (cl) {
      const pts = cl.pts, j = pts.findIndex((p) => p.patients >= cohort.patients);
      const a = pts[Math.max(0, j - 1)], b = pts[Math.max(0, j)];
      const yv = b.patients === a.patients ? a.power : a.power + ((cohort.patients - a.patients) / (b.patients - a.patients)) * (b.power - a.power);
      el("circle", { cx: X(cohort.patients), cy: Y(yv), r: 6, fill: "#2BD4A4" });
    }
  }
  // The design the judge asked for.
  el("circle", { cx: X(sel.patients), cy: Y(sel.power), r: 7, fill: "#0B1018", stroke: "#F4F7FA", "stroke-width": 2.5 });

  svg.setAttribute("aria-label",
    `Power against patients at ${fovLabel(sel.fov_um)} and ${pct(sel.miscall)} mis-called. ` +
    lines.map((l) => `${l.k} regions: ${pct(l.pts[l.pts.length - 1].power)} at ${l.pts[l.pts.length - 1].patients} patients`).join("; ") +
    `. Your design: ${pct(sel.power)}.`);
  box.replaceChildren(svg);
  $("chart-title").textContent = `Power against patients · ${fovLabel(sel.fov_um)} field of view · ${pct(sel.miscall)} mis-called`;
  $("note").textContent =
    `HR ${grid.effect_hr_per_sd} per SD planted · ${grid.studies_per_point.toLocaleString("en-GB")} studies per point. ` +
    `White ring: your design. Green: our cohort (${cohort.regions} region${cohort.regions > 1 ? "s" : ""} per patient).` +
    (grid.note ? ` ${grid.note}` : "");
}

async function main() {
  for (const id of ["patients", "regions", "fov", "miscall"]) $(id).addEventListener("input", showSliders);
  $("design").addEventListener("submit", (e) => { e.preventDefault(); if (grid) run(); });
  try {
    const r = await fetch("data/grid.json");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    grid = await r.json();
  } catch (err) {
    $("error").hidden = false;
    $("error").innerHTML =
      "Could not read <code>data/grid.json</code>. Browsers block it from a double-clicked file; " +
      "from the repo root run <code>python3 -m http.server 8000 -d demo</code> and open " +
      "<code>http://localhost:8000</code>.";
    $("run").disabled = true;
    return;
  }
  $("synthetic").hidden = grid.source === "MEASURED";
  headline();
  setSliders(grid.cohort);
  run();
  let t;
  addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => drawChart(shown), 150); });
}

main();
