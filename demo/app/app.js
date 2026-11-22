// Current vs Ours. Every number and name comes from data.json (built by build_data.py from the result files);
// every core is real cohort cells from cells.json, drawn in the browser: raw multiplex from each cell's marker
// values at its real position, segmentation as the Voronoi of the real centroids, clipped to a cell radius,
// coloured by real cell type. Step 2 overlays the DAB signal of real HPA photos (Ruifrok & Johnston 2001 H-DAB).
const $ = (id) => document.getElementById(id);
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const Q = new URLSearchParams(location.search); // ?t=5 freezes the race at 5 s, ?step=2 opens step 2, ?scan=0.6 fixes its scanline (screenshots)
const DPR = Math.min(2, devicePixelRatio || 1);
const EMERALD = "43,212,164";
const FAMILY = { // segmentation colours per cell-type group, Ref 2 style
  tumour: ["#ff9436", "#ff6f4f", "#ffb347", "#f2734b", "#ffc15e", "#ff8a65", "#e9a03b"],
  immune: ["#a3ff3b", "#38d9f5", "#f24bb8", "#ffe14d", "#7cfc9a", "#c77dff"],
  stroma: ["#2fb8c9", "#3d7bd9", "#5fd3b0"],
};
const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const ease = (x) => 1 - (1 - x) ** 3;
let D, C, typeRGB, markerRGB; // data.json, cells.json
let oursFinal = false, t = 0, t0 = 0, running = false, looping = false, step = 1, unlocked = false, step2At = 0;

// ---- one core: nearest-cell raster (the clipped Voronoi), then raw and segmented paints of it ----
function raster(core, S) {
  const [w, h] = core.size_um, Dm = Math.min(w, h), ox = (w - Dm) / 2, oy = (h - Dm) / 2, k = S / Dm, R = S / 2;
  const cells = core.cells.filter((c) => (c[0] - ox - Dm / 2) ** 2 + (c[1] - oy - Dm / 2) ** 2 < (Dm / 2) ** 2);
  const rc = 0.95 * Math.sqrt((Math.PI * R * R) / Math.max(1, cells.length)), rc2 = rc * rc; // clip radius ~ mean cell spacing
  const best = new Float32Array(S * S).fill(rc2), near = new Int32Array(S * S).fill(-1);
  cells.forEach((c, i) => {
    const px = (c[0] - ox) * k, py = (c[1] - oy) * k;
    const x0 = Math.max(0, Math.floor(px - rc)), x1 = Math.min(S - 1, Math.ceil(px + rc));
    const y0 = Math.max(0, Math.floor(py - rc)), y1 = Math.min(S - 1, Math.ceil(py + rc));
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const d = (x + 0.5 - px) ** 2 + (y + 0.5 - py) ** 2, j = y * S + x;
      if (d < best[j]) { best[j] = d; near[j] = i; }
    }
  });
  const inside = (j) => ((j % S) + 0.5 - R) ** 2 + (((j / S) | 0) + 0.5 - R) ** 2 < R * R;
  return { S, cells, rc, rc2, best, near, inside, edge: rc > 3 };
}

function toCanvas(S, fill, blur = 0) {
  const c = document.createElement("canvas"); c.width = c.height = S;
  const img = new ImageData(S, S); fill(img.data);
  if (!blur) { c.getContext("2d").putImageData(img, 0, 0); return c; }
  const tmp = document.createElement("canvas"); tmp.width = tmp.height = S; tmp.getContext("2d").putImageData(img, 0, 0);
  const g = c.getContext("2d"); g.filter = `blur(${blur}px)`; g.drawImage(tmp, 0, 0); return c;
}

function paintSeg(r) {
  const { S, cells, near, best, inside, rc2, edge } = r;
  return toCanvas(S, (d) => {
    for (let j = 0; j < S * S; j++) {
      const i = near[j]; if (i < 0 || !inside(j)) continue;
      const c = typeRGB[cells[i][2]];
      const border = edge && (near[j + 1] !== i || near[j + S] !== i || best[j] > 0.8 * rc2);
      const f = border ? 0.3 : 1;
      d[4 * j] = c[0] * f; d[4 * j + 1] = c[1] * f; d[4 * j + 2] = c[2] * f; d[4 * j + 3] = 245;
    }
  });
}

// raw multiplex: each cell glows in each marker's colour by its real value; nuclear markers stay tight
function paintRaw(r, only = -1) {
  const { S, cells, near, best, inside, rc, rc2 } = r, M = markerRGB.length;
  const gain = cells.map((c) => Array.from({ length: M }, (_, m) => (c[3 + m] / 255) ** 1.6));
  return toCanvas(S, (d) => {
    for (let j = 0; j < S * S; j++) {
      if (!inside(j)) continue;
      const i = near[j];
      let R = 10, G = 16, B = 22, a = only < 0 ? 255 : 0;
      if (i >= 0) {
        const u = best[j] / rc2, nuc = Math.exp(-u * 5), cyt = Math.exp(-u * 1.2) * 0.85;
        for (let m = 0; m < M; m++) {
          if (only >= 0 && m !== only) continue;
          const v = gain[i][m] * (D.nuclear[m] ? nuc : cyt);
          if (only >= 0) { [R, G, B] = markerRGB[m]; a = Math.min(255, 700 * v); continue; } // one stain layer: its colour, alpha by value
          R += markerRGB[m][0] * v; G += markerRGB[m][1] * v; B += markerRGB[m][2] * v;
        }
      }
      d[4 * j] = R; d[4 * j + 1] = G; d[4 * j + 2] = B; d[4 * j + 3] = a;
    }
  }, Math.max(0.6, rc * 0.12)); // soft, like a scan, so the polygons only appear once segmented
}

const cache = new Map(); // key -> { raw, seg }
function core(i, S) {
  const key = i + ":" + S;
  if (!cache.has(key)) { const r = raster(C.cores[i % C.cores.length], S); cache.set(key, { raw: paintRaw(r), seg: paintSeg(r) }); }
  return cache.get(key);
}

// draw one core: raw everywhere, segmentation left of the scanline at p (0..1), glowing line while scanning
function drawCore(g, im, x, y, d, p, line) {
  g.save(); g.beginPath(); g.arc(x + d / 2, y + d / 2, d / 2, 0, 7); g.clip();
  g.drawImage(im.raw, x, y, d, d);
  if (p > 0) { g.beginPath(); g.rect(x, y, d * p, d); g.clip(); g.drawImage(im.seg, x, y, d, d); }
  g.restore();
  g.strokeStyle = "rgba(243,239,230,.10)"; g.lineWidth = DPR; g.beginPath(); g.arc(x + d / 2, y + d / 2, d / 2, 0, 7); g.stroke();
  if (line && p > 0 && p < 1 && !REDUCED) scanline(g, x + d * p, y - line, d + 2 * line);
}
function scanline(g, x, y, h) {
  g.save(); g.lineCap = "round";
  for (const [w, a] of [[14, 0.10], [6, 0.25], [2, 1]]) {
    g.strokeStyle = `rgba(${w === 2 ? "150,255,215" : EMERALD},${a})`; g.lineWidth = w * DPR;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x, y + h); g.stroke();
  }
  g.restore();
}

function fit(cv) {
  const w = Math.round(cv.clientWidth * DPR), h = Math.round(cv.clientHeight * DPR);
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const g = cv.getContext("2d"); g.clearRect(0, 0, w, h); return g;
}

// ---- step 1: the race ----
const fmt = (s) => s < 60 ? s.toFixed(1) + " s"
  : s < 3600 ? `${Math.floor(s / 60)} m ${Math.floor(s % 60)} s` : `${Math.floor(s / 3600)} h ${Math.floor(s % 3600 / 60)} m`;
const num = (n) => n.toLocaleString("en-US");

function lane(id, s, total) {
  const real = t / D.race.race_s_per_real_s, done = Math.min(D.studies, Math.floor(real / s + 1e-9)), el = $(id);
  el.querySelector("[data-n]").textContent = `${num(done)} / ${num(D.studies)}`;
  el.querySelector("[data-time]").textContent = `${fmt(Math.min(real, total))} / ${fmt(total)}`;
  el.querySelector("[data-bar]").style.width = (100 * done / D.studies) + "%";
  return { real, done };
}

function drawCurrent() {
  const { real } = lane("current", D.current.s_per_study, D.current.total_s);
  const g = fit(document.querySelector("#current canvas")), W = g.canvas.width, H = g.canvas.height;
  // one worker's view: each core takes the measured per-study worker time (8 workers share the pool rate above)
  const per = D.current.worker_s_per_study, i = Math.floor(real / per), f = (real % per) / per;
  const p = REDUCED ? 0 : f, d = Math.min(W, H) * 0.86, S = Math.min(1100, Math.round(d));
  drawCore(g, core(i, S), (W - d) / 2, (H - d) / 2, d, p, 10 * DPR);
  return S;
}

function grid(W, H, n) {
  let best = null;
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols), cell = Math.min(W / cols, H / rows);
    if (!best || cell > best.cell) best = { cols, rows, cell };
  }
  return best;
}

function drawOurs() {
  const { done } = lane("ours", D.ours.s_per_study, D.ours.total_s), cv = document.querySelector("#ours canvas");
  const key = done >= D.studies && `${cv.clientWidth}x${cv.clientHeight}`;
  if (key && key === oursFinal) return; // finished grid already on screen: no redraw while Current runs on
  oursFinal = key;
  const g = fit(cv), W = g.canvas.width, H = g.canvas.height, n = C.cores.length;
  const pad = 12 * DPR, { cols, rows, cell } = grid(W - 2 * pad, H - 2 * pad, n), d = cell * 0.88, S = Math.round(d);
  const x0 = (W - cols * cell) / 2, y0 = (H - rows * cell) / 2, F = D.race.ours_finish_s, waves = 2;
  const lag = 0.4 * F, span = (F - lag) / waves; // diagonal stagger, then `waves` sweeps per core
  for (let k = 0; k < n; k++) {
    const r = (k / cols) | 0, c = k % cols, o = lag * (r + c) / Math.max(1, rows + cols - 2);
    const s = (t - o) / span;
    const p = done >= D.studies || s >= waves ? 1 : s <= 0 ? 0 : REDUCED ? 0 : s % 1;
    drawCore(g, core(k, S), x0 + c * cell + (cell - d) / 2, y0 + r * cell + (cell - d) / 2, d, p, 3 * DPR);
  }
  if (done >= D.studies && !unlocked) unlock();
}

function unlock() { unlocked = true; $("tab-2").disabled = false; $("to-ihc").hidden = false; }

// ---- step 2: marker stack and routine IHC ----
let layers = null, photos = null;
function drawStack(now) {
  const g = fit($("stack")), W = g.canvas.width, H = g.canvas.height, M = D.markers.length;
  const S = Math.round(Math.min(W * 0.62, H * 0.9));
  if (!layers || layers.S !== S) { const r = raster(C.cores[C.pick], S); layers = { S, img: D.markers.map((_, m) => paintRaw(r, m)) }; }
  const e = REDUCED || Q.has("scan") ? 1 : ease(Math.min(1, (now - step2At) / 900));
  const rx = S / 2, ry = rx * 0.42, gap = Math.min((H - 2 * ry - 40 * DPR) / (M - 1), ry * 0.9) * e;
  const cx = W * 0.4, top = (H - (2 * ry + gap * (M - 1))) / 2 + ry;
  g.font = `500 ${15 * DPR}px Inter, system-ui, sans-serif`; g.textBaseline = "middle";
  for (let l = 0; l < M; l++) { // top layer first in the list; the first marker sits at the bottom
    const m = M - 1 - l, y = top + l * gap, kept = D.kept.includes(D.markers[m]);
    g.save(); g.translate(cx, y); g.rotate(-0.06); g.scale(1, ry / rx);
    g.globalCompositeOperation = "lighter"; g.globalAlpha = kept ? 1 : 0.22;
    g.drawImage(layers.img[m], -rx, -rx, 2 * rx, 2 * rx);
    g.globalCompositeOperation = "source-over"; g.globalAlpha = kept ? 0.35 : 0.12;
    g.strokeStyle = kept ? D.marker_colors[m] : "#F3EFE6"; g.lineWidth = DPR * rx / ry;
    g.beginPath(); g.arc(0, 0, rx, 0, 7); g.stroke(); g.restore();
    const lx = cx + rx + 28 * DPR, col = kept ? "#F3EFE6" : "rgba(243,239,230,.38)";
    g.strokeStyle = kept ? "rgba(243,239,230,.5)" : "rgba(243,239,230,.15)"; g.lineWidth = DPR;
    g.beginPath(); g.moveTo(cx + rx * 0.96, y); g.lineTo(lx - 10 * DPR, y); g.stroke();
    g.globalAlpha = kept ? 1 : 0.4; g.fillStyle = D.marker_colors[m];
    g.beginPath(); g.arc(lx, y, 5 * DPR, 0, 7); g.fill(); g.globalAlpha = 1;
    g.fillStyle = col; g.fillText(D.markers[m], lx + 14 * DPR, y);
  }
}

// DAB per pixel by colour deconvolution, smoothed into a density, then a green-to-yellow heat
function heat(img) {
  const N = 256, c = document.createElement("canvas"); c.width = c.height = N;
  const g = c.getContext("2d"); g.drawImage(img, 0, 0, N, N);
  const px = g.getImageData(0, 0, N, N), d = px.data;
  const norm = (v) => { const l = Math.hypot(...v); return v.map((x) => x / l); };
  const [a, b, cc, dd, e, f, gg, h, i] = [norm([0.65, 0.70, 0.29]), norm([0.07, 0.99, 0.11]), norm([0.27, 0.57, 0.78])].flat();
  const det = a * (e * i - f * h) - b * (dd * i - f * gg) + cc * (dd * h - e * gg);
  const dab = [(b * f - cc * e) / det, -(a * f - cc * dd) / det, (a * e - b * dd) / det]; // third column of the inverse
  for (let k = 0; k < N * N; k++) {
    let A = 0; for (let j = 0; j < 3; j++) A += -Math.log10((d[4 * k + j] + 1) / 256) * dab[j];
    const v = Math.max(0, Math.min(255, (A - 0.15) * 400)); d[4 * k] = d[4 * k + 1] = d[4 * k + 2] = v; d[4 * k + 3] = 255;
  }
  g.putImageData(px, 0, 0);
  const s = document.createElement("canvas"); s.width = s.height = N;
  const sg = s.getContext("2d"); sg.filter = "blur(5px)"; sg.drawImage(c, 0, 0);
  const o = sg.getImageData(0, 0, N, N), od = o.data;
  const top = Math.max(32, [...od.filter((_, k) => k % 4 === 0)].sort((x, y) => x - y)[Math.floor(N * N * 0.99)]); // this photo's own 99th percentile
  for (let k = 0; k < N * N; k++) {
    const v = Math.min(1, od[4 * k] / top), u = Math.max(0, (v - 0.25) / 0.75) ** 1.4;
    od[4 * k] = 30 + 225 * u; od[4 * k + 1] = 170 + 60 * u; od[4 * k + 2] = 110 - 80 * u;
    od[4 * k + 3] = v < 0.18 ? 0 : 175 * Math.min(1, (v - 0.18) / 0.25);
  }
  sg.filter = "none"; sg.putImageData(o, 0, 0); return s;
}

function drawIhc(now) {
  const g = fit($("ihc")), W = g.canvas.width, H = g.canvas.height;
  if (!photos) return;
  const n = photos.length, label = 54 * DPR, d = Math.min((W - 26 * DPR * (n + 1)) / n, H - label - 40 * DPR);
  const y = (H - d - label) / 2 + label, scan = Q.has("scan") ? +Q.get("scan") : REDUCED ? 1 : 0.5 - 0.42 * Math.cos(((now - step2At) / 6000) * 2 * Math.PI);
  photos.forEach((ph, k) => {
    const x = (W / n) * k + (W / n - d) / 2, cx = x + d / 2;
    g.save(); g.beginPath(); g.arc(cx, y + d / 2, d / 2, 0, 7); g.clip();
    g.fillStyle = "#fff"; g.fillRect(x, y, d, d); g.drawImage(ph.img, x - d * 0.02, y - d * 0.02, d * 1.04, d * 1.04);
    g.beginPath(); g.rect(x, y, d * scan, d); g.clip(); g.drawImage(ph.heat, x - d * 0.02, y - d * 0.02, d * 1.04, d * 1.04);
    g.restore();
    g.strokeStyle = "rgba(243,239,230,.55)"; g.lineWidth = 3 * DPR; g.beginPath(); g.arc(cx, y + d / 2, d / 2, 0, 7); g.stroke();
    if (scan > 0 && scan < 1 && !REDUCED) scanline(g, x + d * scan, y - 6 * DPR, d + 12 * DPR);
    g.textAlign = "center"; g.textBaseline = "alphabetic";
    g.fillStyle = "#F3EFE6"; g.font = `600 ${20 * DPR}px Inter, system-ui, sans-serif`; g.fillText(ph.gene, cx, y - 30 * DPR);
    g.fillStyle = "#7D8796"; g.font = `${14 * DPR}px Inter, system-ui, sans-serif`; g.fillText("routine IHC", cx, y - 10 * DPR);
  });
}

// ---- loop and controls ----
function frame(now) {
  if (running) { // rAF time can precede the click, so clamp at 0
    t = Math.min(D.race.seconds, Math.max(0, now - t0) / 1e3);
    if (t >= D.race.seconds) { running = false; $("run").disabled = false; }
  }
  $("clock").textContent = `${t.toFixed(1)}s / ${D.race.seconds.toFixed(1)}s`;
  if (step === 1) { drawCurrent(); drawOurs(); } else { drawStack(now); drawIhc(now); }
  looping = running || (step === 2 && !REDUCED && !Q.has("scan"));
  if (looping) requestAnimationFrame(frame);
}
const kick = () => { if (!looping) { looping = true; requestAnimationFrame(frame); } };

function show(n) {
  step = n; step2At = performance.now();
  for (const k of [1, 2]) { $("tab-" + k).setAttribute("aria-selected", k === n); $("step-" + k).hidden = k !== n; }
  kick();
}
function run() { show(1); t = 0; t0 = performance.now(); running = true; $("run").disabled = true; kick(); }
function reset() { running = false; t = 0; $("run").disabled = false; show(1); }

async function start() {
  [D, C] = await Promise.all(["data.json", "cells.json"].map((f) => fetch(f).then((r) => { if (!r.ok) throw new Error(f + " " + r.status); return r.json(); })));
  markerRGB = D.marker_colors.map(rgb);
  const used = {}; typeRGB = C.type_groups.map((grp) => rgb(FAMILY[grp][(used[grp] = (used[grp] ?? -1) + 1) % FAMILY[grp].length]));
  $("run").textContent = `Run ${D.race.seconds}s Speed Race`;
  for (const id of ["current", "ours"]) document.querySelector(`#${id} canvas`).title = D.cells_credit;
  $("stack").title = D.cells_credit; $("ihc").title = D.ihc.map((p) => `${p.gene}: ${p.credit}`).join("\n");
  $("tab-1").onclick = () => show(1); $("tab-2").onclick = () => unlocked && show(2); $("to-ihc").onclick = () => show(2);
  $("run").onclick = run; $("reset").onclick = reset;
  addEventListener("resize", () => { cache.clear(); oursFinal = false; kick(); });
  photos = await Promise.all(D.ihc.map(async (p) => {
    const img = new Image(); img.src = p.photo; await img.decode(); return { gene: p.gene, img, heat: heat(img) };
  }));
  // prerender every study Current reaches in the race, so no frame stalls on a new core
  const S = drawCurrent(), reach = Math.ceil(D.race.seconds / D.race.race_s_per_real_s / D.current.worker_s_per_study);
  for (let i = 0; i <= reach; i++) core(i, S);
  if (Q.has("t")) { t = Math.min(D.race.seconds, +Q.get("t")); }
  if (Q.get("step") === "2") { unlock(); show(2); } else show(1);
  $("run").disabled = false;
}
start().catch((e) => { console.error(e); $("run").textContent = "Data failed to load"; });
