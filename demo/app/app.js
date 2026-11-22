// One-screen live demo: colour deconvolution (Ruifrok & Johnston 2001, H-DAB) computed in the browser
// from the photo's pixels, revealed tile by tile at a replay of measured pace (Current ≈ 21 s, Ours ≈ 1 s: 21×, 0.881 s vs 0.042 s per evaluated study).
const MODES = {
  current: { label: "CPU · open-source libraries (lab pipeline)", ms: 21000 },
  ours: { label: "Preflight", ms: 1000 },
};
const N = 400, GRID = 16; // analysis resolution and tile grid
const $ = (id) => document.getElementById(id);
const img = $("photo"), cv = $("overlay"), ctx = cv.getContext("2d");
let mode = "ours", running = false, overlay = null;
const done = {};

function inv3(m) {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  return [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((v) => v / det);
}
const norm = (v) => { const l = Math.hypot(...v); return v.map((x) => x / l); };
// Rows: haematoxylin, eosin (residual), DAB stain OD vectors
const M = [norm([0.65, 0.70, 0.29]), norm([0.07, 0.99, 0.11]), norm([0.27, 0.57, 0.78])].flat();
const D = inv3(M);

function clean(mask) { // light blob cleanup: keep a pixel only if ≥4 of its 8 neighbours agree
  const out = new Uint8Array(mask.length);
  for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
    const k = y * N + x; if (!mask[k]) continue;
    let s = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += mask[k + dy * N + dx];
    out[k] = s - 1 >= 4;
  }
  return out;
}

function compute() {
  const c = document.createElement("canvas"); c.width = c.height = N;
  const g = c.getContext("2d"); g.drawImage(img, 0, 0, N, N);
  const px = g.getImageData(0, 0, N, N).data;
  const h = new Uint8Array(N * N), dab = new Uint8Array(N * N);
  for (let k = 0; k < N * N; k++) {
    const od = [0, 1, 2].map((j) => -Math.log10((px[k * 4 + j] + 1) / 256));
    const H = od[0] * D[0] + od[1] * D[3] + od[2] * D[6];
    const A = od[0] * D[2] + od[1] * D[5] + od[2] * D[8];
    dab[k] = A > 0.45 && A > H; h[k] = !dab[k] && H > 0.2;
  }
  const hc = clean(h), dc = clean(dab);
  const o = g.createImageData(N, N);
  for (let k = 0; k < N * N; k++) {
    const col = dc[k] ? [232, 111, 81] : hc[k] ? [43, 212, 164] : null;
    if (col) { o.data.set(col, k * 4); o.data[k * 4 + 3] = 200; }
  }
  return o;
}

function setMode(m) {
  if (running) return;
  mode = m; document.body.dataset.mode = m;
  document.querySelectorAll(".pills button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.mode === m));
  $("mode").textContent = MODES[m].label;
  reset();
}
function reset() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  $("timer").textContent = "0.0 s"; $("prog").style.width = "0";
  const r = $("risk"); r.classList.remove("on"); r.hidden = true;
}

function run() {
  if (running) return;
  reset(); running = true; $("run").disabled = true;
  overlay ||= compute();
  cv.width = cv.height = N;
  const src = document.createElement("canvas"); src.width = src.height = N; src.getContext("2d").putImageData(overlay, 0, 0);
  const tiles = []; for (let i = 0; i < GRID * GRID; i++) tiles.push(i);
  for (let i = tiles.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [tiles[i], tiles[j]] = [tiles[j], tiles[i]]; }
  const T = N / GRID, total = MODES[mode].ms, t0 = performance.now();
  let shown = 0;
  (function frame(now) {
    const f = Math.min(1, (now - t0) / total), want = Math.round(f * tiles.length);
    for (; shown < want; shown++) { const t = tiles[shown], x = (t % GRID) * T, y = ((t / GRID) | 0) * T; ctx.drawImage(src, x, y, T, T, x, y, T, T); }
    $("timer").textContent = ((now - t0) / 1000).toFixed(1) + " s"; $("prog").style.width = f * 100 + "%";
    if (f < 1) return requestAnimationFrame(frame);
    finish((now - t0) / 1000);
  })(t0);
}

function finish(sec) {
  running = false; $("run").disabled = false;
  done[mode] = sec;
  $("results").innerHTML = Object.entries(done).map(([m, s]) => `<span>${m === "ours" ? "Preflight" : "Current"} · ${s.toFixed(1)} s replay</span>`).join("");
  $("fast").hidden = !(done.current && done.ours);
  if (mode === "ours") { const r = $("risk"); r.hidden = false; requestAnimationFrame(() => r.classList.add("on")); }
}

document.querySelectorAll(".pills button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
$("run").addEventListener("click", run);
$("run").disabled = true; // wait for the photo, or the overlay is computed from a blank image
img.decode().then(() => { if (!running) $("run").disabled = false; }, () => { $("run").textContent = "Photo failed to load"; });
const q = new URLSearchParams(location.search).get("mode");
setMode(MODES[q] ? q : "ours");
