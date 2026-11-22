// One-screen live demo: colour deconvolution (Ruifrok & Johnston 2001, H-DAB) computed in the browser
// from the photo's pixels once, then revealed side by side, tile by tile, at a replay of measured pace
// (Current ≈ 21 s, Ours ≈ 1 s: 21×, 0.881 s vs 0.042 s per evaluated study).
const LANES = { current: 21000, ours: 1000 };
const N = 400, GRID = 16; // analysis resolution and tile grid
const $ = (id) => document.getElementById(id);
const imgs = [...document.querySelectorAll(".frame img")], img = imgs[0];
const FADE = matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 150; // per-tile fade, ms
let running = 0, overlay = null;

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

function lane(id, src, order, t0) {
  const el = $("lane-" + id), cv = el.querySelector("canvas"), ctx = cv.getContext("2d");
  const timer = el.querySelector("[data-timer]"), prog = el.querySelector("[data-prog]"), done = el.querySelector("[data-done]");
  cv.width = cv.height = N; done.hidden = true;
  const T = N / GRID, total = LANES[id], n = order.length, span = Math.max(1, total - FADE);
  let shown = 0;
  return new Promise((resolve) => (function frame(now) {
    const e = Math.max(0, now - t0), f = Math.min(1, e / total);
    // tile i starts at i/n of the span and fades in over FADE ms; redraw only tiles still fading
    for (let i = shown; i < n; i++) {
      const s0 = (i / n) * span; if (e < s0) break;
      const a = FADE ? Math.min(1, (e - s0) / FADE) : 1, t = order[i], x = (t % GRID) * T, y = ((t / GRID) | 0) * T;
      ctx.clearRect(x, y, T, T); ctx.globalAlpha = 1 - (1 - a) ** 3; ctx.drawImage(src, x, y, T, T, x, y, T, T);
      if (a === 1 && i === shown) shown++;
    }
    ctx.globalAlpha = 1;
    timer.textContent = (f * total / 1000).toFixed(1) + " s"; prog.style.width = f * 100 + "%";
    if (shown < n || f < 1) return requestAnimationFrame(frame);
    done.textContent = "Done in " + (total / 1000).toFixed(1) + " s"; done.hidden = false;
    resolve();
  })(t0));
}

async function run() {
  if (running) return;
  running = 1; $("run").disabled = true; thumbs.forEach((b) => (b.disabled = true));
  $("banner").classList.remove("on");
  const r = $("risk"); r.classList.remove("on"); r.hidden = true;
  overlay ||= compute();
  const src = document.createElement("canvas"); src.width = src.height = N; src.getContext("2d").putImageData(overlay, 0, 0);
  const order = []; for (let i = 0; i < GRID * GRID; i++) order.push(i);
  for (let i = order.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [order[i], order[j]] = [order[j], order[i]]; }
  const t0 = performance.now(); // one moment for both lanes
  await Promise.all(Object.keys(LANES).map((id) => lane(id, src, order, t0)));
  $("banner").classList.add("on");
  r.hidden = false; requestAnimationFrame(() => r.classList.add("on"));
  running = 0; $("run").disabled = false; thumbs.forEach((b) => (b.disabled = false));
}

function reset() {
  for (const id of Object.keys(LANES)) {
    const el = $("lane-" + id), cv = el.querySelector("canvas");
    cv.getContext("2d").clearRect(0, 0, cv.width, cv.height); el.querySelector("[data-done]").hidden = true;
    el.querySelector("[data-timer]").textContent = "0.0 s"; el.querySelector("[data-prog]").style.width = "0";
  }
  $("banner").classList.remove("on"); const r = $("risk"); r.classList.remove("on"); r.hidden = true;
}

function load(btn) { // swap both panels to this photo; Run waits until both have decoded
  thumbs.forEach((b) => b.setAttribute("aria-pressed", b === btn));
  overlay = null; reset();
  $("gene").textContent = btn.dataset.gene; $("credit").href = btn.dataset.page;
  const run = $("run"); run.disabled = true; run.textContent = "Run";
  imgs.forEach((i) => (i.src = "../workflow/img/" + btn.dataset.file));
  Promise.all(imgs.map((i) => i.decode())).then(
    () => { if (!running && btn.getAttribute("aria-pressed") === "true") run.disabled = false; },
    () => { if (btn.getAttribute("aria-pressed") === "true") run.textContent = "Photo failed to load"; });
}

const thumbs = [...document.querySelectorAll(".thumbs button")];
thumbs.forEach((b) => b.addEventListener("click", () => !running && load(b)));
$("run").addEventListener("click", run);
load(thumbs[0]);
