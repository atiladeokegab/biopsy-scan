// Replays measured CPU and GPU utilisation (data.json, built by monitor/build.py) on the race's clock.
// Each value is the last recorded sample at or before the replay time; nothing is interpolated.
const $ = (id) => document.getElementById(id);
const Q = new URLSearchParams(location.search);
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const INK = "#F3EFE6", MUTED = "#7D8796", HAIR = "rgba(243,239,230,.12)";
const WINDOW_RACE_S = 3; // sparkline span, in race seconds
let D, lanes = [], t = 0, t0 = 0, raf = 0, lastDrawn = -1, ymax = 1;

const fmt = (s) => s < 60 ? `${s.toFixed(1)} s` : s < 3600 ? `${Math.floor(s / 60)} m ${String(Math.floor(s % 60)).padStart(2, "0")} s`
  : `${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")} m`;
const at = (rows, x) => { // index of the last sample with time <= x, or -1
  let lo = 0, hi = rows.length - 1, k = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (rows[m][0] <= x) { k = m; lo = m + 1; } else hi = m - 1; }
  return k;
};
const fit = (c) => {
  const r = c.getBoundingClientRect(), dpr = devicePixelRatio || 1, w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return [c.getContext("2d"), w, h, dpr];
};

function gauge(c, frac, value, unit, label, color) {
  const [g, W, H, dpr] = fit(c), cx = W / 2, r = Math.min(W / 2, H * 0.62) * 0.78, cy = H * 0.58;
  const a0 = Math.PI * 0.8, a1 = Math.PI * 2.2, lw = Math.max(6, r * 0.14);
  g.clearRect(0, 0, W, H); g.lineCap = "round"; g.lineWidth = lw;
  g.strokeStyle = HAIR; g.beginPath(); g.arc(cx, cy, r, a0, a1); g.stroke();
  if (frac > 0.002) {
    g.strokeStyle = color; g.shadowColor = color; g.shadowBlur = 14 * dpr;
    g.beginPath(); g.arc(cx, cy, r, a0, a0 + (a1 - a0) * Math.min(1, frac)); g.stroke(); g.shadowBlur = 0;
  }
  g.textAlign = "center"; g.textBaseline = "alphabetic";
  g.fillStyle = INK; g.font = `${Math.round(r * 0.42)}px "JetBrains Mono", monospace`; g.fillText(value, cx, cy + r * 0.12);
  g.fillStyle = MUTED; g.font = `${Math.round(Math.max(11 * dpr, r * 0.13))}px "JetBrains Mono", monospace`;
  g.fillText(unit, cx, cy + r * 0.38);
  g.fillStyle = color; g.fillText(label, cx, H - Math.max(10 * dpr, H * 0.06));
}

function spark(L, real) {
  const [g, W, H, dpr] = fit(L.spark), pad = 6 * dpr, span = WINDOW_RACE_S / D.race.race_s_per_real_s;
  const end = Math.min(real, L.endT), x0 = Math.max(0, end - span); // the window stops when the recording ends
  const x = (s) => ((s - x0) / span) * W, y = (f) => pad + (H - 2 * pad) * (1 - f);
  g.clearRect(0, 0, W, H); g.strokeStyle = HAIR; g.lineWidth = dpr;
  for (const f of [0, 0.5, 1]) { g.beginPath(); g.moveTo(0, y(f)); g.lineTo(W, y(f)); g.stroke(); }
  const line = (rows, max, color, fill) => {
    const lim = end; let first = null, lx = 0;
    g.beginPath();
    for (const [s, v] of rows) {
      if (s < x0 - 0.2) continue; if (s > lim) break;
      const px = x(s), py = y(Math.min(1, v / max));
      first === null ? (first = px, g.moveTo(px, py)) : g.lineTo(px, py); lx = px;
    }
    if (first === null) return;
    g.strokeStyle = color; g.lineWidth = 1.6 * dpr; g.stroke();
    if (fill) { g.lineTo(lx, y(0)); g.lineTo(first, y(0)); g.closePath(); g.globalAlpha = 0.16; g.fillStyle = color; g.fill(); g.globalAlpha = 1; }
  };
  line(L.d.gpu_util, 100, INK, false);
  line(L.d.cpu, ymax, L.color, true);
  g.font = `${11 * dpr}px "JetBrains Mono", monospace`; g.textBaseline = "top";
  g.fillStyle = L.color; g.textAlign = "left"; g.fillText(`CPU ${ymax} cores`, 8 * dpr, 6 * dpr);
  g.fillStyle = MUTED; g.textAlign = "right"; g.fillText("GPU 100%", W - 8 * dpr, 6 * dpr);
}

function render() {
  const real = t / D.race.race_s_per_real_s;
  $("clock").textContent = `${fmt(real)}`;
  for (const L of lanes) {
    const lr = real, over = lr > L.endT; // past the recording: empty gauges, not the last value
    const k = over ? -1 : at(L.d.cpu, lr), j = over ? -1 : at(L.d.gpu_util, lr);
    const cores = k < 0 ? 0 : L.d.cpu[k][1], util = j < 0 ? 0 : L.d.gpu_util[j][1];
    gauge(L.cpu, cores / ymax, cores.toFixed(1), "cores", "CPU", L.color);
    gauge(L.gpu, util / 100, `${Math.round(util)}%`, L.d.gpu || "GPU", "GPU", INK);
    L.time.textContent = `${fmt(Math.min(Math.max(0, lr), L.d.wall_s))} / ${fmt(L.d.wall_s)}`;
    spark(L, real);
  }
}

const HOLD_S = 1; // pause on the last frame before the loop restarts
function frame(now) {
  const el = Math.max(0, now - t0) / 1e3; // rAF time can precede t0, so clamp at 0
  if (el >= D.race.seconds + HOLD_S) { t0 = now; return frame(now); }
  t = Math.min(D.race.seconds, el);
  if (!REDUCED || Math.floor(t) !== lastDrawn) { lastDrawn = Math.floor(t); render(); } // reduced motion: 1 update/s
  raf = requestAnimationFrame(frame);
}
function restart() {
  if (!D || Q.has("t")) return;
  cancelAnimationFrame(raf); t = 0; lastDrawn = -1; t0 = performance.now(); raf = requestAnimationFrame(frame);
}

fetch("data.json").then((r) => r.json()).then((d) => {
  D = d;
  ymax = Math.ceil(Math.max(...Object.values(D.lanes).flatMap((l) => l.cpu.map((r) => r[1]))));
  for (const id of ["current", "ours"]) {
    const el = $(id), d = D.lanes[id], q = (s) => el.querySelector(s);
    q("[data-name]").textContent = d.name;
    q("[data-studies]").textContent = `${d.studies.toLocaleString("en-GB")} studies`;
    lanes.push({ d, color: getComputedStyle(el).getPropertyValue("--c").trim(), cpu: q("[data-cpu]"), gpu: q("[data-gpu]"),
      spark: q(".spark canvas"), time: q("[data-time]"), endT: Math.max(d.cpu.at(-1)[0], d.gpu_util.at(-1)[0]) });
  }
  if (Q.has("t")) { t = Math.min(D.race.seconds, +Q.get("t")); render(); return; } // frozen frame for screenshots
  restart();
});
$("reset").addEventListener("click", restart);
addEventListener("keydown", (e) => { if (e.key === "r" || e.key === "R") restart(); });
addEventListener("resize", () => D && render());
