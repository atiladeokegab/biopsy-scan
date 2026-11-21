// THE RACE: the same N studies on the CPU (all cores, Numba) and the GPU (batched studies).
// Each tile is one simulated study: green when it found the pattern, grey when it missed.
// Timing comes only from bench.json. Without a MEASURED bench both lanes still play, but at a fixed
// on-screen pace that is not to scale: no seconds and no ratio are shown, and both say UNMEASURED.
import { tok } from "./chart.js";

const GPU_SHOW_MS = 1400; // on-screen time for the GPU lane; the CPU lane runs on the same scale
const CPU_SHOW_MS = 7000; // how long we let the CPU lane crawl before stopping the clock
const UNMEASURED_CPU_MS = 140; // on-screen ms per CPU study when unmeasured: a display pace, not a measurement

// Which way the measured comparison points: a slower GPU must never read as "1× faster".
const x = (v) => `${v < 10 ? v.toFixed(1) : fmt(Math.round(v))}×`;
export function speedWord(cpuSeconds, gpuSeconds) {
  const r = cpuSeconds / gpuSeconds;
  return r > 1.05 ? `GPU ${x(r)} faster` : r < 0.95 ? `GPU ${x(1 / r)} slower` : "about the same";
}

const fmt = (n) => n.toLocaleString("en-GB");
// The race plays only when "Run the studies" is pressed; a changed design clears the lanes, so no stale run shows.
export const raceAction = ({ pressed, changed }) => (pressed ? "run" : changed ? "clear" : null);

export const duration = (s) =>
  s < 60 ? `${+s.toPrecision(2)} s` : s < 3600 ? `${+(s / 60).toPrecision(2)} min` : `${+(s / 3600).toPrecision(2)} h`;

function layout(canvas, n) {
  const dpr = Math.min(devicePixelRatio, 2);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  // Near-square tiles that fill the whole lane; filled column by column, so both lanes reveal
  // left to right (docs/design.md, motion 2).
  const rows = Math.max(1, Math.round(Math.sqrt((n * h) / w))), cols = Math.ceil(n / rows);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, tw: w / cols, th: h / rows, rows };
}

function tile(L, i, colour) {
  const gap = Math.min(L.tw, L.th) > 5 ? 1 : 0.5;
  L.g.fillStyle = colour;
  L.g.fillRect(Math.floor(i / L.rows) * L.tw, (i % L.rows) * L.th, L.tw - gap, L.th - gap);
}

export function createRace(cpuCanvas, gpuCanvas, out) {
  const HIT = tok("--accent"), MISS = tok("--tile-miss"), EMPTY = tok("--surface-elevated");
  let timer = 0, last = null, lastN = 0;

  function clear(n) {
    cancelAnimationFrame(timer);
    last = null; lastN = n;
    const cpu = layout(cpuCanvas, n), gpu = layout(gpuCanvas, n);
    for (let i = 0; i < n; i++) { tile(cpu, i, EMPTY); tile(gpu, i, EMPTY); }
    out.cpu.innerHTML = out.gpu.innerHTML = `<b>0</b> of ${fmt(n)} studies`;
    out.speed.textContent = "";
  }

  function run(args) {
    const { studies: n, power, bench, reducedMotion } = args;
    cancelAnimationFrame(timer);
    last = args;
    // Study outcomes: a fixed order per run, with round(power * n) hits, the same on both lanes.
    const hits = Math.round(power * n);
    const found = new Uint8Array(n);
    const order = Array.from({ length: n }, (_, i) => i);
    let s = 12345;
    for (let i = n - 1; i > 0; i--) {
      s = (s * 1103515245 + 12345) >>> 0;
      const j = s % (i + 1);
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (let i = 0; i < hits; i++) found[order[i]] = 1;

    const cpu = layout(cpuCanvas, n), gpu = layout(gpuCanvas, n);
    for (let i = 0; i < n; i++) { tile(cpu, i, EMPTY); tile(gpu, i, EMPTY); }
    // No speed-up without the hardware it was measured on (check_data.py enforces the same).
    const measured = bench && bench.status === "MEASURED" && !!bench.cpu && !!bench.gpu;
    const ratio = measured ? bench.cpu_seconds / bench.gpu_seconds : null;

    out.speed.textContent = measured
      ? `Measured on one design point · GPU: ${bench.gpu}, ${duration(bench.gpu_seconds)} · ` +
        `CPU: all cores of ${bench.cpu}, ${duration(bench.cpu_seconds)} · ${speedWord(bench.cpu_seconds, bench.gpu_seconds)}`
      : bench && bench.error ? "Bench timing unavailable; grid power estimates still active."
      : "Measured CPU/GPU timing pending bench run. Speeds on screen are not to scale.";

    const t0 = performance.now();
    let gpuDone = 0, cpuDone = 0;
    const step = (now) => {
      const t = reducedMotion ? Infinity : now - t0;
      // GPU: studies land in warp-sized batches of 32, sweeping left to right, all within GPU_SHOW_MS.
      const g = Math.min(n, Math.ceil((t / GPU_SHOW_MS) * n / 32) * 32);
      for (; gpuDone < g; gpuDone++) tile(gpu, gpuDone, found[gpuDone] ? HIT : MISS);
      // CPU: all cores; on the GPU's clock slowed by the measured ratio, else a fixed display pace.
      const tc = Math.min(t, CPU_SHOW_MS);
      const c = Math.min(n, Math.floor(measured ? (tc / (GPU_SHOW_MS * ratio)) * n : tc / UNMEASURED_CPU_MS));
      for (; cpuDone < c; cpuDone++) tile(cpu, cpuDone, found[cpuDone] ? HIT : MISS);
      if (measured) {
        const tGpu = Math.min(t, GPU_SHOW_MS) / GPU_SHOW_MS * bench.gpu_seconds;
        out.gpu.innerHTML = `<b>${fmt(gpuDone)}</b> of ${fmt(n)} studies · ${duration(tGpu)}`;
        out.cpu.innerHTML = `<b>${fmt(cpuDone)}</b> of ${fmt(n)} studies` +
          (t >= CPU_SHOW_MS && cpuDone < n ? ` · needs ${duration(bench.cpu_seconds)} in all` : "");
      } else {
        out.gpu.innerHTML = `<b>${fmt(gpuDone)}</b> of ${fmt(n)} studies <span class="tag">UNMEASURED</span>`;
        out.cpu.innerHTML = `<b>${fmt(cpuDone)}</b> of ${fmt(n)} studies <span class="tag">UNMEASURED</span>`;
      }
      const more = gpuDone < n || (cpuDone < n && t < CPU_SHOW_MS);
      if (more) timer = requestAnimationFrame(step);
    };
    timer = requestAnimationFrame(step);
  }

  // Resize: repaint at the new size, never replaying the race.
  const redraw = () => (last ? run({ ...last, reducedMotion: true }) : clear(lastN));

  return { run, clear, redraw };
}

// THE SPEED RACE (data/baseline.json, filled from bench/results by collect_race.py): every row times the same unit,
// one design point of 1,000 simulated studies. Each bar fills at its measured compute rate, scaled so the slowest
// timed row takes RACE_MS on screen; its real seconds print at its finish, start-up shown apart and never counted.
// A row without a time is PENDING: no bar, no motion, no number. No ratio is ever printed, and the workflow
// footnote (different work: the R-14 Cox screen) is seconds only.
export const RACE_MS = 10000;
const timed = (r) => Number.isFinite(r.seconds) && r.seconds > 0;
export function raceState(rows, t, total = RACE_MS) {
  const max = Math.max(...rows.filter(timed).map((r) => r.seconds));
  return rows.map((r) => {
    if (!timed(r)) return { frac: 0, clock: null, done: false };
    const frac = Math.min(1, t / ((r.seconds / max) * total));
    return { frac, clock: frac * r.seconds, done: frac === 1 };
  });
}

const esc = (v) => String(v).replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
const secs = (s) => (s < 600 ? `${s < 10 ? s.toFixed(2) : s.toFixed(1)} s` : duration(s));

const where = (r) => [r.hardware, r.gpu || (r.cores ? `${r.cores} core${r.cores > 1 ? "s" : ""}` : null)]
  .filter(Boolean).map(esc).join(", ");

// Every dynamic string is escaped; a row whose seconds are not a positive finite number is drawn as untimed.
export function speedRaceHTML(data) {
  return `<ol class="sr">${data.rows.map((r) => {
    const tag = `<span class="tag${r.label === "MEASURED" ? " ok" : ""}">${esc(r.label)}</span>`;
    const src = [r.source, r.note, r.startup_s > 0 ? `+ ${secs(r.startup_s)} start-up, not counted` : null]
      .filter(Boolean).map(esc).join(" · ");
    if (!timed(r)) // no bar, no number: the track only says where the time will come from
      return `<li class="pending"><span class="sr-name">${esc(r.method)}</span><span class="sr-time">${tag}</span>` +
        `<span class="sr-track">not yet timed · ${src}</span></li>`;
    return `<li class="${r.role === "baseline" ? "baseline" : "screen"}"><span class="sr-name">${esc(r.method)}<small>${where(r)}</small></span>` +
      `<span class="sr-time">${tag}<b></b></span><span class="sr-track"><i></i></span><small class="sr-src">${src}</small></li>`;
  }).join("")}</ol>`;
}

// Different work from the race, so seconds only: never a bar, never a ratio against a race row.
export function workflowHTML(w) {
  if (!w || !timed(w)) return "";
  return `Workflow comparison, not part of the race: ${esc(w.method)}, ${secs(w.seconds)} on ${where(w)} ` +
    `<span class="tag${w.label === "MEASURED" ? " ok" : ""}">${esc(w.label)}</span>` +
    `${w.note ? ` (${esc(w.note)})` : ""} · ${esc(w.source)}`;
}

export function createSpeedRace(el, data, { reducedMotion = false } = {}) {
  el.innerHTML = speedRaceHTML(data);
  const lanes = [...el.querySelectorAll("li")];
  let timer = 0;

  function paint(t) {
    raceState(data.rows, t).forEach((st, i) => {
      if (st.clock == null) return;
      lanes[i].querySelector("i").style.width = `${st.frac * 100}%`;
      const b = lanes[i].querySelector("b");
      b.textContent = st.frac === 0 ? "" : secs(st.done ? data.rows[i].seconds : st.clock);
      lanes[i].classList.toggle("done", st.done);
    });
  }

  function run() {
    cancelAnimationFrame(timer);
    const t0 = performance.now();
    const step = (now) => {
      const t = reducedMotion ? Infinity : now - t0;
      paint(t);
      if (t < RACE_MS) timer = requestAnimationFrame(step);
    };
    paint(0);
    timer = requestAnimationFrame(step);
  }

  return { run };
}
