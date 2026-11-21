// THE RACE: the same N studies on the CPU (all cores, Numba) and the GPU (batched studies).
// Each tile is one simulated study: green when it found the pattern, grey when it missed.
// Timing comes only from bench.json. Without a MEASURED bench both lanes still play, but at a fixed
// on-screen pace that is not to scale: no seconds and no ratio are shown, and both say UNMEASURED.
import { tok } from "./chart.js";

const GPU_SHOW_MS = 1400; // on-screen time for the GPU lane; the CPU lane runs on the same scale
const CPU_SHOW_MS = 7000; // how long we let the CPU lane crawl before stopping the clock
const UNMEASURED_CPU_MS = 140; // on-screen ms per CPU study when unmeasured: a display pace, not a measurement

// Which way the measured comparison points: a slower GPU must never read as "1× faster".
export function speedWord(cpuSeconds, gpuSeconds) {
  const r = cpuSeconds / gpuSeconds;
  const x = (v) => `${v < 10 ? v.toFixed(1) : fmt(Math.round(v))}×`;
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
