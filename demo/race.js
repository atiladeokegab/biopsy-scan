// THE RACE: the same N studies on the CPU (one at a time) and the GPU (all at once).
// Each tile is one simulated study: green when it found the pattern, grey when it missed.
// Timing comes only from bench.json. Without a MEASURED bench both lanes still play, but at a fixed
// on-screen pace that is not to scale: no seconds and no ratio are shown, and both say UNMEASURED.
import { tok } from "./chart.js";

const GPU_SHOW_MS = 1400; // on-screen time for the GPU lane; the CPU lane runs on the same scale
const CPU_SHOW_MS = 7000; // how long we let the CPU lane crawl before stopping the clock
const UNMEASURED_CPU_MS = 140; // on-screen ms per CPU study when unmeasured: a display pace, not a measurement

const fmt = (n) => n.toLocaleString("en-GB");
export const duration = (s) =>
  s < 60 ? `${+s.toPrecision(2)} s` : s < 3600 ? `${+(s / 60).toPrecision(2)} min` : `${+(s / 3600).toPrecision(2)} h`;

function layout(canvas, n) {
  const dpr = Math.min(devicePixelRatio, 2);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  // Largest square tile (with a 1px gap) such that n tiles fit in w x h.
  let size = Math.floor(Math.sqrt((w * h) / n));
  while (size > 2 && Math.floor(w / size) * Math.floor(h / size) < n) size--;
  const cols = Math.floor(w / size);
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, size, cols, ox: (w - cols * size) / 2 };
}

function tile(L, i, colour) {
  const gap = L.size > 5 ? 1 : 0.5;
  L.g.fillStyle = colour;
  L.g.fillRect(L.ox + (i % L.cols) * L.size, Math.floor(i / L.cols) * L.size, L.size - gap, L.size - gap);
}

export function createRace(cpuCanvas, gpuCanvas, out) {
  const HIT = tok("--hit"), MISS = tok("--miss"), EMPTY = tok("--tile-empty");
  let timer = 0;

  function run({ studies: n, power, bench, reducedMotion }) {
    cancelAnimationFrame(timer);
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
        `CPU: all cores of ${bench.cpu}, ${duration(bench.cpu_seconds)} · ${fmt(Math.round(ratio))}× faster`
      : "Speeds not to scale until the benchmark is measured.";

    const t0 = performance.now();
    let gpuDone = 0, cpuDone = 0;
    const step = (now) => {
      const t = reducedMotion ? Infinity : now - t0;
      // GPU: studies land in warp-sized batches in a scattered order, all within GPU_SHOW_MS.
      const g = Math.min(n, Math.ceil((t / GPU_SHOW_MS) * n / 32) * 32);
      for (; gpuDone < g; gpuDone++) { const i = order[(gpuDone * 7919) % n]; tile(gpu, i, found[i] ? HIT : MISS); }
      // CPU: one study after another; on the GPU's clock slowed by the measured ratio, else a fixed display pace.
      const tc = Math.min(t, CPU_SHOW_MS);
      const c = Math.min(n, Math.floor(measured ? (tc / (GPU_SHOW_MS * ratio)) * n : tc / UNMEASURED_CPU_MS));
      for (; cpuDone < c; cpuDone++) tile(cpu, cpuDone, found[cpuDone] ? HIT : MISS);
      if (measured) {
        const tGpu = Math.min(t, GPU_SHOW_MS) / GPU_SHOW_MS * bench.gpu_seconds;
        out.gpu.innerHTML = `<b>${fmt(gpuDone)}</b> of ${fmt(n)} studies · ${duration(tGpu)}`;
        out.cpu.innerHTML = `<b>${fmt(cpuDone)}</b> of ${fmt(n)} studies` +
          (t >= CPU_SHOW_MS && cpuDone < n ? ` · needs ${duration(bench.cpu_seconds)} in all` : "");
      } else {
        out.gpu.innerHTML = `<b>${fmt(gpuDone)}</b> of ${fmt(n)} studies <span class="unm">UNMEASURED</span>`;
        out.cpu.innerHTML = `<b>${fmt(cpuDone)}</b> of ${fmt(n)} studies <span class="unm">UNMEASURED</span>`;
      }
      const more = gpuDone < n || (cpuDone < n && t < CPU_SHOW_MS);
      if (more) timer = requestAnimationFrame(step);
    };
    timer = requestAnimationFrame(step);
  }

  return { run };
}
