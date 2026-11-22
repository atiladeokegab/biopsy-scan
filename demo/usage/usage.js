// Replays measured CPU and GPU utilisation (traces.json, written by bench/modal_usage.py) at real speed.
// Every value shown is a recorded sample: the last one at or before the replay time; nothing is interpolated.
const ARMS = {
  current: { tag: "Current · off-the-shelf", name: "squidpy + lifelines", how: "8 CPU workers; the A100 sits idle", cls: "current" },
  fused: { tag: "Preflight · CPU", name: "Fused kernel", how: "8 CPU workers, same studies" },
  gpu: { tag: "Preflight · GPU", name: "A100 path", how: "one process; re-imaging and scoring on the A100" },
};
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (id) => document.getElementById(id);
const REPO = "https://github.com/atiladeokegab/sota-parallelism-nov21/blob/integration/";
let data, views = [], t0 = 0, raf = 0, last = -1, end = 0, cur = 0, ymax = 1;

const at = (rows, t) => { // last sample with time <= t (rows sorted by time)
  let lo = 0, hi = rows.length - 1, k = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (rows[m][0] <= t) { k = m; lo = m + 1; } else hi = m - 1; }
  return k;
};
const secs = (s) => (s < 10 ? s.toFixed(2) : s.toFixed(1)) + " s";

function build() {
  const tpl = $("arm-tpl").content;
  for (const [key, meta] of Object.entries(ARMS)) {
    const a = data.arms[key]; if (!a) continue;
    const el = tpl.cloneNode(true).firstElementChild, q = (k) => el.querySelector(`[data-k="${k}"]`);
    if (meta.cls) el.classList.add(meta.cls);
    el.setAttribute("aria-label", `${meta.tag}: ${meta.name}`);
    q("tag").textContent = meta.tag; q("name").textContent = meta.name; q("how").textContent = meta.how;
    q("of").textContent = ` / ${a.studies}`;
    q("coreslabel").textContent = `CPU · cores busy of ${a.cores}`;
    q("gpulabel").textContent = `GPU · ${(a.gpu_name || "none").replace(/^NVIDIA /, "")}`;
    const cores = [...Array(a.cores)].map(() => q("cores").appendChild(document.createElement("i")));
    $("arms").appendChild(el);
    const s = a.summary;
    q("mean").textContent = `mean ${s.mean_cpu_cores_busy.toFixed(1)} cores · GPU ${Math.round(s.mean_gpu_util_pct)}%`;
    views.push({ key, a, el, q, cores, canvas: el.querySelector("canvas") });
  }
  ymax = Math.ceil(Math.max(...views.flatMap((v) => v.a.cpu.map((r) => r[1])))); // CPU axis: the busiest sample, in cores
  end = Math.max(...views.map((v) => v.a.wall_s)) + 0.5;
  const w = Object.fromEntries(views.map((v) => [v.key, v.a.wall_s]));
  if (w.current && w.fused) $("lede").innerHTML = `The same ${views[0].a.studies} studies on the same A100 machine: off the shelf takes ` +
    `${secs(w.current)}; Preflight's CPU kernel takes <em>${secs(w.fused)}</em>.`;
  const src = data.source;
  $("caption").innerHTML = `Replay of measured utilisation, sampled every 100 ms (${data.label}) · source <a href="${REPO}${src}">${src}</a> · ` +
    `each container shows ${views[0].a.cores} cores, ${views[0].a.cores_reserved} reserved; time 0 is each arm's first study (load and warm-up not shown) · squidpy and lifelines never touch the GPU · ` +
    "at 8 studies each Preflight worker's start-up dominates; at 3,000 studies the fused kernel runs 0.042 s/study " +
    `(<a href="${REPO}bench/results/modal-design-20261121T205536Z.json">modal-design-20261121T205536Z.json</a>).`;
}

function draw(v, t) {
  const c = v.canvas, r = c.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  if (c.width !== Math.round(r.width * dpr) || c.height !== Math.round(r.height * dpr)) { c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr); }
  const g = c.getContext("2d"), W = c.width, H = c.height, pad = 4 * dpr;
  g.clearRect(0, 0, W, H);
  g.strokeStyle = "rgba(243,239,230,.12)"; g.lineWidth = dpr;
  for (const y of [0, 0.5, 1]) { g.beginPath(); g.moveTo(0, pad + (H - 2 * pad) * y); g.lineTo(W, pad + (H - 2 * pad) * y); g.stroke(); }
  const x = (s) => (Math.max(0, s) / end) * W, y = (p) => pad + (H - 2 * pad) * (1 - p);
  const line = (rows, color, max, fill) => {
    g.strokeStyle = color; g.lineWidth = 1.5 * dpr; g.beginPath(); let first = null, lastX = 0;
    for (const [s, p] of rows) { if (s > t) break; first === null ? (first = x(s), g.moveTo(first, y(p / max))) : g.lineTo(x(s), y(p / max)); lastX = x(s); }
    g.stroke();
    if (fill && first !== null) { g.lineTo(lastX, y(0)); g.lineTo(first, y(0)); g.closePath(); g.globalAlpha = 0.16; g.fillStyle = color; g.fill(); g.globalAlpha = 1; }
  };
  const col = getComputedStyle(v.el).getPropertyValue("--c").trim();
  line(v.a.gpu, "#F3EFE6", 100);
  line(v.a.cpu, col, ymax, true);
  g.font = `${11 * dpr}px "JetBrains Mono", monospace`; g.textBaseline = "top";
  g.fillStyle = col; g.textAlign = "left"; g.fillText(`${ymax} cores`, 2 * dpr, pad + 2 * dpr);
  g.fillStyle = "#7D8796"; g.textAlign = "right"; g.fillText("100% GPU", W - 2 * dpr, pad + 2 * dpr);
  g.textBaseline = "bottom"; g.fillText(`${end.toFixed(0)} s`, W - 2 * dpr, H - pad - 2 * dpr);
  g.textAlign = "left"; g.fillText("0 s", 2 * dpr, H - pad - 2 * dpr);
}

function frame(now) {
  const t = Math.min((now - t0) / 1000, end);
  if (REDUCED && Math.floor(t) === last && t < end) { raf = requestAnimationFrame(frame); return; } // reduced motion: 1 update/s
  last = Math.floor(t);
  render(t);
  if (t < end) raf = requestAnimationFrame(frame);
  else $("start").disabled = false, $("start").textContent = "Replay";
}

function render(t) {
  cur = t;
  $("clock").textContent = t.toFixed(1) + " s";
  for (const v of views) {
    const { a, q } = v, over = t > a.cpu[a.cpu.length - 1][0] + 0.1; // past the recording: show nothing, not the last value
    const k = over ? -1 : at(a.cpu, t), j = over ? -1 : at(a.gpu, t);
    const busy = k < 0 ? 0 : a.cpu[k][1]; // one cell per visible core, filled up to the cores busy
    v.cores.forEach((el, i) => { el.style.opacity = (0.06 + 0.94 * Math.min(1, Math.max(0, busy - i))).toFixed(2); });
    q("cpu").textContent = k < 0 ? (over ? "ended" : "–") : `${busy.toFixed(1)} cores`;
    const gpu = j < 0 ? null : a.gpu[j];
    q("gpu").textContent = gpu ? `${gpu[1]}% · ${(gpu[2] / 1024).toFixed(1)} GiB` : over ? "ended" : "–";
    q("gbar").style.width = (gpu ? gpu[1] : 0) + "%";
    const done = a.done_s.filter((s) => s <= t).length;
    q("done").textContent = done;
    const fin = t >= a.wall_s;
    q("state").textContent = fin ? `done in ${secs(a.wall_s)}` : t > 0 ? "running" : "waiting";
    q("state").classList.toggle("fin", fin);
    draw(v, t);
  }
}

function start() {
  cancelAnimationFrame(raf); last = -1;
  $("start").disabled = true; t0 = performance.now(); raf = requestAnimationFrame(frame);
}

fetch("traces.json").then((r) => r.json()).then((d) => {
  data = d; build(); render(0);
  $("start").disabled = false; $("start").addEventListener("click", start);
  addEventListener("resize", () => views.forEach((v) => draw(v, cur)));
});
