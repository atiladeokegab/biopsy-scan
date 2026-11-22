// Step 2, decode the tumour: the picked core in pseudo-fluorescence; a scan line sweeps across it and leaves
// behind the segmentation map, every measured cell in its type's colour. Then the merged screen result appears.
import { TYPE_COLORS } from "./palette.js";
import { drawCore, glow, sizeCanvas } from "./step1.js";

const SCAN_MS = 2600;
// wide: the core sits right, leaving the left for the result card and the legend; narrow: centred (legend hidden)
const cxOf = (w) => w / 2; // centred; the result card sits top-right, the legend bottom-left

export function mount(el, data) {
  const core = data.tma.cores[data.tma.pick];
  const result = el.querySelector(".wf-result");
  const legend = el.appendChild(document.createElement("ul"));
  legend.className = "wf-legend";
  data.tma.type_names.forEach((t, i) => {
    const li = legend.appendChild(document.createElement("li"));
    li.innerHTML = `<i style="background:${TYPE_COLORS[i]}"></i>`;
    li.append(t);
  });
  let raf = 0, timer = 0, layers = null;
  const layer = (w, h, d, mode, withGlow) => {
    const c = document.createElement("canvas");
    c.width = w * d; c.height = h * d;
    const x = c.getContext("2d");
    x.setTransform(d, 0, 0, d, 0, 0);
    const r = Math.min(w, h) * 0.42;
    drawCore(x, core, cxOf(w, h), h / 2, r, mode);
    if (withGlow) glow(c, x, w, h);
    return c;
  };
  const build = () => {
    const { cv, ctx, w, h } = sizeCanvas(el), d = devicePixelRatio || 1;
    layers = { cv, ctx, w, h, fluor: layer(w, h, d, "fluor", true), seg: layer(w, h, d, "seg", false), r: Math.min(w, h) * 0.42 };
  };
  const frame = (t0) => (now) => {
    const { ctx, w, h, fluor, seg, r } = layers;
    const k = Math.min(1, (now - t0) / SCAN_MS), x = cxOf(w, h) - r + 2 * r * k;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(fluor, 0, 0, w, h);
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, x, h); ctx.clip(); ctx.drawImage(seg, 0, 0, w, h); ctx.restore();
    if (k < 1) {
      ctx.save();
      ctx.strokeStyle = getComputedStyle(el).getPropertyValue("--accent") || "#2BD4A4";
      ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = 18; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, h / 2 - r * 1.08); ctx.lineTo(x, h / 2 + r * 1.08); ctx.stroke();
      ctx.restore();
      raf = requestAnimationFrame(frame(t0));
    }
  };
  addEventListener("resize", () => { if (!el.hidden) { build(); frame(-1e9)(0); } });
  return {
    play() {
      this.stop();
      if (!layers || layers.w !== el.clientWidth) build();
      result.hidden = true; legend.classList.remove("on");
      raf = requestAnimationFrame((now) => frame(now)(now));
      timer = setTimeout(() => { result.hidden = false; legend.classList.add("on"); }, SCAN_MS + 300);
    },
    stop() { cancelAnimationFrame(raf); clearTimeout(timer); },
  };
}
