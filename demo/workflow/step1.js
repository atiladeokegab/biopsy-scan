// Step 1, tissue context: a tilted grid of real Charville cores glowing in pseudo-fluorescence from each cell's
// measured marker intensities; the grid flattens, one core is picked, and the view zooms into it.
import { TYPE_COLORS, MARKER_COLORS } from "./palette.js";

const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const MC = MARKER_COLORS.map(hex);
const ZOOM_R = 0.38; // the zoomed core's radius, as a share of the stage's short side

// Draw one core into ctx as a disc of radius r at (cx, cy). mode "fluor": each cell glows in the sum of its
// marker channels' colours; mode "seg": each cell is a filled disc in its type's colour.
export function drawCore(ctx, core, cx, cy, r, mode) {
  const [sw, sh = sw] = [].concat(core.size_um), side = Math.max(sw, sh), s = (2 * r) / side; // [w, h], or one side
  const ox = ((side - sw) / 2) * s, oy = ((side - sh) / 2) * s; // centre the shorter axis
  ctx.save();
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.clip();
  ctx.fillStyle = mode === "seg" ? "#05070b" : "#020306";
  ctx.fillRect(cx - r, cy - r, 2 * r, 2 * r);
  // a cell's footprint: the core's area shared among the cells it really holds, shown at the subsampled count
  const cell = Math.max(0.8, (2 * r) / Math.sqrt(core.cells.length) * (mode === "seg" ? 0.5 : 0.42));
  ctx.globalCompositeOperation = mode === "seg" ? "source-over" : "lighter";
  // additive cells overlap when a small core holds many cells (the 0.8 px floor): dim each by its overlap so the sum never whites out
  const alpha = 0.85 / Math.max(1, (core.cells.length * cell * cell) / (r * r));
  for (const c of core.cells) {
    const x = cx - r + ox + c[0] * s, y = cy - r + oy + c[1] * s;
    if (mode === "seg") ctx.fillStyle = TYPE_COLORS[c[2]];
    else {
      // squared intensities keep each cell's strongest channels and let weak ones fall dark, as in a real IF image
      let R = 0, G = 0, B = 0;
      for (let k = 0; k < MC.length && 3 + k < c.length; k++) {
        const w = (c[3 + k] / 255) ** 2 * 0.5;
        R += MC[k][0] * w; G += MC[k][1] * w; B += MC[k][2] * w;
      }
      ctx.fillStyle = `rgba(${Math.min(255, R | 0)},${Math.min(255, G | 0)},${Math.min(255, B | 0)},${alpha})`;
    }
    ctx.beginPath(); ctx.arc(x, y, cell, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.restore();
}

// The glow: one blurred copy of the finished bitmap added under-and-over it (a filter per cell would stall software GL).
export function glow(cv, ctx, w, h) {
  const copy = document.createElement("canvas");
  copy.width = cv.width; copy.height = cv.height;
  copy.getContext("2d").drawImage(cv, 0, 0);
  ctx.save(); ctx.globalCompositeOperation = "lighter"; ctx.filter = "blur(4px)"; ctx.globalAlpha = 0.45;
  ctx.drawImage(copy, 0, 0, w, h); ctx.restore();
}

export function sizeCanvas(el, oversample = 1) {
  const cv = el.querySelector("canvas") ?? el.insertBefore(document.createElement("canvas"), el.firstChild);
  const w = el.clientWidth, h = el.clientHeight, d = Math.min(4, (devicePixelRatio || 1) * oversample); // capped: a 4x bitmap of the grid is ~40 MB
  cv.width = Math.round(w * d); cv.height = Math.round(h * d);
  const ctx = cv.getContext("2d");
  ctx.setTransform(d, 0, 0, d, 0, 0);
  return { cv, ctx, w, h };
}

export function mount(el, data) {
  const { tma } = data, n = Math.ceil(Math.sqrt(tma.cores.length));
  let timers = [], geo = null;
  const draw = () => {
    const { cv, ctx, w, h } = sizeCanvas(el, 2.5); // oversampled so the zoomed core stays sharp
    const side = Math.min(w, h) * 0.92, cell = side / n, r = cell * 0.44;
    const x0 = (w - side) / 2, y0 = (h - side) / 2;
    tma.cores.forEach((c, i) => drawCore(ctx, c, x0 + cell * (i % n + 0.5), y0 + cell * (Math.floor(i / n) + 0.5), r, "fluor"));
    glow(cv, ctx, w, h);
    const p = tma.pick;
    geo = { cv, px: x0 + cell * (p % n + 0.5), py: y0 + cell * (Math.floor(p / n) + 0.5), r, w, h };
  };
  // The zoom's end state, drawn sharp at its final size: a scaled bitmap of the grid goes soft (and whites out on phones).
  el.insertBefore(document.createElement("canvas"), el.firstChild); // the grid's canvas: first, so sizeCanvas finds it
  const sharp = el.appendChild(document.createElement("canvas"));
  const drawSharp = () => {
    const { w, h } = geo, d = Math.min(3, devicePixelRatio || 1);
    sharp.width = Math.round(w * d); sharp.height = Math.round(h * d);
    const ctx = sharp.getContext("2d");
    ctx.setTransform(d, 0, 0, d, 0, 0);
    drawCore(ctx, tma.cores[tma.pick], w / 2, h / 2, Math.min(w, h) * ZOOM_R, "fluor");
    glow(sharp, ctx, w, h);
  };
  const ring = el.appendChild(document.createElement("div"));
  ring.className = "wf-ring";
  addEventListener("resize", () => { if (!el.hidden) { draw(); drawSharp(); } });
  return {
    play() {
      this.stop();
      if (!geo || geo.w !== el.clientWidth) { draw(); drawSharp(); }
      const { cv, px, py, r, w, h } = geo;
      sharp.style.transition = "none"; sharp.style.opacity = 0; cv.style.opacity = 1;
      cv.style.transition = "none";
      cv.style.transformOrigin = "50% 50%";
      cv.style.transform = "perspective(900px) rotateX(50deg) rotateZ(-28deg) scale(0.78)";
      ring.style.cssText = `left:${px - r - 4}px;top:${py - r - 4}px;width:${2 * r + 8}px;height:${2 * r + 8}px;opacity:0`;
      void cv.offsetWidth;
      timers.push(setTimeout(() => { cv.style.transition = "transform 1.6s cubic-bezier(.2,.7,.2,1)"; cv.style.transform = "none"; }, 600));
      timers.push(setTimeout(() => { ring.style.opacity = 1; }, 2300));
      timers.push(setTimeout(() => {
        ring.style.opacity = 0;
        cv.style.transformOrigin = `${px}px ${py}px`; // the grid is flat now, so moving the origin does not jump
        const z = Math.min(w, h) * ZOOM_R / r;
        cv.style.transition = "transform 1.8s cubic-bezier(.5,0,.2,1)";
        cv.style.transform = `translate(${w / 2 - px}px, ${h / 2 - py}px) scale(${z})`;
      }, 3300));
      timers.push(setTimeout(() => { // the zoom has landed: swap the soft bitmap for the sharp core
        sharp.style.transition = cv.style.transition = "opacity 0.5s";
        sharp.style.opacity = 1; cv.style.opacity = 0;
      }, 5100));
    },
    stop() { timers.forEach(clearTimeout); timers = []; },
  };
}
