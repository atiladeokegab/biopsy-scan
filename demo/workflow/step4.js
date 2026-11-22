// Workflow step 4, "Down to 1-2 routine stains": the pick core unrolls into its marker channels as stacked tilted
// layers (each cell glows by its measured intensity for that channel), then the two chosen markers are re-rendered
// as simulated brown DAB on a haematoxylin counterstain. The shell draws the labels and the credit.
import { MARKER_COLORS, DAB, HAEMATOXYLIN } from "./palette.js";

// shortcut: no translate/ Step 1 selection is merged on integration, so the pair is an illustration; take it from
// that result's selected panel once it lands.
export const PICKED = { markers: ["PanCK", "CD8"], caption: "illustration" };
const SETTLE = 5.4; // seconds

const sprites = new Map();
function glow(hex) {
  if (!sprites.has(hex)) {
    const c = document.createElement("canvas"); c.width = c.height = 32;
    const x = c.getContext("2d"), g = x.createRadialGradient(16, 16, 0, 16, 16, 16);
    g.addColorStop(0, "#ffffff"); g.addColorStop(0.2, hex); g.addColorStop(0.55, hex + "55"); g.addColorStop(1, hex + "00");
    x.fillStyle = g; x.fillRect(0, 0, 32, 32); sprites.set(hex, c);
  }
  return sprites.get(hex);
}
const ease = (t, a, b) => { const x = Math.max(0, Math.min(1, (t - a) / (b - a))); return x < 0.5 ? 4 * x ** 3 : 1 - (-2 * x + 2) ** 3 / 2; };

export function mount(el, data) {
  const cv = document.createElement("canvas"), g = cv.getContext("2d");
  cv.style.cssText = "display:block;width:100%;height:100%";
  el.appendChild(cv);
  const { markers } = data.tma, core = data.tma.cores[data.tma.pick], [w, h] = core.size_um, half = Math.max(w, h) / 2;
  const cells = core.cells.map(([x, y, , ...m]) => ({ u: (x - w / 2) / half, v: (y - h / 2) / half, m }))
    .filter((c) => Math.hypot(c.u, c.v) <= 1); // a round core, as cut for a TMA
  const pick = PICKED.markers.map((n) => markers.indexOf(n));
  const css = (n, f) => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || f;
  const body = css("--text-body", "#B9C6D4");
  let raf = 0, t0 = 0, last = SETTLE;

  function draw(t) {
    const W = el.clientWidth || 720, H = el.clientHeight || 450, d = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(W * d) || cv.height !== Math.round(H * d)) { cv.width = Math.round(W * d); cv.height = Math.round(H * d); }
    g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, W, H);
    const font = getComputedStyle(document.body).fontFamily || "sans-serif", n = markers.length;
    const unroll = ease(t, 0.6, 2.2), out = ease(t, 3.0, 3.8), ihc = ease(t, 3.4, 4.6);
    // Layers: discs squashed in y and spread vertically; at unroll = 0 they sit on one another as the flat core.
    if (out < 1) {
      const R = Math.min(W * 0.26, H * 0.4), ox = W * 0.42, oy = H / 2, gap = (H * 0.72) / (n - 1), dot = Math.max(1.8, R / 80);
      g.save();
      markers.forEach((name, i) => {
        const tilt = 1 - unroll * 0.72, dy = (i - (n - 1) / 2) * gap * unroll, spr = glow(MARKER_COLORS[i]);
        g.globalCompositeOperation = "lighter";
        for (const c of cells) {
          const q = c.m[i] / 255; if (q < 0.15) continue;
          g.globalAlpha = (1 - out) * ease(t, 0, 0.6) * q * q * (unroll > 0.02 ? 1 : 0.5);
          g.drawImage(spr, ox + c.u * R - dot * 1.5, oy + dy + c.v * R * tilt - dot * 1.5, dot * 3, dot * 3);
        }
        g.globalCompositeOperation = "source-over";
        const a = (1 - out) * ease(t, 1.6, 2.2); if (a <= 0) return; // the channel's name beside its layer
        g.globalAlpha = a; g.fillStyle = MARKER_COLORS[i]; g.beginPath(); g.arc(ox + R + 14, oy + dy, 3.5, 0, 6.29); g.fill();
        g.font = `500 12px ${font}`; g.fillStyle = body; g.fillText(name, ox + R + 24, oy + dy + 4);
      });
      g.restore();
    }
    // Simulated IHC: haematoxylin on every nucleus, brown DAB scaled by the cell's measured intensity for the marker.
    if (ihc > 0) {
      const r2 = Math.min(W * 0.2, H * 0.36), s = r2 / 80;
      pick.forEach((mi, k) => {
        const x2 = W / 2 + (k - 0.5) * r2 * 2.5, y2 = H * 0.53;
        g.save(); g.globalAlpha = ihc; g.translate(0, (1 - ihc) * 14);
        g.beginPath(); g.arc(x2, y2, r2, 0, 6.29); g.fillStyle = "#EEE8F1"; g.fill(); g.clip();
        for (const c of cells) {
          const x = x2 + c.u * r2, y = y2 + c.v * r2, q = Math.max(0, (c.m[mi] - 90) / 165);
          g.globalAlpha = ihc * 0.55; g.fillStyle = HAEMATOXYLIN; g.beginPath(); g.arc(x, y, 0.9 * s + 0.4, 0, 6.29); g.fill();
          if (q > 0) { g.globalAlpha = ihc * Math.min(1, 0.25 + q); g.fillStyle = DAB; g.beginPath(); g.arc(x + 0.3, y, (1.0 + q) * s + 0.4, 0, 6.29); g.fill(); }
        }
        g.restore();
        g.save(); g.globalAlpha = ihc; g.font = `600 13px ${font}`; g.fillStyle = body; g.textAlign = "center";
        g.fillText(PICKED.markers[k], x2, y2 - r2 - 10); g.restore();
      });
    }
  }
  const frame = (now) => { last = Math.min((now - t0) / 1000, SETTLE); draw(last); if (last < SETTLE) raf = requestAnimationFrame(frame); };
  new ResizeObserver(() => { if (!raf || last >= SETTLE) draw(last); }).observe(el);
  draw(0);
  return {
    play() { cancelAnimationFrame(raf); t0 = performance.now(); raf = requestAnimationFrame(frame); },
    stop() { cancelAnimationFrame(raf); raf = 0; },
    renderAt(t) { last = t; draw(t); }, // stills for screenshots
  };
}
