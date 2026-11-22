// Workflow step 4, "Down to 1-2 routine stains": the core unrolls into tilted marker-channel layers (each a glow of
// the cells positive for that channel), a marker-combination search runs over every 1- and 2-channel panel, then the
// chosen pair is re-rendered as simulated brown DAB on a haematoxylin counterstain.
// shortcut: tissue.json carries display classes, not marker intensities, so a channel here is "cells of one class"
// and the chosen pair is an illustration; swap in the translate/ Step 1 selection once its results are merged.
import { pickPatient } from "../reimage3d.js";
import { loadData, makeCanvas, color, glow, background, chip, credit, font, span, player } from "./common.js";

export const END = 13; // seconds
// Channels, top layer first: name, the cells positive for it, glow colour.
const CHANNELS = [
  ["tumour-cell marker", (c) => c.type === "tumour", "#FF4F7B"],
  ["immune-cell marker", (c) => c.type === "immune", "#F5C451"],
  ["stromal marker", (c) => c.type === "stroma", "#2BD4A4"],
  ["nuclear stain", () => true, "#4C7DFF"],
];
const PICK = [0, 1]; // the illustrated pair
const panels = () => { const p = CHANNELS.map((_, i) => [i]); CHANNELS.forEach((_, i) => CHANNELS.forEach((_, j) => j > i && p.push([i, j]))); return p; };

export function mount(el, data, { onDone } = {}) {
  const { g, size } = makeCanvas(el);
  let render = () => {};
  const ready = loadData(data).then((d) => {
    const cells = pickPatient(d.tissue).cells;
    let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const c of cells) { lo = [Math.min(lo[0], c.x), Math.min(lo[1], c.y)]; hi = [Math.max(hi[0], c.x), Math.max(hi[1], c.y)]; }
    const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2, half = Math.max(hi[0] - lo[0], hi[1] - lo[1]) / 2;
    const core = cells.filter((c) => Math.hypot(c.x - cx, c.y - cy) <= half).map((c) => ({ ...c, u: (c.x - cx) / half, v: (c.y - cy) / half }));
    const P = panels(), names = d.markers ?? null; // d.markers: optional [{name}] from translate/ results, when merged
    const label = (i) => names?.[i] ?? CHANNELS[i][0];
    render = (t) => draw(t, core, P, label);
  });

  function draw(t, core, P, label) {
    const { W, H, narrow } = size();
    background(g, W, H);
    const R = narrow ? W * 0.26 : Math.min(W * 0.2, H * 0.3), ox = narrow ? W * 0.33 : W * 0.28, oy = narrow ? 70 + R * 1.1 : H * 0.46;
    const unroll = span(t, 0.8, 3.0), search = span(t, 3.2, 3.8), out = span(t, 7.4, 8.4), ihc = span(t, 8.2, 9.4);
    const n = CHANNELS.length, gap = R * (narrow ? 0.42 : 0.5), dot = Math.max(1.8, R / 70);
    // Layers: each a tilted disc (y squashed) stacked vertically; at unroll = 0 they sit on one another as the core.
    if (out < 1) {
      g.save(); g.globalCompositeOperation = "lighter";
      CHANNELS.forEach(([, pos, hex], i) => {
        const chosen = PICK.includes(i), fade = chosen ? 1 : 1 - span(t, 6.6, 7.4);
        const tilt = 1 - unroll * 0.66, dy = (i - (n - 1) / 2) * gap * unroll;
        g.globalAlpha = (1 - out) * fade * span(t, 0, 0.8) * (unroll > 0 ? 1 : 0.6);
        const spr = glow(hex, 6);
        for (const c of core) if (pos(c)) g.drawImage(spr, ox + c.u * R - dot * 1.5, oy + dy + c.v * R * tilt - dot * 1.5, dot * 3, dot * 3);
        g.globalCompositeOperation = "source-over";
        if (unroll > 0.6) { // channel name beside its layer
          g.globalAlpha = (unroll - 0.6) / 0.4 * (1 - out) * fade; font(g, 12, 500); g.fillStyle = hex;
          g.beginPath(); g.arc(ox + R + 14, oy + dy, 3.5, 0, 6.29); g.fill();
          g.fillStyle = color("--text-body"); g.fillText(label(i), ox + R + 24, oy + dy + 4);
        }
        g.globalCompositeOperation = "lighter";
      });
      g.restore();
    }
    // The marker-combination search: every 1- and 2-channel panel lights in turn; the chosen pair stays lit.
    if (search > 0 && out < 1) {
      const px = narrow ? 16 : W * 0.7, py = narrow ? oy + R * 1.9 + 8 : H * 0.2, pw = narrow ? W - 32 : W * 0.27;
      g.save(); g.globalAlpha = search * (1 - out);
      const cols = narrow ? 5 : 2, cw = (pw - 24) / cols, rh = narrow ? 24 : 30, ph = 50 + Math.ceil(P.length / cols) * rh + 36;
      g.fillStyle = "#121A25e6"; g.strokeStyle = color("--border"); g.beginPath(); g.roundRect(px, py, pw, ph, 10); g.fill(); g.stroke();
      font(g, 12, 600); g.fillStyle = color("--text-head"); g.fillText("Marker combination search", px + 12, py + 20);
      font(g, 12, 400); g.fillStyle = color("--text-muted"); g.fillText(`${P.length} panels of 1 or 2 channels`, px + 12, py + 38);
      const step = Math.floor(Math.max(0, t - 3.8) / 0.28), done = t > 3.8 + P.length * 0.28;
      P.forEach((p, k) => {
        const x = px + 12 + (k % cols) * cw, y = py + 50 + Math.floor(k / cols) * rh;
        const win = p.length === 2 && p[0] === PICK[0] && p[1] === PICK[1];
        const lit = done ? win : k === step % P.length;
        p.forEach((i, m) => { g.fillStyle = CHANNELS[i][2]; g.globalAlpha = search * (1 - out) * (lit ? 1 : 0.28);
          g.shadowColor = CHANNELS[i][2]; g.shadowBlur = lit ? 10 : 0;
          g.beginPath(); g.arc(x + 8 + m * 14, y + 8, 5, 0, 6.29); g.fill(); });
        g.shadowBlur = 0;
        if (done && win) { g.globalAlpha = search * (1 - out); g.strokeStyle = color("--accent"); g.strokeRect(x, y - 2, cw - 6, 20); }
      });
      g.globalAlpha = search * (1 - out);
      chip(g, px + 12, py + ph - 30, "ILLUSTRATION", color("--text-body"));
      g.restore();
    }
    // Simulated IHC: two round cores, haematoxylin counterstain on every cell, brown DAB on the chosen channel's.
    if (ihc > 0) {
      const r2 = narrow ? W * 0.22 : Math.min(W * 0.17, H * 0.28), y2 = narrow ? 100 + r2 : H * 0.48;
      PICK.forEach((ch, m) => {
        const x2 = narrow ? W / 2 : W / 2 + (m - 0.5) * r2 * 2.5, yy = narrow ? y2 + m * (r2 * 2 + 56) : y2;
        g.save(); g.globalAlpha = ihc; g.translate(0, (1 - ihc) * 16);
        g.beginPath(); g.arc(x2, yy, r2, 0, 6.29); g.fillStyle = "#EFE9F2"; g.fill(); g.clip();
        const s = r2 / 70;
        for (const c of core) {
          const x = x2 + c.u * r2, y = yy + c.v * r2, pos = CHANNELS[ch][1](c);
          if (pos) { g.fillStyle = "#A8733F40"; g.beginPath(); g.arc(x, y, 2.4 * s + 0.6, 0, 6.29); g.fill(); } // DAB halo
          g.fillStyle = "#5F5BA680"; g.beginPath(); g.arc(x, y, 0.9 * s + 0.4, 0, 6.29); g.fill(); // haematoxylin nucleus
          if (pos) { g.fillStyle = "#6E4219c0"; g.beginPath(); g.arc(x + 0.3, y, 1.3 * s + 0.4, 0, 6.29); g.fill(); } // DAB
        }
        g.restore();
        g.save(); g.globalAlpha = ihc; font(g, 14, 600); g.fillStyle = color("--text-head"); g.textAlign = "center";
        g.fillText(label(ch), x2, yy - r2 - 12); g.restore();
      });
      chip(g, 12, 40, "SIMULATED IHC", color("--accent"), ihc);
      chip(g, 12 + 128, 40, "markers: illustration", color("--text-body"), ihc);
    }
    chip(g, 12, 12, "MEASURED cells", color("--text-body"), span(t, 0, 0.8));
    credit(g, W, H);
  }

  const p = player((t) => render(t), END, onDone);
  return { play: () => ready.then(p.play), stop: p.stop, renderAt: (t) => ready.then(() => render(t)) };
}
