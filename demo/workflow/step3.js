// Workflow step 3, "Stress-test before the trial": the hero core (real Charville cells) with the 3D view's one
// simulated look drawn over it (two 400 µm fields, 10% of the cells in view mis-called: they flicker, then keep the
// wrong colour, ringed), then textbook power vs simulated power at 200 patients, both read from preflight.json.
import { drawStudy, TYPES, FOV_UM, EPS, WINDOWS } from "../reimage3d.js";
import { loadData, makeCanvas, color, glow, background, chip, credit, font, span, player, clamp01 } from "./common.js";

export const END = 10; // seconds
export const AT = 200; // patients: the comparison the brief and the results section quote

// The two numbers, straight from the merged file.
export function powerAt(preflight, n = AT) {
  const sim = preflight.designs.find((d) => d.id === "cropped");
  const pt = sim.points.find((p) => p.patients === n);
  return { analytic: preflight.formula.power[String(n)], analyticLabel: preflight.formula.label,
    sim: pt.power, lo: pt.lo, hi: pt.hi, simLabel: sim.label, window: sim.window, miscall: sim.miscall };
}

export function mount(el, data, { onDone } = {}) {
  const { g, size } = makeCanvas(el);
  let ready = loadData(data).then((d) => setup(d)), render = () => {};
  const hue = () => TYPES.map((t) => color(`--${t}`));

  function setup(d) {
    const s = drawStudy(d.tissue), P = powerAt(d.preflight);
    const cx = (s.lo[0] + s.hi[0]) / 2, cy = (s.lo[1] + s.hi[1]) / 2, half = Math.max(s.hi[0] - s.lo[0], s.hi[1] - s.lo[1]) / 2;
    const keep = s.cells.map((c) => Math.hypot(c.x - cx, c.y - cy) <= half); // a round core, as cut for a TMA
    const phase = s.cells.map((_, j) => (j * 0.618) % 1);
    render = (t) => {
      const { W, H, narrow } = size(), cols = hue();
      background(g, W, H);
      const R = narrow ? W * 0.4 : Math.min(H * 0.4, W * 0.3), ox = narrow ? W / 2 : W * 0.34, oy = narrow ? 40 + R : H * 0.5;
      const k = R / half, X = (x) => ox + (x - cx) * k, Y = (y) => oy + (y - cy) * k;
      const appear = span(t, 0, 1.2), fov = span(t, 1.3, 2.4), flick = t > 2.6 && t < 4.8, settled = t >= 4.8;
      const dot = Math.max(2.2, R / 75);
      g.save(); g.globalCompositeOperation = "lighter";
      s.cells.forEach((c, j) => {
        if (!keep[j]) return;
        const wrong = s.inside[j] && s.obs[j] !== s.type[j];
        let ty = s.type[j];
        if (wrong && (settled || (flick && Math.sin((t * 9 + phase[j] * 6.28) * 2) > 0))) ty = s.obs[j];
        g.globalAlpha = appear * (s.inside[j] ? 1 : 1 - fov * 0.82);
        g.drawImage(glow(cols[ty], 6), X(c.x) - dot * 1.6, Y(c.y) - dot * 1.6, dot * 3.2, dot * 3.2);
      });
      g.restore();
      // Rings round the cells whose call crossed colour groups, as in the 3D view.
      if (t > 2.6) {
        g.save(); g.strokeStyle = color("--text-head"); g.lineWidth = 1; g.globalAlpha = span(t, 2.6, 3.4) * 0.9;
        s.cells.forEach((c, j) => { if (keep[j] && s.inside[j] && s.obs[j] !== s.type[j]) { g.beginPath(); g.arc(X(c.x), Y(c.y), dot * 1.5, 0, 6.29); g.stroke(); } });
        g.restore();
      }
      // The two fields of view: they drop in from slightly larger and settle.
      s.crops.forEach((cr, i) => {
        if (!cr) return;
        const f = clamp01(span(t, 1.3 + i * 0.3, 2.2 + i * 0.3)), grow = 1 + (1 - f) * 0.25;
        const w = cr[2] * k * grow, x = X(cr[0] + cr[2] / 2) - w / 2, y = Y(cr[1] + cr[2] / 2) - w / 2;
        g.save(); g.globalAlpha = f; g.strokeStyle = color("--accent"); g.lineWidth = 1.5;
        g.shadowColor = color("--accent"); g.shadowBlur = 10; g.strokeRect(x, y, w, w);
        g.shadowBlur = 0; font(g, 12, 600); g.fillStyle = color("--accent"); g.fillText(`${FOV_UM} µm`, x + 6, y + 16);
        g.restore();
      });
      chip(g, 12, 12, "MEASURED cells", color("--text-body"), appear);
      chip(g, 12, 40, `SIMULATED: ${WINDOWS} × ${FOV_UM} µm fields, ${Math.round(EPS * 100)}% mis-called`, color("--accent"), fov);
      // The result: two bars, the formula's answer and the simulation's.
      const r = span(t, 5.2, 6.4);
      if (r > 0) {
        const px = narrow ? 16 : W * 0.66, py = narrow ? oy + R + 28 : H * 0.3, pw = narrow ? W - 32 : W * 0.3;
        g.save(); g.globalAlpha = r; g.translate(0, (1 - r) * 12);
        font(g, 14, 600); g.fillStyle = color("--text-head"); g.fillText(`Power at ${AT} patients`, px, py);
        const rows = [[P.analyticLabel, P.analytic, color("--text-body"), "assumes the whole core, measured exactly"],
          [P.simLabel, P.sim, color("--tumour"), `${P.window.replace("um", " µm")} fields, ${Math.round(P.miscall * 100)}% mis-called`]];
        rows.forEach(([label, v, c, note], i) => {
          const y = py + 22 + i * (narrow ? 70 : 116), grow = span(t, 6.0 + i * 0.9, 7.2 + i * 0.9);
          chip(g, px, y, label, c);
          font(g, narrow ? 28 : 34, 700); g.fillStyle = c; g.fillText(`${Math.round(v * 100 * grow)}%`, px, y + (narrow ? 56 : 64));
          const bx = px + (narrow ? 90 : 0), by = y + (narrow ? 40 : 76), bw = pw - (narrow ? 90 : 0);
          g.fillStyle = "#22303F"; g.fillRect(bx, by, bw, 6);
          g.fillStyle = c; g.fillRect(bx, by, bw * v * grow, 6);
          font(g, 12, 400); g.fillStyle = color("--text-muted"); g.fillText(note, bx, by + 20, bw);
        });
        g.restore();
      }
      credit(g, W, H);
    };
  }

  const p = player((t) => render(t), END, onDone);
  return {
    play: () => ready.then(p.play),
    stop: p.stop,
    renderAt: (t) => ready.then(() => render(t)), // stills for screenshots
  };
}
