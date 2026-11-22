// Workflow step 3, "Stress-test before the trial": the pick core (real Charville cells, coloured by type), two
// 400 µm fields of view land on it, 10% of the cells they see flicker to another type's colour (Preflight's mis-call:
// keep the type with prob 1 - eps, else a uniform other type), then the formula's power against the simulated power
// at 200 patients, both read from preflight.json. The shell draws the labels and the credit.
import { TYPE_COLORS } from "./palette.js";
import { drawCore } from "./step1.js";

export const FOV_UM = 400, EPS = 0.1, WINDOWS = 2, AT = 200, SEED = 16;
const SETTLE = 5; // seconds

export function rng(seed) { // same generator as demo/tissue.js
  let s = seed >>> 0;
  return () => ((s = (s + 0x6d2b79f5) >>> 0), (((s ^ (s >>> 15)) * (s | 1)) >>> 0) / 4294967296);
}

// The two numbers, from the file: the textbook formula's and the simulated 2 x 400 µm, 10% mis-called design's.
export function powerAt(preflight, n = AT) {
  const sim = preflight.designs.find((d) => d.id === "cropped"), pt = sim.points.find((p) => p.patients === n);
  return { analytic: preflight.formula.power[String(n)], analyticLabel: preflight.formula.label, sim: pt.power, simLabel: sim.label };
}

// One simulated look: WINDOWS square fields placed uniformly in the core's box, then a mis-call draw per cell.
export function look(core, K = TYPE_COLORS.length) {
  const r = rng(SEED), [w, h = w] = [core.size_um].flat();
  const crops = Array.from({ length: WINDOWS }, () => [r() * Math.max(w - FOV_UM, 0), r() * Math.max(h - FOV_UM, 0)]);
  const inside = core.cells.map(([x, y]) => crops.some(([cx, cy]) => x >= cx && x < cx + FOV_UM && y >= cy && y < cy + FOV_UM));
  const called = core.cells.map(([, , t], j) => {
    const u = r();
    if (!inside[j] || u >= EPS) return t;
    return (t + 1 + Math.floor((u / EPS) * (K - 1))) % K; // uniform over the K - 1 other types
  });
  return { crops, inside, called };
}

const ease = (t, a, b) => { const x = Math.max(0, Math.min(1, (t - a) / (b - a))); return x < 0.5 ? 4 * x ** 3 : 1 - (-2 * x + 2) ** 3 / 2; };

export function mount(el, data) {
  const cv = document.createElement("canvas"), g = cv.getContext("2d"), mask = document.createElement("canvas");
  cv.style.cssText = "display:block;width:100%;height:100%";
  el.appendChild(cv);
  const core = data.tma.cores[data.tma.pick], P = powerAt(data.preflight), L = look(core);
  const [cw, ch = cw] = [core.size_um].flat(), S = Math.max(cw, ch); // drawCore maps [0, S] onto the disc's diameter
  const dx = (S - cw) / 2, dy = (S - ch) / 2; // centre the region in the disc
  const css = (n, f) => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || f;
  const head = css("--text-head", "#F4F7FA"), muted = css("--text-muted", "#8898AA"), accent = css("--accent", "#2BD4A4"), warn = css("--tumour", "#FF7A59");
  let raf = 0, t0 = 0, last = SETTLE;

  function draw(t) {
    const W = el.clientWidth || 720, H = el.clientHeight || 450, d = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(W * d) || cv.height !== Math.round(H * d)) { cv.width = Math.round(W * d); cv.height = Math.round(H * d); }
    g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, W, H);
    const R = Math.min(H * 0.44, W * 0.3), ox = W * 0.32, oy = H / 2, k = (2 * R) / S;
    const X = (x) => ox - R + (x + dx) * k, Y = (y) => oy - R + (y + dy) * k;
    const appear = ease(t, 0, 0.8), fov = ease(t, 0.8, 1.6), settled = t >= 3.4;
    // The core as steps 1-2 draw it, with each mis-called cell flickering to (then keeping) its called type.
    const cells = core.cells.map((c, j) => (L.called[j] !== c[2] && (settled || (t > 1.8 && Math.sin(t * 18 + j) > 0)) ? [c[0] + dx, c[1] + dy, L.called[j]] : [c[0] + dx, c[1] + dy, c[2]]));
    g.globalAlpha = appear; drawCore(g, { ...core, size_um: S, cells }, ox, oy, R, "seg"); g.globalAlpha = 1;
    // Dim everything outside the two fields of view.
    const m = mask.getContext("2d"); mask.width = cv.width; mask.height = cv.height; m.setTransform(d, 0, 0, d, 0, 0);
    m.fillStyle = `rgba(2,3,6,${0.72 * fov})`; m.beginPath(); m.arc(ox, oy, R + 1, 0, 6.29); m.fill();
    m.globalCompositeOperation = "destination-out"; // cut the fields out of the shade (their overlap included)
    L.crops.forEach(([cx, cy]) => m.fillRect(X(cx), Y(cy), FOV_UM * k, FOV_UM * k));
    g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.drawImage(mask, 0, 0); g.restore();
    g.save(); g.strokeStyle = head; g.lineWidth = R < 120 ? 0.6 : 1; g.globalAlpha = ease(t, 1.8, 2.4) * 0.8; // ring the mis-called cells
    const ring = Math.max(2, R / 60);
    core.cells.forEach(([x, y, ty], j) => { if (L.called[j] !== ty) { g.beginPath(); g.arc(X(x), Y(y), ring, 0, 6.29); g.stroke(); } });
    g.restore();
    L.crops.forEach(([cx, cy], i) => { // the fields drop in from slightly larger and settle
      const f = ease(t, 0.8 + i * 0.25, 1.6 + i * 0.25), s = FOV_UM * k * (1 + (1 - f) * 0.25);
      g.save(); g.globalAlpha = f; g.strokeStyle = accent; g.lineWidth = 1.5; g.shadowColor = accent; g.shadowBlur = 10;
      g.strokeRect(X(cx + FOV_UM / 2) - s / 2, Y(cy + FOV_UM / 2) - s / 2, s, s); g.restore();
    });
    // Formula vs simulation at 200 patients: two bars that grow to the file's values.
    const px = W * 0.65, pw = W * 0.32, big = Math.max(22, Math.min(40, W * 0.05)), font = getComputedStyle(document.body).fontFamily || "sans-serif";
    [[P.analytic, P.analyticLabel, head], [P.sim, P.simLabel, warn]].forEach(([v, label, c], i) => {
      const a = ease(t, 2.4 + i * 0.6, 3.6 + i * 0.6); if (a <= 0) return;
      const y = H * 0.3 + i * H * 0.32;
      g.save(); g.globalAlpha = Math.min(1, a * 2);
      g.font = `600 12px ${font}`; g.fillStyle = muted; g.fillText(W < 560 ? label : `${label}, ${AT} patients`, px, y, pw);
      g.font = `700 ${big}px ${font}`; g.fillStyle = c; g.fillText(`${Math.round(v * 100 * a)}%`, px, y + big + 4);
      g.fillStyle = "#22303F"; g.fillRect(px, y + big + 14, pw, 5); g.fillStyle = c; g.fillRect(px, y + big + 14, pw * v * a, 5);
      g.restore();
    });
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
