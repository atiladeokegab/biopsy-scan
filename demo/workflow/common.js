// Shared plumbing for the workflow step visuals: a DPR-aware canvas sized to its box, the page's colour tokens,
// cached glow sprites, the label chip and the data credit. Canvas 2D only, offline.
import { tok } from "../chart.js";

export const CREDIT = "Charville CODEX, Wu et al. 2022, Zenodo 13179600, CC-BY-4.0";
const FALLBACK = { "--bg-canvas": "#070B11", "--surface": "#121A25", "--border": "#22303F", "--text-head": "#F4F7FA",
  "--text-body": "#B9C6D4", "--text-muted": "#8898AA", "--accent": "#2BD4A4", "--tumour": "#FF7A59", "--immune": "#F5C451",
  "--stroma": "#56687A", "--font-sans": "Inter, system-ui, sans-serif" };
export const color = (name) => tok(name) || FALLBACK[name];

export const clamp01 = (x) => Math.max(0, Math.min(1, x));
export const ease = (x) => { x = clamp01(x); return x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2; };
export const span = (t, a, b) => ease((t - a) / (b - a)); // 0 before a, 1 after b, eased between

// Data: what the caller passes, else the page's own merged files (fetched relative to this module).
export async function loadData(data = {}) {
  const get = async (f) => (await fetch(new URL(`../data/${f}`, import.meta.url))).json();
  const [tissue, preflight] = await Promise.all([data.tissue ?? get("tissue.json"), data.preflight ?? get("preflight.json")]);
  await document.fonts?.load(`600 12px ${color("--font-sans")}`).catch(() => {}); // canvas text needs the face loaded first
  return { ...data, tissue, preflight };
}

// One canvas filling el's width; height from the layout's aspect. Narrow boxes (phones) get a portrait layout.
export function makeCanvas(el) {
  const cv = document.createElement("canvas");
  cv.style.cssText = "display:block;width:100%;height:auto;border-radius:var(--radius-surface,10px)";
  el.appendChild(cv);
  const g = cv.getContext("2d");
  const size = () => {
    const W = Math.max(300, Math.min(el.clientWidth || 720, 960)), narrow = W < 560;
    const H = Math.round(narrow ? W * 1.5 : (W * 520) / 720), d = window.devicePixelRatio || 1;
    cv.width = Math.round(W * d); cv.height = Math.round(H * d); cv.style.aspectRatio = `${W} / ${H}`;
    g.setTransform(d, 0, 0, d, 0, 0);
    return { W, H, narrow };
  };
  return { cv, g, size };
}

// A soft round glow of one colour, drawn once and stamped per cell with drawImage (fast for ~3k cells a frame).
const sprites = new Map();
export function glow(hex, r = 8) {
  const key = hex + r;
  if (!sprites.has(key)) {
    const c = document.createElement("canvas"); c.width = c.height = r * 4;
    const x = c.getContext("2d"), gr = x.createRadialGradient(r * 2, r * 2, 0, r * 2, r * 2, r * 2);
    gr.addColorStop(0, "#ffffff"); gr.addColorStop(0.18, hex); gr.addColorStop(0.5, hex + "55"); gr.addColorStop(1, hex + "00");
    x.fillStyle = gr; x.fillRect(0, 0, r * 4, r * 4);
    sprites.set(key, c);
  }
  return sprites.get(key);
}

export function background(g, W, H) {
  g.fillStyle = color("--bg-canvas"); g.fillRect(0, 0, W, H);
  const gr = g.createRadialGradient(W * 0.4, H * 0.45, 0, W * 0.4, H * 0.45, Math.max(W, H) * 0.7);
  gr.addColorStop(0, "#12233a"); gr.addColorStop(1, "#070B1100");
  g.fillStyle = gr; g.fillRect(0, 0, W, H);
}

export function font(g, px, weight = 500) { g.font = `${weight} ${px}px ${color("--font-sans")}`; }

// The label chip (MEASURED / ANALYTIC / ESTIMATE / SIMULATED ...): small caps text in a bordered pill.
export function chip(g, x, y, text, hue = color("--text-body"), alpha = 1) {
  if (alpha <= 0) return;
  g.save(); g.globalAlpha = alpha; font(g, 12, 600);
  const w = g.measureText(text).width + 16;
  g.fillStyle = "#0B1018cc"; g.strokeStyle = hue; g.lineWidth = 1;
  g.beginPath(); g.roundRect(x, y, w, 22, 11); g.fill(); g.stroke();
  g.fillStyle = hue; g.textBaseline = "middle"; g.fillText(text, x + 8, y + 11.5);
  g.restore();
  return w;
}

export function credit(g, W, H) {
  g.save(); font(g, 12, 400); g.fillStyle = color("--text-muted"); g.textBaseline = "alphabetic";
  g.fillText(CREDIT, 12, H - 10, W - 24); g.restore();
}

// Play a timeline: render(t) each frame until end, then onDone. Reduced motion shows the last frame at once.
export function player(render, end, onDone) {
  let raf = 0, t0 = 0, done = false, timer = 0;
  const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  const frame = (now) => {
    const t = (now - t0) / 1000;
    render(Math.min(t, end));
    if (t < end) raf = requestAnimationFrame(frame);
    else if (!done) { done = true; onDone?.(); }
  };
  return {
    play() { cancelAnimationFrame(raf); done = false; if (reduce) { render(end); raf = 0; timer = setTimeout(() => onDone?.(), 3000); return; } t0 = performance.now(); raf = requestAnimationFrame(frame); },
    stop() { cancelAnimationFrame(raf); clearTimeout(timer); raf = 0; },
  };
}
