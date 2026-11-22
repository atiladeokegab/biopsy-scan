// Workflow step 4, "Down to 1-2 routine stains": the pick core unrolls into its six MEASURED marker channels (each a
// plane where every cell glows by its real CODEX intensity), a marker-combination search sweeps every 1- and
// 2-marker panel (an animation: no search has run on real outcomes, so it names no winner), then two REAL routine IHC
// cores from the Human Protein Atlas (other patients): KRT5 (a cytokeratin, as PanCK detects) and ACTA2 (the gene
// for aSMA). If the images fail to load, the two channels are re-rendered as simulated DAB and say so.
import { MARKER_COLORS, DAB, HAEMATOXYLIN } from "./palette.js";

export const SEARCH_LABEL = "ILLUSTRATION · search not yet run on real outcomes";
export const HPA_CREDIT = "Human Protein Atlas (proteinatlas.org), head and neck cancer, CC BY-SA 3.0";
export const IHC = [ // real cores: file, gene, antibody (img/hpa_manifest.json), the CODEX channel it stands next to
  { file: "hpa_KRT5_02.webp", gene: "KRT5", antibody: "CAB000027", channel: "PanCK", note: "tumour cells" },
  { file: "hpa_ACTA2_02.webp", gene: "ACTA2", antibody: "CAB000002", channel: "aSMA", note: "stroma" },
];
const SETTLE = 5.4; // seconds

const sprites = new Map();
function glowDot(hex) {
  if (!sprites.has(hex)) {
    const c = document.createElement("canvas"); c.width = c.height = 32;
    const x = c.getContext("2d"), g = x.createRadialGradient(16, 16, 0, 16, 16, 16);
    g.addColorStop(0, "#ffffff"); g.addColorStop(0.2, hex); g.addColorStop(0.55, hex + "55"); g.addColorStop(1, hex + "00");
    x.fillStyle = g; x.fillRect(0, 0, 32, 32); sprites.set(hex, c);
  }
  return sprites.get(hex);
}
const ease = (t, a, b) => { const x = Math.max(0, Math.min(1, (t - a) / (b - a))); return x < 0.5 ? 4 * x ** 3 : 1 - (-2 * x + 2) ** 3 / 2; };
export const panels = (n) => { const p = []; for (let i = 0; i < n; i++) p.push([i]); for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) p.push([i, j]); return p; };

export function mount(el, data) {
  const cv = document.createElement("canvas"), g = cv.getContext("2d");
  cv.style.cssText = "display:block;width:100%;height:100%";
  el.appendChild(cv);
  const { markers } = data.tma, core = data.tma.cores[data.tma.pick];
  const [w, h = w] = [core.size_um].flat(), half = Math.max(w, h) / 2;
  const cells = core.cells.map(([x, y, , ...m]) => ({ u: (x - w / 2) / half, v: (y - h / 2) / half, m }))
    .filter((c) => Math.hypot(c.u, c.v) <= 1); // a round core, as cut for a TMA
  const P = panels(markers.length);
  const css = (n, f) => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || f;
  const head = css("--text-head", "#F4F7FA"), body = css("--text-body", "#B9C6D4"), muted = css("--text-muted", "#8898AA");
  let raf = 0, t0 = 0, last = SETTLE, real = null; // real: true once both HPA images decode, false if either fails
  const imgs = IHC.map((c) => Object.assign(new Image(), { src: new URL(`./img/${c.file}`, import.meta.url).href }));
  Promise.all(imgs.map((i) => i.decode())).then(() => { real = true; }, () => { real = false; }).then(() => { if (!raf) draw(last); });

  function draw(t) {
    const W = el.clientWidth || 720, H = el.clientHeight || 450, d = window.devicePixelRatio || 1, narrow = W < 560;
    if (cv.width !== Math.round(W * d) || cv.height !== Math.round(H * d)) { cv.width = Math.round(W * d); cv.height = Math.round(H * d); }
    g.setTransform(d, 0, 0, d, 0, 0); g.clearRect(0, 0, W, H);
    const font = getComputedStyle(document.body).fontFamily || "sans-serif", n = markers.length;
    const unroll = ease(t, 0.6, 2.0), out = ease(t, 3.4, 4.0), ihc = ease(t, 3.8, 5.0);
    // Channel planes: discs squashed in y and spread in depth; at unroll = 0 they add up to the flat core.
    if (out < 1) {
      const R = Math.min(W * (narrow ? 0.17 : 0.2), H * 0.4), ox = W * (narrow ? 0.22 : 0.3), oy = H / 2;
      const gap = (H * 0.74) / (n - 1), dot = Math.max(1.6, R / 70);
      g.save();
      markers.forEach((name, i) => {
        const tilt = 1 - unroll * 0.72, dy = (i - (n - 1) / 2) * gap * unroll, dx = (i - (n - 1) / 2) * -R * 0.08 * unroll;
        const spr = glowDot(MARKER_COLORS[i]);
        g.globalCompositeOperation = "lighter";
        for (const c of cells) {
          const q = c.m[i] / 255; if (q < 0.15) continue;
          g.globalAlpha = (1 - out) * ease(t, 0, 0.6) * q * q;
          g.drawImage(spr, ox + dx + c.u * R - dot * 1.5, oy + dy + c.v * R * tilt - dot * 1.5, dot * 3, dot * 3);
        }
        g.globalCompositeOperation = "source-over";
        const a = (1 - out) * ease(t, 1.4, 2.0); if (a <= 0) return; // the channel's real name beside its plane
        g.globalAlpha = a; g.fillStyle = MARKER_COLORS[i]; g.beginPath(); g.arc(ox + dx + R + 12, oy + dy, 3.5, 0, 6.29); g.fill();
        g.font = `500 12px ${font}`; g.fillStyle = body; g.fillText(name, ox + dx + R + 21, oy + dy + 4);
      });
      g.restore();
    }
    // The combination search: every 1- and 2-marker panel lights in turn. No winner: it has not run on outcomes.
    const sa = ease(t, 2.0, 2.4) * (1 - out);
    if (sa > 0) {
      const px = W * (narrow ? 0.6 : 0.62), pw = W * (narrow ? 0.38 : 0.34), cols = 7, cw = (pw - 20) / cols, rh = narrow ? 18 : 24;
      const ph = 58 + Math.ceil(P.length / cols) * rh + 8, py = (H - ph) / 2;
      g.save(); g.globalAlpha = sa;
      g.fillStyle = "#121A25e6"; g.strokeStyle = "#22303F"; g.beginPath(); g.roundRect(px, py, pw, ph, 10); g.fill(); g.stroke();
      g.font = `600 12px ${font}`; g.fillStyle = head; g.fillText("Marker combination search", px + 10, py + 20, pw - 20);
      g.font = `500 11px ${font}`; g.fillStyle = muted;
      g.fillText("ILLUSTRATION ·", px + 10, py + 36, pw - 20); g.fillText("not yet run on real outcomes", px + 10, py + 50, pw - 20);
      const sweep = Math.max(0, t - 2.2) / 0.06; // panels per second of sweep
      P.forEach((p, k) => {
        const x = px + 10 + (k % cols) * cw + cw / 2 - (p.length - 1) * 4, y = py + 58 + Math.floor(k / cols) * rh + rh / 2;
        const lit = Math.max(0, 1 - Math.abs(sweep % (P.length + 6) - k) / 3);
        p.forEach((i, m) => {
          g.globalAlpha = sa * (0.3 + 0.7 * lit); g.fillStyle = MARKER_COLORS[i]; g.shadowColor = MARKER_COLORS[i]; g.shadowBlur = 10 * lit;
          g.beginPath(); g.arc(x + m * 8, y, 3.2, 0, 6.29); g.fill();
        });
      });
      g.restore();
    }
    // Two routine-stain cores: real HPA IHC when the images load, else simulated DAB on haematoxylin (labelled).
    if (ihc > 0) {
      const r2 = Math.min(W * 0.2, H * 0.34), y2 = H * 0.47;
      IHC.forEach((c, k) => {
        const x2 = W / 2 + (k - 0.5) * r2 * 2.5;
        g.save(); g.globalAlpha = ihc; g.translate(0, (1 - ihc) * 14);
        g.beginPath(); g.arc(x2, y2, r2, 0, 6.29); g.fillStyle = "#EEE8F1"; g.fill(); g.clip();
        if (real) g.drawImage(imgs[k], x2 - r2 * 1.17, y2 - r2 * 1.17, r2 * 2.34, r2 * 2.34); // the core fills ~85% of the HPA frame
        else {
          const mi = markers.indexOf(c.channel), s = r2 / 80;
          for (const q of cells) {
            const x = x2 + q.u * r2, y = y2 + q.v * r2, v = Math.max(0, (q.m[mi] - 90) / 165);
            g.globalAlpha = ihc * 0.55; g.fillStyle = HAEMATOXYLIN; g.beginPath(); g.arc(x, y, 0.9 * s + 0.4, 0, 6.29); g.fill();
            if (v > 0) { g.globalAlpha = ihc * Math.min(1, 0.25 + v); g.fillStyle = DAB; g.beginPath(); g.arc(x, y, (1 + v) * s + 0.4, 0, 6.29); g.fill(); }
          }
        }
        g.restore();
        g.save(); g.globalAlpha = ihc; g.textAlign = "center";
        g.font = `600 13px ${font}`; g.fillStyle = head; g.fillText(real ? c.gene : `${c.channel} (simulated)`, x2, y2 - r2 - 22);
        g.font = `500 11px ${font}`; g.fillStyle = muted; g.fillText(real ? `${c.note} · ${c.antibody}` : c.note, x2, y2 - r2 - 8);
        g.restore();
      });
      g.save(); g.globalAlpha = ihc; g.font = `600 11px ${font}`; g.fillStyle = body; g.textAlign = "center";
      g.fillText(real ? "REAL IHC · other patients · not scored here" : "SIMULATED IHC", W / 2, y2 + r2 + 20, W - 24);
      if (real) { g.font = `400 11px ${font}`; g.fillStyle = muted; g.fillText(HPA_CREDIT, W / 2, y2 + r2 + 36, W - 24); }
      g.restore();
    }
  }
  const frame = (now) => { last = Math.min((now - t0) / 1000, SETTLE); draw(last); if (last < SETTLE) raf = requestAnimationFrame(frame); else raf = 0; };
  new ResizeObserver(() => { if (!raf) draw(last); }).observe(el);
  draw(0);
  return {
    play() { cancelAnimationFrame(raf); t0 = performance.now(); raf = requestAnimationFrame(frame); },
    stop() { cancelAnimationFrame(raf); raf = 0; },
    renderAt(t) { last = t; draw(t); }, // stills for screenshots
  };
}
