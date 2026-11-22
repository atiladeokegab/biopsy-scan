// RE-IMAGE 3D: one real Charville core as a point cloud on a terrain of local cell density. The camera flies from
// the whole core down to cell level and back, then one simulated look plays out: two 400 µm fields of view land,
// the cells outside them dim, and 10% of the cells they see are mis-called (they flicker, then keep the wrong
// colour, ringed). Microns are world units. One Points draw call; colours are rewritten on the CPU (~3k cells).
import * as THREE from "./vendor/three/three.module.min.js";
import { tok } from "./chart.js";
import { rng } from "./tissue.js";

export const FOV_UM = 400, EPS = 0.1, WINDOWS = 2; // the design the results section reports: 2 x 400 µm, 10% mis-called
export const TYPES = ["tumour", "stroma", "immune"]; // display classes, as tissue.js draws them
const PATIENT = "CHV-24133"; // every cell of its region (2,846 of 2,846) and a clear tumour-stroma boundary
const DOT = 10; // µm: about one cell across
const LIFT = 64; // µm of terrain at the densest point
const DIM = 0.14; // how much of a cell's colour survives outside every field of view
const SEED = 16; // the one simulated look shown: window placement and per-cell mis-call draws

// Preflight's mis-call (core/reimage_ref.py::miscall with confusion_matrix): keep the type with prob 1 - eps,
// else a uniform other type; the called type is the first k with cumsum(M[type])[k] > u.
// shortcut: K is the 3 display classes, not the source's 15 types, so a mis-call inside a class (CD4 to CD8) never shows.
export function miscall(type, u, eps, K) {
  let cum = 0;
  for (let k = 0; k < K; k++) {
    cum += k === type ? 1 - eps : eps / (K - 1);
    if (cum > u) return k;
  }
  return K - 1;
}

// Preflight's random_crop: a square of side s placed uniformly inside the bounding box [lo, hi], null if s covers it.
export function randomCrop(lo, hi, s, a) {
  if (s >= Math.max(hi[0] - lo[0], hi[1] - lo[1])) return null;
  return [0, 1].map((i) => lo[i] + a[i] * Math.max(hi[i] - lo[i] - s, 0)).concat(s);
}

export const inCrop = (x, y, c) => c === null || (x >= c[0] && x < c[0] + c[2] && y >= c[1] && y < c[1] + c[2]);

// Cells binned on an n x n grid over the square [lo, lo + side], Gaussian-blurred (sigma µm, cut at 3 sigma).
function blurred(cells, keep, lo, side, n, sigma) {
  const g = new Float32Array(n * n), cell = side / n;
  for (const c of cells) {
    if (!keep(c)) continue;
    const i = Math.min(n - 1, Math.floor((c.x - lo[0]) / cell)), j = Math.min(n - 1, Math.floor((c.y - lo[1]) / cell));
    g[j * n + i]++;
  }
  const s = sigma / cell, rad = Math.ceil(3 * s), k = [];
  for (let d = -rad; d <= rad; d++) k.push(Math.exp(-(d * d) / (2 * s * s)));
  const pass = (src, dx, dy) => {
    const out = new Float32Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      let v = 0;
      for (let d = -rad; d <= rad; d++) {
        const ii = i + d * dx, jj = j + d * dy;
        if (ii >= 0 && ii < n && jj >= 0 && jj < n) v += src[jj * n + ii] * k[d + rad];
      }
      out[j * n + i] = v;
    }
    return out;
  };
  return pass(pass(g, 1, 0), 0, 1);
}

const square = (lo, hi) => [lo, Math.max(hi[0] - lo[0], hi[1] - lo[1])];

// Local cell density, 0 to 1, with a bilinear lookup at any point in microns.
export function heightField(cells, lo, hi, n, sigma) {
  const [o, side] = square(lo, hi), g = blurred(cells, () => true, o, side, n, sigma);
  let max = 0;
  for (const v of g) max = Math.max(max, v);
  for (let i = 0; i < g.length; i++) g[i] /= max || 1;
  const cell = side / n;
  const at = (x, y) => {
    const fx = Math.max(0, Math.min(n - 1, (x - o[0]) / cell - 0.5)), fy = Math.max(0, Math.min(n - 1, (y - o[1]) / cell - 0.5));
    const i = Math.min(n - 2, Math.floor(fx)), j = Math.min(n - 2, Math.floor(fy)), u = fx - i, v = fy - j;
    const q = (a, b) => g[b * n + a];
    return (q(i, j) * (1 - u) + q(i + 1, j) * u) * (1 - v) + (q(i, j + 1) * (1 - u) + q(i + 1, j + 1) * u) * v;
  };
  return { at, max: 1 };
}

// Where tumour meets the rest: the grid point with the most of both (max of the smaller blurred density).
export function boundaryPoint(cells, lo, hi, n, sigma) {
  const [o, side] = square(lo, hi), cell = side / n;
  const t = blurred(cells, (c) => c.type === "tumour", o, side, n, sigma);
  const r = blurred(cells, (c) => c.type !== "tumour", o, side, n, sigma);
  let best = 0, bi = 0;
  for (let i = 0; i < t.length; i++) if (Math.min(t[i], r[i]) > best) { best = Math.min(t[i], r[i]); bi = i; }
  return [o[0] + ((bi % n) + 0.5) * cell, o[1] + (Math.floor(bi / n) + 0.5) * cell];
}

// The scene's script. r: framed radius in core radii; elev/azim: radians; focus: 0 the core centre, 1 the boundary.
const SHOTS = {
  whole: { r: 0.86, elev: 1.0, azim: -0.55, focus: 0 },
  cell: { r: 0.15, elev: 0.62, azim: -0.2, focus: 1 },
  look: { r: 0.9, elev: 0.86, azim: 0.3, focus: 0 },
  rest: { r: 0.4, elev: 0.72, azim: 0.75, focus: 1 },
};
const MOVES = [[0, 0.8, "whole", "whole"], [0.8, 4.6, "whole", "cell"], [4.6, 6.2, "cell", "cell"],
  [6.2, 9.0, "cell", "look"], [9.0, 10.6, "look", "look"], [10.6, 12.2, "look", "rest"], [12.2, 15, "rest", "rest"]];
const SWEEP = [9.0, 10.4], DIMS = [10.0, 10.8], FLICK = [12.2, 14.6], STEPS = [0, 4.6, 8.4, 11.2];
export const END = 15;

const clamp01 = (x) => Math.max(0, Math.min(1, x));
const ease = (x) => (x < 0.5 ? 4 * x ** 3 : 1 - (-2 * x + 2) ** 3 / 2);
const span = (t, [a, b]) => clamp01((t - a) / (b - a));

export function timeline(t) {
  t = Math.max(0, Math.min(END, t));
  const [a, b, from, to] = MOVES.findLast(([s]) => t >= s);
  const e = ease(span(t, [a, b])), A = SHOTS[from], B = SHOTS[to];
  const camera = {
    r: Math.exp(Math.log(A.r) + (Math.log(B.r) - Math.log(A.r)) * e), // zoom in log space: an even fall in scale
    elev: A.elev + (B.elev - A.elev) * e,
    azim: A.azim + (B.azim - A.azim) * e,
    focus: A.focus + (B.focus - A.focus) * e,
  };
  const settled = t >= FLICK[1];
  // Under 3 flashes a second (WCAG 2.3.1): the called colour shows on alternate 0.4 s beats, then stays.
  const called = settled || (t >= FLICK[0] && Math.floor((t - FLICK[0]) / 0.4) % 2 === 0);
  return {
    step: STEPS.findLastIndex((s) => t >= s), camera, windows: span(t, SWEEP), dim: span(t, DIMS),
    called, rings: t >= FLICK[0], settled,
  };
}

function sprite(draw) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  draw(c.getContext("2d"));
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.generateMipmaps = false; // mipmaps square off small discs
  t.minFilter = THREE.LinearFilter;
  return t;
}

export const pickPatient = (tissue) =>
  tissue.patients.find((q) => q.id === PATIENT) ?? tissue.patients.reduce((a, b) => (b.cells.length > a.cells.length ? b : a));

// The scene's clock: it runs only while playing and on screen, so scrolling away holds the take where it was.
export function clock(now, { held = 0, playing = true } = {}) {
  let t0 = null, seen = false;
  const t = () => (playing && seen && t0 !== null ? (now() - t0) / 1000 : held);
  const resume = () => { if (playing && seen) t0 = now() - held * 1000; };
  return {
    t,
    get playing() { return playing; },
    show(on) { if (on === seen) return; if (!on) held = t(); seen = on; resume(); },
    toggle() { if (playing) { held = t(); playing = false; } else { playing = true; resume(); } return playing; },
    replay() { held = 0; playing = true; resume(); },
  };
}
export const VISIBLE = 0.35; // share of the canvas on screen before the take plays
export const inView = (e) => e.isIntersecting && e.intersectionRatio >= VISIBLE;
// ?t3d=<seconds> holds the scene for stills; anything but a finite number plays it normally.
export const captureTime = (v) => (v !== null && v.trim() !== "" && Number.isFinite(+v) ? +v : null);

export function createReimage3d(canvas, tissue, { reducedMotion, freezeAt = null, onStep = () => {} }) {
  const p = pickPatient(tissue);
  const cells = p.cells, n = cells.length;
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const c of cells) { lo = [Math.min(lo[0], c.x), Math.min(lo[1], c.y)]; hi = [Math.max(hi[0], c.x), Math.max(hi[1], c.y)]; }
  const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2], RC = Math.max(hi[0] - lo[0], hi[1] - lo[1]) / 2;
  const field = heightField(cells, lo, hi, 96, 14);
  const lift = (x, y) => LIFT * field.at(x, y) ** 0.7;
  const boundary = boundaryPoint(cells, lo, hi, 48, 30);

  // One simulated look, in Preflight's draw order: the crops, then a uniform u per cell for the mis-call.
  const r = rng(SEED);
  const crops = Array.from({ length: WINDOWS }, () => randomCrop(lo, hi, FOV_UM, [r(), r()]));
  const type = new Uint8Array(n), obs = new Uint8Array(n), inside = new Uint8Array(n);
  cells.forEach((c, j) => {
    type[j] = TYPES.indexOf(c.type);
    obs[j] = miscall(type[j], r(), EPS, TYPES.length);
    inside[j] = crops.some((cr) => inCrop(c.x, c.y, cr));
  });

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const BG = new THREE.Color(tok("--bg-canvas"));
  renderer.setClearColor(BG);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(BG, 1, 2);
  const camera = new THREE.PerspectiveCamera(34, 1, 1, 20000);
  scene.add(new THREE.HemisphereLight(0xb8c7d8, BG, 0.7));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(-0.8, 0.7, 0.3); // low, so density relief casts shading
  scene.add(sun);

  // World: x right, y up, z = cell y; the core centred on the origin.
  const wx = (x) => x - mid[0], wz = (y) => y - mid[1];

  // Terrain: the core as a disc of tissue whose height is local cell density.
  const reach = cells.map((c) => Math.hypot(wx(c.x), wz(c.y))).sort((a, b) => a - b)[Math.floor(n * 0.98)]; // a stray cell never sizes the disc
  const tg = new THREE.RingGeometry(0.01, reach + DOT * 2, 160, 80);
  tg.rotateX(-Math.PI / 2);
  const tp = tg.attributes.position;
  for (let i = 0; i < tp.count; i++) tp.setY(i, lift(tp.getX(i) + mid[0], tp.getZ(i) + mid[1]));
  tg.computeVertexNormals();
  const ground = new THREE.Color(tok("--surface")).lerp(new THREE.Color(tok("--border")), 0.5);
  scene.add(new THREE.Mesh(tg, new THREE.MeshStandardMaterial({ color: ground, roughness: 0.85, metalness: 0, side: THREE.DoubleSide })));

  // Cells: one shaded disc each, sitting on the terrain.
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  cells.forEach((c, j) => pos.set([wx(c.x), lift(c.x, c.y) + DOT * 0.4, wz(c.y)], j * 3));
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const ball = sprite((g) => {
    const grad = g.createRadialGradient(24, 22, 2, 32, 32, 30);
    grad.addColorStop(0, "#ffffff"); grad.addColorStop(0.6, "#ececec"); grad.addColorStop(1, "#a4a4a4");
    g.fillStyle = grad; g.beginPath(); g.arc(32, 32, 29, 0, Math.PI * 2); g.fill();
  });
  const dots = new THREE.PointsMaterial({ size: DOT, vertexColors: true, map: ball, alphaTest: 0.5 });
  scene.add(new THREE.Points(geo, dots));

  // Rings round the cells this look mis-calls inside a field of view.
  const wrong = [...Array(n).keys()].filter((j) => inside[j] && obs[j] !== type[j]);
  const rpos = new Float32Array(wrong.length * 3);
  wrong.forEach((j, k) => rpos.set([pos[j * 3], pos[j * 3 + 1] + 0.5, pos[j * 3 + 2]], k * 3));
  const rgeo = new THREE.BufferGeometry();
  rgeo.setAttribute("position", new THREE.BufferAttribute(rpos, 3));
  const ringTex = sprite((g) => { g.strokeStyle = "#fff"; g.lineWidth = 6; g.beginPath(); g.arc(32, 32, 26, 0, Math.PI * 2); g.stroke(); });
  const rings = new THREE.Points(rgeo, new THREE.PointsMaterial({ size: DOT * 2.4, map: ringTex, color: tok("--text-head"), alphaTest: 0.5 }));
  scene.add(rings);

  // Fields of view: a bright square frame over a faint curtain of light down to the tissue.
  const H = LIFT + 34;
  const frameMat = new THREE.MeshBasicMaterial({ color: tok("--text-head"), side: THREE.DoubleSide, transparent: true });
  const wins = crops.filter(Boolean).map((cr, i) => {
    const g = new THREE.Group(), half = cr[2] / 2;
    const frame = new THREE.Mesh(new THREE.RingGeometry(half * Math.SQRT2 - 2.5, half * Math.SQRT2 + 2.5, 4, 1), frameMat);
    frame.rotation.set(-Math.PI / 2, 0, Math.PI / 4);
    frame.position.y = H;
    const curtain = new THREE.Mesh(new THREE.CylinderGeometry(half * Math.SQRT2, half * Math.SQRT2, H, 4, 1, true),
      new THREE.MeshBasicMaterial({ color: tok("--text-head"), transparent: true, opacity: 0, side: THREE.DoubleSide,
        depthWrite: false, blending: THREE.AdditiveBlending, fog: false }));
    curtain.rotation.y = Math.PI / 4;
    curtain.position.y = H / 2;
    g.add(frame, curtain);
    const at = new THREE.Vector3(wx(cr[0] + half), 0, wz(cr[1] + half));
    scene.add(g);
    return { g, curtain, at, from: new THREE.Vector3(at.x + (i ? -1 : 1) * RC * 1.6, 160, at.z) };
  });

  const base = TYPES.map((t) => new THREE.Color(tok(`--${t}`)));
  const c = new THREE.Color();
  let painted = "";
  function paint(s) {
    const key = `${s.dim}|${s.called}`;
    if (key === painted) return;
    painted = key;
    for (let j = 0; j < n; j++) {
      c.copy(base[inside[j] && s.called ? obs[j] : type[j]]);
      if (!inside[j]) c.lerp(BG, (1 - DIM) * s.dim);
      col.set([c.r, c.g, c.b], j * 3);
    }
    geo.attributes.color.needsUpdate = true;
  }

  const target = new THREE.Vector3(), bt = new THREE.Vector3(wx(boundary[0]), lift(...boundary), wz(boundary[1]));
  function draw(t, orbit) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.floor(w * renderer.getPixelRatio())) renderer.setSize(w, h, false);
    camera.aspect = w / h;
    const s = timeline(t), k = s.camera;
    // Frame k.r core radii whichever side of the canvas is shorter.
    const half = (camera.fov * Math.PI) / 360, fit = Math.min(half, Math.atan(Math.tan(half) * camera.aspect));
    const dist = (k.r * RC) / Math.sin(fit), az = k.azim + orbit;
    dots.size = DOT * (1 + 0.3 * clamp01((k.r - 0.15) / 0.7)); // a touch larger from afar, true size at cell level
    target.set(0, LIFT * 0.3, 0).lerp(bt, k.focus);
    camera.position.set(target.x + dist * Math.cos(k.elev) * Math.sin(az), target.y + dist * Math.sin(k.elev),
      target.z + dist * Math.cos(k.elev) * Math.cos(az));
    camera.near = Math.max(1, dist * 0.05);
    camera.updateProjectionMatrix();
    camera.lookAt(target);
    scene.fog.near = dist * 1.1; scene.fog.far = dist * 2.6;
    wins.forEach((wi, i) => {
      const e = ease(clamp01((s.windows - i * 0.3) / 0.7));
      wi.g.position.lerpVectors(wi.from, wi.at, e);
      wi.g.visible = e > 0;
      wi.curtain.material.opacity = 0.05 * e;
    });
    frameMat.opacity = s.windows > 0 ? 1 : 0;
    rings.visible = s.rings;
    paint(s);
    renderer.render(scene, camera);
    return s;
  }

  // Reduced motion and stills hold one moment; otherwise the clock runs while the canvas is on screen.
  const clk = clock(() => performance.now(), { held: freezeAt ?? (reducedMotion ? END : 0), playing: !reducedMotion && freezeAt === null });
  let onScreen = false, raf = null, step = -1;
  function frame() {
    raf = null;
    const t = clk.t();
    const s = draw(t, reducedMotion ? 0 : Math.max(0, t - END) * 0.05);
    if (s.step !== step) onStep((step = s.step));
    if (clk.playing && onScreen) raf = requestAnimationFrame(frame);
  }
  const kick = () => { if (raf === null) raf = requestAnimationFrame(frame); };
  new IntersectionObserver(([e]) => {
    onScreen = inView(e);
    clk.show(onScreen);
    if (onScreen) kick();
  }, { threshold: VISIBLE }).observe(canvas);
  addEventListener("resize", kick);
  kick();

  return {
    patient: p,
    replay() { clk.replay(); kick(); },
    toggle() { const on = clk.toggle(); kick(); return on; },
    get playing() { return clk.playing; },
  };
}

// The section around the canvas: provenance first (it holds without WebGL), controls only once the scene runs.
export function mountReimage($, tissue, { reducedMotion, search, create = createReimage3d }) {
  const p = pickPatient(tissue), n = p.cells.length, real = tissue.source === "MEASURED";
  $("reimage-n").textContent = n === p.cells_in_region
    ? `all ${n.toLocaleString("en-GB")} cells of one region`
    : `${n.toLocaleString("en-GB")} of the region's ${p.cells_in_region.toLocaleString("en-GB")} cells`;
  if (!real) {
    $("reimage-label").firstChild.textContent = "SYNTHETIC cells: real Charville data is not loaded; imaging simulated by Preflight. ";
    $("reimage-tag").textContent = "SYNTHETIC cells";
    $("reimage-tag").classList.remove("ok");
    $("reimage").setAttribute("aria-label", "A synthetic tumour core in 3D: two simulated fields of view, the rest dimmed, each cell in view given a 10% chance of a mis-call between colour groups");
  }
  $("reimage-credit").textContent = (real ? `Cells: Charville CODEX, Wu et al. 2022 (Zenodo 13179600, CC-BY-4.0), patient ${p.id}, region ${p.region}. ` : "") +
    "Fields of view and mis-calls: one simulated look. In the study the two fields fall on two different regions; here both are drawn on one.";
  const steps = [...$("reimage-steps").children];
  let view;
  try {
    view = create($("reimage"), tissue, {
      reducedMotion, freezeAt: captureTime(new URLSearchParams(search).get("t3d")),
      onStep: (k) => steps.forEach((li, i) => (i === k ? li.setAttribute("aria-current", "step") : li.removeAttribute("aria-current"))),
    });
  } catch {
    $("reimage").hidden = true;
    $("reimage-fallback").hidden = false;
    $("reimage-fallback").textContent = `3D view off: this browser could not start WebGL. The steps beside it still describe the ${real ? "MEASURED" : "SYNTHETIC"} cells and the SIMULATED look.`;
    $("reimage-credit").textContent += " WebGL is unavailable here, so nothing is drawn.";
    return;
  }
  if (reducedMotion) return;
  const pause = $("reimage-pause");
  $("reimage-replay").hidden = pause.hidden = false;
  $("reimage-replay").addEventListener("click", () => { view.replay(); pause.textContent = "Pause"; pause.setAttribute("aria-pressed", "false"); });
  pause.addEventListener("click", () => {
    const on = view.toggle();
    pause.textContent = on ? "Pause" : "Play";
    pause.setAttribute("aria-pressed", String(!on));
  });
}
