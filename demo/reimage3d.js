// RE-IMAGE 3D: one real Charville core as a point cloud on a terrain of local cell density. The camera flies from
// the whole core down to cell level and back, then one simulated look plays out: two 400 µm fields of view land,
// the cells outside them dim, and 10% of the cells they see are mis-called (they flicker, then keep the wrong
// colour, ringed). Microns are world units. One Points draw call; colours are rewritten on the CPU (~3k cells).
import * as THREE from "./vendor/three/three.module.min.js";
import { tok } from "./chart.js";
import { rng } from "./tissue.js";

export const FOV_UM = 400, EPS = 0.1, WINDOWS = 2; // the design the results section reports: 2 x 400 µm, 10% mis-called
const TYPES = ["tumour", "stroma", "immune"]; // display classes, as tissue.js draws them
const PATIENT = "CHV-24133"; // every cell of its region (2,846 of 2,846) and a clear tumour-stroma boundary
const DOT = 8; // µm: about one cell across
const LIFT = 40; // µm of terrain at the densest point
const DIM = 0.14; // how much of a cell's colour survives outside every field of view
const SEED = 11; // the one simulated look shown: window placement and per-cell mis-call draws

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
  whole: { r: 1.05, elev: 1.05, azim: -0.55, focus: 0 },
  cell: { r: 0.15, elev: 0.62, azim: -0.2, focus: 1 },
  look: { r: 1.0, elev: 0.88, azim: 0.3, focus: 0 },
  rest: { r: 0.92, elev: 0.8, azim: 0.6, focus: 0.1 },
};
const MOVES = [[0, 0.8, "whole", "whole"], [0.8, 4.6, "whole", "cell"], [4.6, 6.2, "cell", "cell"],
  [6.2, 9.0, "cell", "look"], [9.0, 14.5, "look", "rest"]];
const SWEEP = [9.0, 10.4], DIMS = [10.0, 10.8], FLICK = [11.0, 13.4], STEPS = [0, 4.6, 8.4, 10.8];
export const END = 14.5;

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
  return t;
}

export function createReimage3d(canvas, tissue, { reducedMotion, freezeAt = null, onStep = () => {} }) {
  const p = tissue.patients.find((q) => q.id === PATIENT) ?? tissue.patients.reduce((a, b) => (b.cells.length > a.cells.length ? b : a));
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
  scene.add(new THREE.HemisphereLight(0xb8c7d8, BG, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(-0.6, 1, 0.35);
  scene.add(sun);

  // World: x right, y up, z = cell y; the core centred on the origin.
  const wx = (x) => x - mid[0], wz = (y) => y - mid[1];

  // Terrain: a slab of tissue whose height is local cell density, fading into the background past the core.
  const side = RC * 2.6, seg = 128;
  const tg = new THREE.PlaneGeometry(side, side, seg, seg);
  tg.rotateX(-Math.PI / 2);
  const tp = tg.attributes.position, tcol = new Float32Array(tp.count * 3);
  const ground = new THREE.Color(tok("--surface")).lerp(new THREE.Color(tok("--border")), 0.7);
  for (let i = 0; i < tp.count; i++) {
    const x = tp.getX(i) + mid[0], y = tp.getZ(i) + mid[1];
    tp.setY(i, lift(x, y));
    const fade = clamp01((Math.hypot(tp.getX(i), tp.getZ(i)) / RC - 1.02) / 0.25);
    const c = ground.clone().lerp(BG, fade);
    tcol.set([c.r, c.g, c.b], i * 3);
  }
  tg.setAttribute("color", new THREE.BufferAttribute(tcol, 3));
  tg.computeVertexNormals();
  scene.add(new THREE.Mesh(tg, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, metalness: 0 })));

  // Cells: one shaded disc each, sitting on the terrain.
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  cells.forEach((c, j) => pos.set([wx(c.x), lift(c.x, c.y) + DOT * 0.4, wz(c.y)], j * 3));
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const ball = sprite((g) => {
    const grad = g.createRadialGradient(24, 22, 2, 32, 32, 30);
    grad.addColorStop(0, "#ffffff"); grad.addColorStop(0.55, "#d8d8d8"); grad.addColorStop(1, "#7c7c7c");
    g.fillStyle = grad; g.beginPath(); g.arc(32, 32, 29, 0, Math.PI * 2); g.fill();
  });
  scene.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: DOT, vertexColors: true, map: ball, alphaTest: 0.5 })));

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
      wi.curtain.material.opacity = 0.09 * e;
    });
    frameMat.opacity = s.windows > 0 ? 1 : 0;
    rings.visible = s.rings;
    paint(s);
    renderer.render(scene, camera);
    return s;
  }

  // The clock runs only while the canvas is on screen and not paused; reduced motion shows the last frame, still.
  let t0 = null, held = freezeAt ?? (reducedMotion ? END : 0), playing = !reducedMotion && freezeAt === null;
  let onScreen = false, raf = null, step = -1;
  const now = () => (playing && t0 !== null ? (performance.now() - t0) / 1000 : held);
  function frame() {
    raf = null;
    const t = now();
    const s = draw(t, reducedMotion ? 0 : Math.max(0, t - END) * 0.05);
    if (s.step !== step) onStep((step = s.step));
    if (playing && onScreen) raf = requestAnimationFrame(frame);
  }
  const kick = () => { if (raf === null) raf = requestAnimationFrame(frame); };
  new IntersectionObserver(([e]) => {
    onScreen = e.isIntersecting;
    if (onScreen && playing && t0 === null) t0 = performance.now(); // the take starts when you arrive
    if (onScreen) kick();
  }, { threshold: 0.35 }).observe(canvas);
  addEventListener("resize", kick);
  kick();

  return {
    patient: p,
    replay() { held = 0; t0 = performance.now(); playing = !reducedMotion; kick(); },
    toggle() {
      if (playing) { held = now(); playing = false; } else { t0 = performance.now() - held * 1000; playing = true; kick(); }
      return playing;
    },
    get playing() { return playing; },
  };
}
