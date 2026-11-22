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
export function miscall(type, u, eps, K) {
  let cum = 0;
  for (let k = 0; k < K; k++) {
    cum += k === type ? 1 - eps : eps / (K - 1);
    if (cum > u) return k;
  }
  return K - 1;
}

// The same 15-type model seen through the 3 colour groups (exact, not an approximation): a source type in group g is
// called as one of the other K - 1 types uniformly with prob eps, so it lands in group h != g with prob eps * n_h / (K - 1).
export function groupMatrix(sizes, eps) {
  const K = sizes.reduce((a, b) => a + b);
  return sizes.map((ng, g) => sizes.map((nh, h) => (h === g ? 1 - (eps * (K - ng)) / (K - 1) : (eps * nh) / (K - 1))));
}
export function callFrom(row, u) {
  let cum = 0;
  for (let k = 0; k < row.length; k++) if ((cum += row[k]) > u) return k;
  return row.length - 1;
}

// Camera distance and target that fit the points inside +-margin of the frame at this angle and aspect.
export function fitView(points, elev, azim, fovDeg, aspect, margin) {
  const box = new THREE.Box3().setFromPoints(points), target = box.getCenter(new THREE.Vector3());
  const cam = new THREE.PerspectiveCamera(fovDeg, aspect, 0.1, 1e6), q = new THREE.Vector3();
  const dir = new THREE.Vector3(Math.cos(elev) * Math.sin(azim), Math.sin(elev), Math.cos(elev) * Math.cos(azim));
  const fits = (d) => {
    cam.position.copy(target).addScaledVector(dir, d); cam.lookAt(target); cam.updateMatrixWorld();
    return points.every((p) => { q.copy(p).project(cam); return q.z < 1 && Math.abs(q.x) <= margin && Math.abs(q.y) <= margin; });
  };
  let lo = 1, hi = 1e5;
  for (let i = 0; i < 40; i++) { const mid = Math.sqrt(lo * hi); if (fits(mid)) hi = mid; else lo = mid; }
  return { target, dist: hi };
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

// The scene's script. fit: what the shot frames (the region, the region and its fields, or the fields), fitted at the
// canvas's own aspect; cell level instead frames r region radii round the tumour-stroma boundary. r also sizes the dots.
const SHOTS = {
  whole: { r: 0.86, elev: 1.0, azim: -0.55, fit: "region" },
  cell: { r: 0.15, elev: 0.62, azim: -0.2, fit: null },
  look: { r: 0.9, elev: 0.86, azim: 0.3, fit: "all" },
  rest: { r: 0.6, elev: 0.72, azim: 0.75, fit: "fields" },
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
    from, to, e,
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

// One simulated look, in Preflight's draw order: the crops, then a uniform u per cell for the mis-call. It needs no
// WebGL, so the counts show even when the view cannot start.
export function drawStudy(tissue) {
  const p = pickPatient(tissue);
  const cells = p.cells, n = cells.length;
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const c of cells) { lo = [Math.min(lo[0], c.x), Math.min(lo[1], c.y)]; hi = [Math.max(hi[0], c.x), Math.max(hi[1], c.y)]; }
  // The study puts each field on a different region; both are drawn on this one, so on a small region they overlap.
  const r = rng(SEED), crops = Array.from({ length: WINDOWS }, () => randomCrop(lo, hi, FOV_UM, [r(), r()]));
  const classes = Object.values(tissue.meta?.display_classes ?? {});
  const M = groupMatrix(TYPES.map((g) => classes.filter((v) => v === g).length || 1), EPS); // no type map: 3 equal groups
  const type = new Uint8Array(n), obs = new Uint8Array(n), inside = new Uint8Array(n);
  let seen = 0, crossed = 0;
  cells.forEach((c, j) => {
    type[j] = TYPES.indexOf(c.type);
    obs[j] = callFrom(M[type[j]], r());
    inside[j] = crops.some((cr) => inCrop(c.x, c.y, cr));
    seen += inside[j]; crossed += inside[j] && obs[j] !== type[j];
  });
  const perField = crops.map((cr) => cells.filter((c) => inCrop(c.x, c.y, cr)).length); // what each field alone sees
  return { p, cells, n, lo, hi, crops, type, obs, inside, seen, crossed, perField };
}

export function createReimage3d(canvas, tissue, { reducedMotion, freezeAt = null, onStep = () => {}, onEnd = () => {} }) {
  const { p, cells, n, lo, hi, crops, type, obs, inside } = drawStudy(tissue);
  const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2], RC = Math.max(hi[0] - lo[0], hi[1] - lo[1]) / 2;
  const field = heightField(cells, lo, hi, 96, 14);
  const lift = (x, y) => LIFT * field.at(x, y) ** 0.7;
  const boundary = boundaryPoint(cells, lo, hi, 48, 30);


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

  // Rings round the cells mis-called into another colour group inside a field of view.
  const wrong = [...Array(n).keys()].filter((j) => inside[j] && obs[j] !== type[j]);
  const rpos = new Float32Array(wrong.length * 3);
  wrong.forEach((j, k) => rpos.set([pos[j * 3], pos[j * 3 + 1] + 0.5, pos[j * 3 + 2]], k * 3));
  const rgeo = new THREE.BufferGeometry();
  rgeo.setAttribute("position", new THREE.BufferAttribute(rpos, 3));
  const ringTex = sprite((g) => { g.strokeStyle = "#fff"; g.lineWidth = 6; g.beginPath(); g.arc(32, 32, 26, 0, Math.PI * 2); g.stroke(); });
  const rings = new THREE.Points(rgeo, new THREE.PointsMaterial({ size: DOT * 2.4, map: ringTex, color: tok("--text-head"), alphaTest: 0.5 }));
  scene.add(rings);

  // Fields of view: a thin square frame, labelled with its size outside its near edge.
  const H = LIFT + 34;
  const frameMat = new THREE.MeshBasicMaterial({ color: tok("--text-body"), side: THREE.DoubleSide, transparent: true });
  const labels = [], labelTex = (() => {
    // A pill on the canvas ground, so the label reads over dots and frame edges.
    const cv = document.createElement("canvas"); cv.width = 512; cv.height = 144; // 2x: sharp at dpr 2
    const g = cv.getContext("2d");
    g.font = `600 72px ${tok("--font-sans") || "sans-serif"}`; g.textAlign = "center"; g.textBaseline = "middle";
    g.lineJoin = "round"; g.lineWidth = 14; g.strokeStyle = tok("--bg-canvas"); g.strokeText(`${FOV_UM} µm`, 256, 76); // a halo, not a plate
    g.fillStyle = tok("--text-head"); g.fillText(`${FOV_UM} µm`, 256, 76);
    const tx = new THREE.CanvasTexture(cv); tx.colorSpace = THREE.SRGBColorSpace; return tx;
  })();
  const wins = crops.filter(Boolean).map((cr, i) => {
    const g = new THREE.Group(), half = cr[2] / 2;
    const frame = new THREE.Mesh(new THREE.RingGeometry(half * Math.SQRT2 - 1.2, half * Math.SQRT2 + 1.2, 4, 1), frameMat);
    frame.rotation.set(-Math.PI / 2, 0, Math.PI / 4);
    frame.position.y = H;
    const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: labelTex, transparent: true, depthTest: false, sizeAttenuation: false }));
    labels.push(label);
    label.position.set(i ? half + 60 : 0, H, i ? 0 : half + 60); // off its own edge that lies outside the other frame
    label.center.set(0.5, 0.5); // centred 60 µm out, so the frame edge never crosses the text
    label.renderOrder = 2;
    g.add(frame, label);
    const at = new THREE.Vector3(wx(cr[0] + half), 0, wz(cr[1] + half));
    scene.add(g);
    return { g, label, half, at, from: new THREE.Vector3(at.x + (i ? -1 : 1) * RC * 1.6, 160, at.z) };
  });

  // What each fitted shot must keep in frame: the region's rim (ground to peak), and the fields' corners (ground to frame).
  const rim = [...Array(48)].flatMap((_, i) => [0, LIFT].map((y) =>
    new THREE.Vector3((reach + DOT * 2) * Math.cos((i / 48) * 2 * Math.PI), y, (reach + DOT * 2) * Math.sin((i / 48) * 2 * Math.PI))));
  const corners = wins.flatMap(({ at }) => [[-1, -1], [-1, 1], [1, -1], [1, 1]].flatMap(([a, b]) => [0, H].map((y) =>
    new THREE.Vector3(at.x + (a * FOV_UM) / 2, y, at.z + (b * FOV_UM) / 2))));
  const subjects = { region: rim, all: rim.concat(corners), fields: corners.length ? corners : rim };

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
  function draw(t) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const pr = renderer.getPixelRatio();
    if (canvas.width !== Math.floor(w * pr) || canvas.height !== Math.floor(h * pr)) renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // Labels a fixed 26 css px tall: unattenuated sprites span scale.y / (2 tan(fov/2)) of the canvas height.
    const ly = (26 / h) * 2 * Math.tan((camera.fov * Math.PI) / 360);
    for (const l of labels) l.scale.set(ly * (512 / 144), ly, 1);
    const s = timeline(t), k = s.camera, az = k.azim;
    // Each end of the move fitted at the current angle, so the phone's aspect stays framed too.
    const half = (camera.fov * Math.PI) / 360, fit = Math.min(half, Math.atan(Math.tan(half) * camera.aspect));
    const view = (name) => {
      const pts = subjects[SHOTS[name].fit];
      return pts ? fitView(pts, k.elev, az, camera.fov, camera.aspect, 0.88) : { target: bt, dist: (SHOTS[name].r * RC) / Math.sin(fit) };
    };
    const A = view(k.from), B = k.to === k.from ? A : view(k.to);
    const dist = Math.exp(Math.log(A.dist) + (Math.log(B.dist) - Math.log(A.dist)) * k.e);
    target.lerpVectors(A.target, B.target, k.e);
    const perPx = (2 * dist * Math.tan((camera.fov * Math.PI) / 360)) / h; // µm per css px at the target
    dots.size = Math.max(DOT * (1 + 0.3 * clamp01((k.r - 0.15) / 0.7)), 3 * perPx); // at least 3 css px, so a phone shows cells
    rings.material.size = Math.max(DOT * 2.4, 9 * perPx); // a ring at least 9 css px, so step 4 reads at 390
    camera.position.set(target.x + dist * Math.cos(k.elev) * Math.sin(az), target.y + dist * Math.sin(k.elev),
      target.z + dist * Math.cos(k.elev) * Math.cos(az));
    camera.near = Math.max(1, dist * 0.05);
    camera.updateProjectionMatrix();
    camera.lookAt(target);
    scene.fog.near = dist * 1.1; scene.fog.far = dist * 2.6;
    const gap = Math.max(60, 34 * perPx); // at least 34 css px off the edge on any screen
    wins.forEach((wi, i) => {
      const e = ease(clamp01((s.windows - i * 0.3) / 0.7));
      wi.label.position.set(i ? wi.half + gap : 0, H, i ? 0 : wi.half + gap);
      wi.g.position.lerpVectors(wi.from, wi.at, e);
      wi.g.visible = e > 0;
      wi.label.material.opacity = e;
    });
    frameMat.opacity = s.windows > 0 ? 1 : 0;
    rings.visible = s.rings;
    paint(s);
    renderer.render(scene, camera);
    return s;
  }

  // Reduced motion and stills hold one moment; otherwise the clock runs while the canvas is on screen.
  const clk = clock(() => performance.now(), { held: freezeAt ?? (reducedMotion ? END : 0), playing: !reducedMotion && freezeAt === null });
  let onScreen = false, raf = null, step = -1, ended = false;
  function frame() {
    raf = null;
    const t = clk.t();
    const s = draw(t);
    if (s.step !== step) onStep((step = s.step));
    if (t >= END && !ended) { ended = true; onEnd(); } // a still held at the end shows the end state too
    if (clk.playing && onScreen && t < END) raf = requestAnimationFrame(frame); // the take ends still: no motion without a Pause
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
    replay() { ended = false; clk.replay(); kick(); },
    toggle() { const on = clk.toggle(); kick(); return on; },
    get playing() { return clk.playing; },
  };
}

// The section around the canvas: provenance first (it holds without WebGL), controls only once the scene runs.
export function mountReimage($, tissue, { reducedMotion, search, create = createReimage3d }) {
  const study = drawStudy(tissue), p = study.p, n = study.n, real = tissue.source === "MEASURED";
  const fmt = (v) => v.toLocaleString("en-GB");
  $("reimage-seen").textContent = `${fmt(study.seen)} of its ${fmt(n)}`; // both fields together: they overlap here
  $("reimage-crossed").textContent = `${fmt(study.crossed)} cells`;
  $("reimage-size").textContent = `${fmt(Math.round(study.hi[0] - study.lo[0]))} × ${fmt(Math.round(study.hi[1] - study.lo[1]))} µm`;
  $("reimage-n").textContent = n === p.cells_in_region
    ? `all ${n.toLocaleString("en-GB")} cells`
    : `${n.toLocaleString("en-GB")} of its ${p.cells_in_region.toLocaleString("en-GB")} cells`;
  if (!real) {
    $("reimage-label").firstChild.textContent = "SYNTHETIC cells. Real Charville data is not loaded. Imaging simulated by Preflight. ";
    $("reimage-tag").textContent = "SYNTHETIC cells";
    $("reimage-tag").classList.remove("measured");
    $("reimage").setAttribute("aria-label", "A synthetic tumour region in 3D. Two simulated fields of view, the rest dimmed. Each cell in view has a 10% chance of a mis-call. Those that cross colour groups are ringed.");
  }
  $("reimage-credit").textContent = (real ? `Cells: Charville CODEX, Wu et al. 2022 (Zenodo 13179600, CC-BY-4.0), patient ${p.id}, region ${p.region}. ` : "") +
    "The fields of view and mis-calls are one simulated draw.";
  const steps = [...$("reimage-steps").children];
  let view;
  try {
    $("reimage").hidden = false; // hidden while the data loads, so the well holds no blank canvas
    view = create($("reimage"), tissue, {
      reducedMotion, freezeAt: captureTime(new URLSearchParams(search).get("t3d")),
      onEnd: () => { $("reimage-pause").hidden = true; $("reimage-steps").classList.remove("playing"); }, // the take is over: only Replay, every step
      onStep: (k) => steps.forEach((li, i) => (i === k ? li.setAttribute("aria-current", "step") : li.removeAttribute("aria-current"))),
    });
  } catch (e) {
    console.warn("3D view off:", e.message);
    $("reimage").hidden = true;
    $("reimage-still").hidden = !real; // the still is of the real region: never shown over synthetic cells
    $("reimage-fallback").hidden = false;
    $("reimage-fallback").textContent = `The 3D view needs WebGL, which this browser has off. Turn on hardware acceleration in its settings, or open the page in another browser. ${
      real ? "Below: the last frame, rendered on a GPU." : "The steps still describe the SYNTHETIC cells."}`;
    return;
  }
  $("reimage-fallback").hidden = true;
  if (reducedMotion) return;
  $("reimage-steps").classList.add("playing");
  const pause = $("reimage-pause");
  $("reimage-replay").hidden = pause.hidden = false;
  $("reimage-replay").addEventListener("click", () => { view.replay(); pause.hidden = false; $("reimage-steps").classList.add("playing"); pause.textContent = "Pause the fly-through"; pause.setAttribute("aria-pressed", "false"); });
  pause.addEventListener("click", () => {
    const on = view.toggle();
    pause.textContent = on ? "Pause the fly-through" : "Play the fly-through";
    $("reimage-steps").classList.toggle("playing", on); // paused, the phone shows every step
    pause.setAttribute("aria-pressed", String(!on));
  });
}
