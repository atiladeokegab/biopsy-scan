// TISSUE: one thin slab of real (or SYNTHETIC) cells per patient, six side by side, with field-of-view squares.
// One Points draw call per slab; colours are rewritten on the CPU when the design changes (~30k cells).
import * as THREE from "./vendor/three/three.module.min.js";
import { tok } from "./chart.js";

// The scene is sized in core widths, and each slab is cut to its own core, not microns: every core (720 µm real, 2500 µm synthetic) is
// R world units across, so dots stay fine and slabs keep their spacing whatever region_um is.
const R = 250; // world units across one core
const PITCH = R * 1.12; // centre to centre of neighbouring cores
const COLS = 3, SHOWN = 6; // six patients, three by two
const DOT = R * 0.016; // point size: a fine dot at real-cell density, at any core size
const TYPES = ["tumour", "stroma", "immune"]; // CSS tokens --tumour, --stroma, --immune
const DIM = 0.14; // how much of a cell's colour survives outside every field of view

// Small deterministic RNG so squares and mis-calls stay put between redraws.
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s + 0x6d2b79f5) >>> 0), (((s ^ (s >>> 15)) * (s | 1)) >>> 0) / 4294967296);
}

function dotTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  g.fillStyle = "#fff";
  g.beginPath(); g.arc(32, 32, 28, 0, Math.PI * 2); g.fill();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function createTissue(canvas, tissue, { reducedMotion }) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(tok("--bg-canvas"));
  const BG = new THREE.Color(tok("--bg-canvas"));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 1, 6000);
  const root = new THREE.Group();
  scene.add(root);

  const UM = tissue.region_um / R; // microns per world unit
  const base = Object.fromEntries(TYPES.map((t) => [t, new THREE.Color(tok(`--${t}`))]));
  const dot = dotTexture();
  const plateMat = new THREE.MeshBasicMaterial({ color: tok("--surface"), transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false });
  const edgeMat = new THREE.LineBasicMaterial({ color: tok("--border") });
  const fovMat = new THREE.LineDashedMaterial({ color: tok("--text-head"), dashSize: 5, gapSize: 3.5 });

  const slabs = tissue.patients.map((p, i) => {
    const n = p.cells.length;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const type = new Uint8Array(n), flip = new Float32Array(n), other = new Uint8Array(n);
    const r = rng(1000 + i);
    // The core's own extent (real cores are not square): cells are centred on the slab, squares stay inside it.
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const c of p.cells) { x0 = Math.min(x0, c.x); x1 = Math.max(x1, c.x); y0 = Math.min(y0, c.y); y1 = Math.max(y1, c.y); }
    const core = { w: (x1 - x0) / UM, h: (y1 - y0) / UM };
    p.cells.forEach((c, j) => {
      pos[j * 3] = (c.x - x0) / UM - core.w / 2;
      pos[j * 3 + 1] = 0.6;
      pos[j * 3 + 2] = (c.y - y0) / UM - core.h / 2;
      type[j] = TYPES.indexOf(c.type);
      flip[j] = r(); // a cell is mis-called when flip < rate, so raising the rate only adds mis-calls
      other[j] = (type[j] + 1 + Math.floor(r() * 2)) % 3;
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({
      size: DOT, vertexColors: true, map: dot, alphaTest: 0.5, transparent: false,
    }));
    const group = new THREE.Group();
    const plate = new THREE.Mesh(new THREE.PlaneGeometry(core.w * 1.06, core.h * 1.06), plateMat);
    plate.rotation.x = -Math.PI / 2;
    const edge = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
      [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => new THREE.Vector3((a * core.w * 1.06) / 2, 0, (b * core.h * 1.06) / 2))), edgeMat);
    const squares = new THREE.Group();
    group.add(plate, edge, pts, squares);
    root.add(group);
    // Fractional square centres, fixed per patient, so a bigger field of view grows in place.
    const sr = rng(5000 + i), spots = Array.from({ length: 8 }, () => [sr(), sr()]);
    return { group, pts, col, pos, type, flip, other, squares, spots, n, core };
  });

  let yaw = 0.6, pitch = 0.5, spin = !reducedMotion, dragging = null, drop = null, shown = 0;

  function squareLoop(w, h) {
    const g = new THREE.BufferGeometry().setFromPoints(
      [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]].map(([a, b]) => new THREE.Vector3((a * w) / 2, 0, (b * h) / 2)));
    const l = new THREE.Line(g, fovMat);
    l.computeLineDistances();
    return l;
  }

  function update(d) {
    const full = d.fov_um === "full";
    const s = full ? Infinity : d.fov_um / UM;
    const k = full ? 1 : d.regions;
    shown = Math.min(SHOWN, slabs.length);
    const rows = Math.ceil(shown / COLS);
    slabs.forEach((sl, i) => {
      sl.group.visible = i < shown;
      sl.group.position.set(((i % COLS) - (COLS - 1) / 2) * PITCH, 0, (Math.floor(i / COLS) - (rows - 1) / 2) * PITCH);
      sl.squares.children.forEach((c) => c.geometry.dispose());
      sl.squares.clear();
      // Each field of view is clamped to the core: never wider than it, never hanging off its edge.
      const w = Math.min(s, sl.core.w), h = Math.min(s, sl.core.h);
      const boxes = sl.spots.slice(0, k).map(([u, v]) => [(u - 0.5) * (sl.core.w - w), (v - 0.5) * (sl.core.h - h)]);
      for (const [cx, cz] of boxes) {
        const l = squareLoop(w, h);
        l.position.set(cx, 1.2, cz);
        sl.squares.add(l);
      }
      const c = new THREE.Color();
      for (let j = 0; j < sl.n; j++) {
        const t = sl.flip[j] < d.miscall ? sl.other[j] : sl.type[j];
        c.copy(base[TYPES[t]]);
        const x = sl.pos[j * 3], z = sl.pos[j * 3 + 2];
        const inside = boxes.some(([cx, cz]) => Math.abs(x - cx) <= w / 2 && Math.abs(z - cz) <= h / 2);
        if (!inside) c.lerp(BG, 1 - DIM);
        sl.col[j * 3] = c.r; sl.col[j * 3 + 1] = c.g; sl.col[j * 3 + 2] = c.b;
      }
      sl.pts.geometry.attributes.color.needsUpdate = true;
    });
    drop = reducedMotion ? null : performance.now();
    canvas.setAttribute("aria-label",
      `${shown} patient slabs of ${full ? "whole-core" : `${k} square${k > 1 ? "s" : ""} of ${d.fov_um} µm`} imaging, ` +
      `${Math.round(d.miscall * 100)}% of cells mis-called (recoloured).`);
    return shown;
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.floor(w * renderer.getPixelRatio())) renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function frame(now) {
    resize();
    if (spin && !dragging) yaw += 0.0025;
    // Squares drop onto the slabs over 600 ms after every design change.
    const t = drop ? Math.min(1, (now - drop) / 600) : 1;
    const lift = 70 * (1 - t) ** 3;
    for (const sl of slabs) sl.squares.position.y = lift;
    if (t >= 1) drop = null;
    // Each core turns in place, so the camera only has to fit the fixed grid of cores (plus a turning core's corners).
    for (const sl of slabs) sl.group.rotation.y = yaw;
    const half = Math.tan((camera.fov * Math.PI) / 360);
    const hw = ((COLS - 1) / 2) * PITCH + R * 0.62, hd = (Math.ceil(shown / COLS) / 2) * PITCH + R * 0.1;
    const dist = Math.max((hw * 1.04) / (half * camera.aspect), (hd * Math.sin(pitch) * 1.2 + R * 0.1) / half);
    camera.position.set(0, Math.sin(pitch) * dist, Math.cos(pitch) * dist);
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  canvas.addEventListener("pointerdown", (e) => { dragging = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    yaw -= (e.clientX - dragging.x) * 0.008;
    pitch = Math.max(0.3, Math.min(1.35, pitch + (e.clientY - dragging.y) * 0.006));
    dragging = { x: e.clientX, y: e.clientY };
  });
  const stop = () => (dragging = null);
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointercancel", stop);
  canvas.addEventListener("keydown", (e) => {
    const step = { ArrowLeft: [-0.15, 0], ArrowRight: [0.15, 0], ArrowUp: [0, 0.08], ArrowDown: [0, -0.08] }[e.key];
    if (!step) return;
    e.preventDefault();
    yaw += step[0];
    pitch = Math.max(0.3, Math.min(1.35, pitch + step[1]));
  });

  requestAnimationFrame(frame);
  return {
    update,
    setSpin(on) { spin = on; },
    get spinning() { return spin; },
  };
}
