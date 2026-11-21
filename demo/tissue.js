// TISSUE: one thin slab of real (or SYNTHETIC) cells per patient, stacked, with field-of-view squares.
// One Points draw call per slab; colours are rewritten on the CPU when the design changes (~30k cells).
import * as THREE from "./vendor/three/three.module.min.js";

const UM = 10; // microns per world unit
const GAP = 40; // world units between slabs
const COLOURS = { tumour: "#FF7A59", stroma: "#2BD4A4", immune: "#F5C451" };
const TYPES = Object.keys(COLOURS);
const BG = new THREE.Color("#0B1018");
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
  renderer.setClearColor("#070B11");
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 1, 6000);
  const root = new THREE.Group();
  scene.add(root);

  const R = tissue.region_um / UM;
  const base = Object.fromEntries(TYPES.map((t) => [t, new THREE.Color(COLOURS[t])]));
  const dot = dotTexture();
  const plateMat = new THREE.MeshBasicMaterial({ color: "#121A25", transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false });
  const edgeMat = new THREE.LineBasicMaterial({ color: "#2A3A4C" });
  const fovMat = new THREE.LineDashedMaterial({ color: "#F4F7FA", dashSize: 5, gapSize: 3.5 });

  const slabs = tissue.patients.map((p, i) => {
    const n = p.cells.length;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const type = new Uint8Array(n), flip = new Float32Array(n), other = new Uint8Array(n);
    const r = rng(1000 + i);
    p.cells.forEach((c, j) => {
      pos[j * 3] = c.x / UM - R / 2;
      pos[j * 3 + 1] = 0.6;
      pos[j * 3 + 2] = c.y / UM - R / 2;
      type[j] = TYPES.indexOf(c.type);
      flip[j] = r(); // a cell is mis-called when flip < rate, so raising the rate only adds mis-calls
      other[j] = (type[j] + 1 + Math.floor(r() * 2)) % 3;
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 4.6, vertexColors: true, map: dot, alphaTest: 0.5, transparent: false,
    }));
    const group = new THREE.Group();
    const plate = new THREE.Mesh(new THREE.PlaneGeometry(R, R), plateMat);
    plate.rotation.x = -Math.PI / 2;
    const edge = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
      [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => new THREE.Vector3((a * R) / 2, 0, (b * R) / 2))), edgeMat);
    const squares = new THREE.Group();
    group.add(plate, edge, pts, squares);
    root.add(group);
    // Fractional square centres, fixed per patient, so a bigger field of view grows in place.
    const sr = rng(5000 + i), spots = Array.from({ length: 8 }, () => [sr(), sr()]);
    return { group, pts, col, pos, type, flip, other, squares, spots, n };
  });

  let yaw = 0.6, pitch = 0.42, spin = !reducedMotion, dragging = null, drop = null, shown = 0;

  function squareLoop(s) {
    const g = new THREE.BufferGeometry().setFromPoints(
      [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]].map(([a, b]) => new THREE.Vector3((a * s) / 2, 0, (b * s) / 2)));
    const l = new THREE.Line(g, fovMat);
    l.computeLineDistances();
    return l;
  }

  function update(d) {
    const full = d.fov_um === "full";
    const s = full ? R : Math.min(R, d.fov_um / UM);
    const k = full ? 1 : d.regions;
    shown = Math.max(1, Math.min(slabs.length, Math.round(d.patients / 25)));
    slabs.forEach((sl, i) => {
      sl.group.visible = i < shown;
      sl.group.position.y = (i - (shown - 1) / 2) * GAP;
      sl.squares.children.forEach((c) => c.geometry.dispose());
      sl.squares.clear();
      const boxes = full ? [[0, 0]] : sl.spots.slice(0, k).map(([u, v]) => [(u - 0.5) * (R - s), (v - 0.5) * (R - s)]);
      for (const [cx, cz] of boxes) {
        const l = squareLoop(s);
        l.position.set(cx, 1.2, cz);
        sl.squares.add(l);
      }
      const c = new THREE.Color();
      for (let j = 0; j < sl.n; j++) {
        const t = sl.flip[j] < d.miscall ? sl.other[j] : sl.type[j];
        c.copy(base[TYPES[t]]);
        const x = sl.pos[j * 3], z = sl.pos[j * 3 + 2];
        const inside = boxes.some(([cx, cz]) => Math.abs(x - cx) <= s / 2 && Math.abs(z - cz) <= s / 2);
        if (!inside) c.lerp(BG, 1 - DIM);
        sl.col[j * 3] = c.r; sl.col[j * 3 + 1] = c.g; sl.col[j * 3 + 2] = c.b;
      }
      sl.pts.geometry.attributes.color.needsUpdate = true;
    });
    drop = reducedMotion ? null : performance.now();
    canvas.setAttribute("aria-label",
      `${shown} patient slabs of ${full ? "whole-region" : `${k} square${k > 1 ? "s" : ""} of ${d.fov_um} µm`} imaging, ` +
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
    // Far enough to fit the stack's height, and its width on narrow (phone) canvases.
    const dist = Math.max(330 + shown * 30 + R * 0.6, (R * 1.5) / (0.574 * camera.aspect));
    camera.position.set(Math.sin(yaw) * Math.cos(pitch) * dist, Math.sin(pitch) * dist, Math.cos(yaw) * Math.cos(pitch) * dist);
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  canvas.addEventListener("pointerdown", (e) => { dragging = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    yaw -= (e.clientX - dragging.x) * 0.008;
    pitch = Math.max(0.08, Math.min(1.45, pitch + (e.clientY - dragging.y) * 0.006));
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
    pitch = Math.max(0.08, Math.min(1.45, pitch + step[1]));
  });

  requestAnimationFrame(frame);
  return {
    update,
    setSpin(on) { spin = on; },
    get spinning() { return spin; },
  };
}
