// "Preflight workflow": the four-step stepper. It loads data/tma.json (real Charville cells) and data/preflight.json,
// mounts each step's module into its .wf-vis, auto-plays one step every STEP_MS and holds on the last; a click jumps.
const STEP_MS = 7000;
const section = document.getElementById("flow");
const nodes = [...section.querySelectorAll(".wf-node")];
const texts = [...section.querySelectorAll(".wf-text")];
const vis = [...section.querySelectorAll(".wf-vis")];
const mods = [];
let current = -1, timer = 0;

function show(i, auto) {
  clearTimeout(timer);
  mods[current]?.stop();
  current = i;
  nodes.forEach((n, k) => { n.toggleAttribute("aria-current", k === i); if (k === i) n.setAttribute("aria-current", "step"); n.classList.toggle("done", k < i); });
  texts.forEach((t, k) => { t.hidden = k !== i; });
  vis.forEach((v, k) => { v.hidden = k !== i; });
  mods[i]?.play();
  if (auto && i < nodes.length - 1) timer = setTimeout(() => show(i + 1, true), STEP_MS);
}

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

const stub = (el, why) => {
  const p = document.createElement("p");
  p.className = "wf-stub";
  p.textContent = why;
  el.append(p);
  return { play() {}, stop() {} };
};

try {
  const [tma, preflight] = await Promise.all([getJSON("data/tma.json"), getJSON("data/preflight.json")]);
  const data = { tma, preflight };
  for (const [i, file] of ["step1.js", "step2.js", "step3.js", "step4.js"].entries()) {
    try {
      mods[i] = (await import(`./${file}`)).mount(vis[i], data);
    } catch (e) {
      console.warn(`workflow ${file}:`, e);
      mods[i] = stub(vis[i], "This step is being prepared.");
    }
  }
} catch (e) {
  console.warn("workflow data:", e);
  vis.forEach((v, i) => { mods[i] = stub(v, "Tissue data not loaded."); });
}
nodes.forEach((n, i) => n.addEventListener("click", () => show(i, false)));
show(0, !matchMedia("(prefers-reduced-motion: reduce)").matches);
