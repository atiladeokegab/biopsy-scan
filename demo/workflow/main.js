// "Preflight workflow": a four-step walkthrough of real Charville tissue. start(root) renders TEMPLATE into root
// (#wf-root by default), loads data/tma.json and data/preflight.json, mounts each step's module into its .wf-vis,
// auto-plays (one step every STEP_MS[i], holding on the last) and steps on a click or the arrow keys.
// It starts itself when the page has a #wf-root.
const STEP_MS = [7000, 7000, 10500, 13500];

export const TEMPLATE = `<div class="wf" id="flow" role="region" aria-labelledby="wf-title" aria-roledescription="walkthrough">
  <div class="wf-copy">
    <p class="wf-eyebrow">Preflight workflow</p>
    <h2 class="wf-head" id="wf-title">Know whether a study can work before the first slide is stained.</h2>
    <div class="wf-text" data-step="0"><h3>Start from real tumour tissue</h3><p>Biopsy cores from the Charville cohort, imaged in 40 channels per cell. Every point is a measured cell, glowing in the colours of its own marker signal.</p></div>
    <div class="wf-text" data-step="1" hidden><h3>Read every cell, then test every pattern</h3><p>Every cell carries one of the 15 cell types published with the data; the screen tests each neighbourhood pattern against survival. Exploratory result: the best pattern scores z 3.46, maxT p 0.1275, within chance.</p></div>
    <div class="wf-text" data-step="2" hidden><h3>Stress-test the design before the trial</h3><p>A hospital images two 400 µm fields and mis-calls one cell in ten. For 200 patients the textbook formula promises 80% power; simulated on this tissue, it is 27%.</p></div>
    <div class="wf-text" data-step="3" hidden><h3>Aim: one or two routine stains</h3><p>The core separates into six of its measured marker channels. The aim is the one or two routine stains a lab already runs; <span data-sim="the brown cores are simulated stains on this core, because the real images did not load.">the brown cores are real stains from other patients, shown for comparison.</span> No marker pair has been chosen on outcomes yet.</p></div>
  </div>
  <div class="wf-stage">
    <div class="wf-vis" data-step="0"><p class="wf-label"><span class="tag">Rendered from real Charville CODEX cells: positions and types REAL, imaging noise SIMULATED</span></p></div> <!-- /vis -->
    <div class="wf-vis" data-step="1" hidden><p class="wf-label"><span class="tag">Rendered from real Charville CODEX cells: positions and types REAL, imaging noise SIMULATED</span></p><div class="wf-result" hidden><b>Screen result, exploratory</b><span>Best pattern z 3.46 · maxT p 0.1275</span><span>Within chance: no pattern passes the 0.05 cutoff.</span></div></div> <!-- /vis -->
    <div class="wf-vis" data-step="2" hidden><p class="wf-label"><span class="tag measured">MEASURED cells</span> <span class="tag">SIMULATED fields and mis-calls</span> <span class="tag">ANALYTIC formula</span> <span class="tag">ESTIMATE simulation</span></p></div> <!-- /vis -->
    <div class="wf-vis" data-step="3" hidden><p class="wf-label"><span class="tag measured">MEASURED channels</span> <span class="tag">ILLUSTRATION marker search</span> <span class="tag measured" data-sim="SIMULATED IHC">REAL IHC, other patients</span></p></div> <!-- /vis -->
    <p class="wf-credit">Charville CODEX, Wu et al. 2022, Zenodo 13179600, CC-BY-4.0</p>
  </div>
  <ol class="wf-steps">
    <li><button type="button" class="wf-node" data-step="0" aria-current="step"><span>01</span> Tissue context</button></li>
    <li><button type="button" class="wf-node" data-step="1"><span>02</span> Decode the tumour</button></li>
    <li><button type="button" class="wf-node" data-step="2"><span>03</span> Stress-test</button></li>
    <li><button type="button" class="wf-node" data-step="3"><span>04</span> Routine stains</button></li>
  </ol>
</div>`;

export async function start(root = document.getElementById("wf-root")) {
  root.innerHTML = TEMPLATE;
  const section = root.querySelector(".wf");
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
    if (auto && i < nodes.length - 1) timer = setTimeout(() => show(i + 1, true), STEP_MS[i]);
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
    const [tma, preflight] = await Promise.all([getJSON(new URL("../data/tma.json", import.meta.url)), getJSON(new URL("../data/preflight.json", import.meta.url))]);
    const data = { tma, preflight };
    for (const [i, file] of ["step1.js", "step2.js", "step3.js", "step4.js"].entries()) {
      try {
        mods[i] = (await import(new URL(`./${file}`, import.meta.url))).mount(vis[i], data);
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
  // Arrow keys step through it while it is on screen, unless a control (the sliders below) has the focus.
  addEventListener("keydown", (e) => {
    if (!["ArrowRight", "ArrowLeft"].includes(e.key) || e.target.closest?.("input, select, textarea")) return;
    const b = section.getBoundingClientRect();
    if (b.bottom < innerHeight * 0.3 || b.top > innerHeight * 0.7) return;
    clearTimeout(timer); // any arrow key ends autoplay, even at either end
    const next = current + (e.key === "ArrowRight" ? 1 : -1);
    if (next >= 0 && next < nodes.length) { e.preventDefault(); show(next, false); }
  });
  show(0, !matchMedia("(prefers-reduced-motion: reduce)").matches);
  return { show };
}

if (typeof document !== "undefined" && document.getElementById("wf-root")) start();
