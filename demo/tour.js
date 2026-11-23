// The first-visit guided tour: one coach mark per feature, Next/Back/Skip, arrow keys and Esc.
// It starts by itself once per viewer (localStorage), never under automation, and the bar's
// "Start tour" button replays it. ?tour=1 forces it, ?tour=0 suppresses it.
export const SEEN_KEY = "preflight-tour-v1";

export const STEPS = [
  { target: "#workflow-app", title: "Start here: the Biopsy Scan workflow",
    text: "It plays itself: four steps on real Charville tumour tissue, from the measured cells to a study design tested before the trial." },
  { target: ".wf-steps", title: "Four steps, any order",
    text: "Tissue context, Decode the tumour, Stress-test, Routine stains. Click a step to jump to it; once the tour is closed, the arrow keys step too." },
  { target: ".wf-stage", title: "Every picture says what is real",
    text: "The tags above each picture say what is measured and what is simulated, analytic, an estimate or an illustration. The biopsy photos in steps 1 and 4 are real HPA IHC from other patients; the cell maps are rendered from real Charville CODEX cell positions." },
  { target: ".bio", title: "The pattern",
    text: "A spatial pattern found in real Charville tumour tissue, with the survival curves it splits. It is exploratory: the cutoff was chosen on the same cohort, so no p-value is claimed." },
  { target: ".speed", title: "Run the studies",
    text: "Press the button to replay one design point, 1,000 simulated studies, five ways. Bars fill at their measured rates; a bar with no measured time stays PENDING." },
  { target: ".reimage", title: "What a study sees",
    text: "Real cell positions from one patient. The study images two fields of view, and some cells are mis-called. That is the gap Biopsy Scan simulates." },
  { target: ".power-card", title: "Would the study find it?",
    text: "Power simulated on real tissue, beside what the textbook formula promises for the same patients." },
  { target: ".honest", title: "When nothing is there, it says so",
    text: "The false-alarm rate under the null, with its interval. A negative result is shown as plainly as a positive one." },
  { target: "#preflight", title: "Plan your own study",
    text: "Set patients, imaged window and mis-call rate. The power appears at once from studies already run; nothing computes live." },
  { target: "#explore", title: "Explore the design space",
    text: "Four sliders over a SYNTHETIC grid, with the tissue cores and the power curve. Every number on the page keeps its label." },
  { target: "#tour-btn", title: "That's the tour",
    text: "Start it again any time from here." },
];

// Index after a move, or -1 when the tour should close.
export function step(i, move, n = STEPS.length) {
  if (move === "skip") return -1;
  const j = i + (move === "next" ? 1 : move === "back" ? -1 : 0);
  return j >= n ? -1 : Math.max(0, j);
}

let ui = null, at = -1, raf = 0, opener = null, autoT = 0;

function els() {
  if (ui) return ui;
  const spot = document.createElement("div");
  spot.className = "tour-spot";
  spot.setAttribute("aria-hidden", "true");
  const card = document.createElement("div");
  card.className = "tour-card";
  card.setAttribute("role", "dialog");
  card.setAttribute("aria-modal", "false");
  card.setAttribute("aria-labelledby", "tour-title");
  card.innerHTML = `<p class="tour-count" id="tour-count"></p><h2 id="tour-title"></h2><p id="tour-text"></p>
    <div class="tour-actions"><button type="button" class="link" data-move="skip">Skip tour</button>
    <span><button type="button" class="tour-back" data-move="back">Back</button>
    <button type="button" class="go tour-next" data-move="next">Next</button></span></div>`;
  card.addEventListener("click", (e) => { const m = e.target.closest("[data-move]"); if (m) go(m.dataset.move); });
  document.body.append(spot, card);
  ui = { spot, card };
  return ui;
}

// Where the spotlight and the card go, so the card never covers what it describes: beside the visible part of
// the target if there is room (desktop), else below it, else above it, else pinned to the bottom with the
// spotlight cut short above it. r: the target's viewport rect; top: the bar's bottom edge.
export function layout(r, cw, ch, vw, vh, top, gap = 12, pad = 6) {
  const s = { top: Math.max(r.top - pad, top + 2), bottom: Math.min(r.bottom + pad, vh - 2), left: r.left - pad, right: r.right + pad };
  const midY = Math.max(gap, Math.min(Math.max(s.top, top + gap), vh - ch - gap));
  if (vw - s.right >= cw + 2 * gap) return { spot: s, card: { top: midY, left: s.right + gap } };
  if (s.left >= cw + 2 * gap) return { spot: s, card: { top: midY, left: s.left - gap - cw } };
  const left = Math.max(gap, Math.min((vw - cw) / 2, vw - cw - gap));
  if (vh - s.bottom >= ch + 2 * gap) return { spot: s, card: { top: s.bottom + gap, left } };
  if (s.top - top >= ch + 2 * gap) return { spot: s, card: { top: s.top - gap - ch, left } };
  const cardTop = Math.max(gap, vh - ch - gap); // a card taller than the screen scrolls inside (max-height)
  return { spot: { ...s, bottom: Math.max(s.top, Math.min(s.bottom, cardTop - gap)) }, card: { top: cardTop, left } };
}

function place() {
  if (at < 0) return;
  const t = document.querySelector(STEPS[at].target);
  const bar = document.querySelector(".bar");
  const barB = bar && STEPS[at].target !== "#tour-btn" ? Math.max(0, bar.getBoundingClientRect().bottom) : 0;
  const r = t ? t.getBoundingClientRect() : { top: innerHeight / 2, bottom: innerHeight / 2, left: innerWidth / 2, right: innerWidth / 2 };
  const vw = document.documentElement.clientWidth;
  const { spot, card } = layout(r, ui.card.offsetWidth, ui.card.offsetHeight, vw, innerHeight, barB);
  Object.assign(ui.spot.style, { top: `${spot.top}px`, left: `${spot.left}px`, width: `${spot.right - spot.left}px`, height: `${spot.bottom - spot.top}px` });
  Object.assign(ui.card.style, { top: `${card.top}px`, left: `${card.left}px` });
}

function loop() { place(); raf = requestAnimationFrame(loop); }

function show(i) {
  at = i;
  const s = STEPS[i], { card } = els();
  card.querySelector("#tour-count").textContent = `${i + 1} of ${STEPS.length}`;
  card.querySelector("#tour-title").textContent = s.title;
  card.querySelector("#tour-text").textContent = s.text;
  card.querySelector(".tour-back").disabled = i === 0;
  card.querySelector(".tour-next").textContent = i === STEPS.length - 1 ? "Done" : "Next";
  const t = document.querySelector(s.target);
  const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  // Top of the target under the bar (scroll-padding-top), so the card fits beside or below it.
  // The bar's own button only needs to be in view (the bar is not sticky on short screens).
  if (t) t.scrollIntoView({ block: s.target === "#tour-btn" ? "nearest" : "start", behavior: smooth ? "smooth" : "auto" });
  document.body.classList.add("touring");
  card.querySelector(".tour-next").focus({ preventScroll: true });
}

function close() {
  clearTimeout(autoT);
  at = -1;
  cancelAnimationFrame(raf);
  document.body.classList.remove("touring");
  ui?.spot.remove(); ui?.card.remove(); ui = null;
  removeEventListener("keydown", onKey, true);
  try { localStorage.setItem(SEEN_KEY, "1"); } catch { /* private mode: the tour just shows again */ }
  opener?.focus({ preventScroll: true });
}

// A stop whose section is not on the page is passed over in the direction of travel.
function go(move) {
  let j = step(at, move);
  while (j >= 0 && !document.querySelector(STEPS[j].target)) {
    const k = step(j, move === "back" ? "back" : "next");
    if (k === j) { j = at; break; }
    j = k;
  }
  if (j < 0) close(); else show(j);
}

function onKey(e) {
  if (at < 0) return;
  const move = { Escape: "skip", ArrowRight: "next", ArrowLeft: "back" }[e.key];
  if (!move) return;
  // A focused slider, field or select keeps its own arrow keys.
  if (move !== "skip" && e.target.closest?.("input, select, textarea, [contenteditable]")) return;
  // While the tour is open its keys win over the workflow's arrow-key stepping.
  e.preventDefault(); e.stopImmediatePropagation();
  go(move);
}

export function startTour(from = null) {
  clearTimeout(autoT); // a manual start replaces the pending first-visit start
  opener = from;
  if (at < 0) { addEventListener("keydown", onKey, true); raf = requestAnimationFrame(loop); }
  at = -1;
  go("next");
}

export function initTour() {
  const btn = document.getElementById("tour-btn");
  btn?.addEventListener("click", () => startTour(btn));
  const q = new URLSearchParams(location.search).get("tour");
  let seen = false;
  try { seen = localStorage.getItem(SEEN_KEY) === "1"; } catch { /* no storage: treat as first visit */ }
  if (q === "1" || (q !== "0" && !seen && !navigator.webdriver)) autoT = setTimeout(() => startTour(btn), 600);
}
