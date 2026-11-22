// The first-visit guided tour: one coach mark per feature, Next/Back/Skip, arrow keys and Esc.
// It starts by itself once per viewer (localStorage), never under automation, and the bar's
// "Start tour" button replays it. ?tour=1 forces it, ?tour=0 suppresses it.
export const SEEN_KEY = "preflight-tour-v1";

export const STEPS = [
  { target: "#flow", title: "Start here: the Preflight workflow",
    text: "Four steps on real tumour tissue: the tissue, decoding it, stress-testing the study design, and the routine stains it comes down to. Each step labels its numbers as measured or simulated." },
  { target: ".bio", title: "The pattern",
    text: "A spatial pattern found in real Charville tumour tissue, with the survival curves it splits. It is exploratory: the cutoff was chosen on the same cohort, so no p-value is claimed." },
  { target: ".speed", title: "Run the studies",
    text: "Press the button to replay one design point, 1,000 simulated studies, five ways. Bars fill at their measured rates; a bar with no measured time stays PENDING." },
  { target: ".reimage", title: "What a study sees",
    text: "Real cell positions from one patient. The study images two fields of view, and some cells are mis-called. That is the gap Preflight simulates." },
  { target: ".power-card", title: "Would the study find it?",
    text: "Power simulated on real tissue, beside what the textbook formula promises for the same patients." },
  { target: ".honest", title: "When nothing is there, it says so",
    text: "The false-alarm rate under the null, with its interval. A negative result is shown as plainly as a positive one." },
  { target: "#preflight", title: "Preflight your own study",
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

let ui = null, at = -1, raf = 0, opener = null;

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

function place() {
  if (at < 0) return;
  const t = document.querySelector(STEPS[at].target);
  const r = t ? t.getBoundingClientRect() : null;
  const pad = 6;
  Object.assign(ui.spot.style, r
    ? { top: `${r.top - pad}px`, left: `${r.left - pad}px`, width: `${r.width + 2 * pad}px`, height: `${r.height + 2 * pad}px` }
    : { top: "50%", left: "50%", width: "0px", height: "0px" });
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
  // Leave room at the bottom for the card on phones.
  if (t) t.scrollIntoView({ block: t.offsetHeight > innerHeight * 0.55 ? "start" : "center", behavior: smooth ? "smooth" : "auto" });
  document.body.classList.add("touring");
  card.querySelector(".tour-next").focus({ preventScroll: true });
}

function close() {
  at = -1;
  cancelAnimationFrame(raf);
  document.body.classList.remove("touring");
  ui?.spot.remove(); ui?.card.remove(); ui = null;
  removeEventListener("keydown", onKey);
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
  if (e.key === "Escape") { e.preventDefault(); go("skip"); }
  else if (e.key === "ArrowRight") { e.preventDefault(); go("next"); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); go("back"); }
}

export function startTour(from = null) {
  opener = from;
  if (at < 0) { addEventListener("keydown", onKey); raf = requestAnimationFrame(loop); }
  at = -1;
  go("next");
}

export function initTour() {
  const btn = document.getElementById("tour-btn");
  btn?.addEventListener("click", () => startTour(btn));
  const q = new URLSearchParams(location.search).get("tour");
  let seen = false;
  try { seen = localStorage.getItem(SEEN_KEY) === "1"; } catch { /* no storage: treat as first visit */ }
  if (q === "1" || (q !== "0" && !seen && !navigator.webdriver)) setTimeout(() => startTour(btn), 600);
}
