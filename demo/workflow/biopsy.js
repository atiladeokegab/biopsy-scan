// Step 1, "Tissue context": real head and neck biopsy photos (HPA IHC, other patients), one core per marker.
// Photos only: nothing is drawn over them. The analysis itself runs on the Charville cell tables (steps 2-4).
import { HPA_CREDIT } from "./step4.js";

export const CORES = [ // img/hpa_manifest.json holds each file's source URL, licence and sha256
  { file: "hpa_best_01_KRT5.webp", gene: "KRT5", note: "tumour cells" },
  { file: "hpa_best_07_ACTA2.webp", gene: "ACTA2", note: "stroma" },
  { file: "hpa_best_08_PTPRC.webp", gene: "PTPRC", note: "immune cells" },
  { file: "hpa_best_11_CD3E.webp", gene: "CD3E", note: "T cells" },
  { file: "hpa_best_12_CD8A.webp", gene: "CD8A", note: "cytotoxic T cells" },
  { file: "hpa_best_13_CD68.webp", gene: "CD68", note: "macrophages" },
];

export function mount(el, data) {
  const grid = el.appendChild(document.createElement("div"));
  grid.className = "wf-hpa";
  grid.innerHTML = CORES.map((c) => `<figure><img src="${new URL(`./img/${c.file}`, import.meta.url).href}" alt="Real ${c.gene} IHC biopsy core, head and neck cancer, Human Protein Atlas" width="760" height="760" decoding="async"><figcaption><b>${c.gene}</b> ${c.note}</figcaption></figure>`).join("")
    + `<p class="wf-hpa-credit">${HPA_CREDIT}</p>`;
  return { play() {}, stop() {} };
}
