"""R-16 pre-registration: the Helsinki candidate 'pEMT tumour cells bordered by fibroblasts' in UPMC terms.

Committed before any search ran and before any outcome was opened. It was written from UPMC cell
types and marker names only (raw_data/*.cell_types.csv and *.expression.csv headers); no label or
outcome file was read. Editing RULE changes RULE_SHA256, and audit/test_rediscovery_definition.py fails.

Source: Punovuori et al., Cell 187:7267 (2024). pEMT epithelium was defined by Slug and Vimentin
(with K14), and the poor-survival stroma was CAF-enriched. Survival was about 21% at five years for
pEMT with CAF-enriched stroma, and about 90% for pEMT with immune-enriched stroma.
"""
from __future__ import annotations

import hashlib
import json

# Every UPMC cell type on disk (17), sorted as core.cohorts.load_cells sorts them.
UPMC_TYPES = sorted(["APC", "B cell", "CD4 T cell", "CD8 T cell", "Granulocyte", "Lymph vessel", "Macrophage",
                     "Naive immune cell", "Stromal / Fibroblast", "Tumor", "Tumor (CD15+)", "Tumor (CD20+)",
                     "Tumor (CD21+)", "Tumor (Ki67+)", "Tumor (Podo+)", "Unassigned", "Vessel"])
# Every UPMC CODEX marker on disk (expression.csv columns other than CELL_ID). R-17 uses M = 22; the plan's ~40 was an estimate.
UPMC_MARKERS = ["CD11b", "CD14", "CD15", "CD163", "CD20", "CD21", "CD31", "CD34", "CD3e", "CD4", "CD45", "CD45RA",
                "CD45RO", "CD68", "CD8", "CollagenIV", "HLA-DR", "Ki67", "PanCK", "Podoplanin", "Vimentin", "aSMA"]
M = len(UPMC_MARKERS)

RULE = {
    "id": "R-16 pEMT tumour bordered by fibroblasts (UPMC)",
    "expressible": True,
    # The one primary candidate is a column of the v1 grammar, so its rank among all candidates is well defined.
    "feature": "Tumor (Podo+)|Stromal / Fibroblast|25um|fraction",
    "focal": "Tumor (Podo+)",
    "neighbour": "Stromal / Fibroblast",
    "radius_um": 25,
    "statistic": "fraction",
    "direction": "higher is worse: hazard ratio > 1; a top rank with HR < 1 counts as not rediscovered",
    "pEMT_proxy": "Podoplanin-positive tumour cells (UPMC type 'Tumor (Podo+)'). Podoplanin is a pEMT-programme "
                  "gene in HNSCC. Slug is not in the UPMC panel. Vimentin is, but the grammar has no "
                  "marker-gated focal type, so Vimentin+ tumour is not a candidate column",
    "fibroblast_proxy": "UPMC type 'Stromal / Fibroblast'. aSMA is a marker, not a type, so it stays out of the primary candidate",
    "deviations": ["The plan's 26 um becomes 25 um, the nearest grammar radius",
                   "The plan's '>30% fibroblasts' threshold becomes the continuous fraction the screen tests"],
    "outcomes_opened": False,
}
RULE_SHA256 = hashlib.sha256(json.dumps(RULE, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
