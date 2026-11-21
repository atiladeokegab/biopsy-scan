"""Print each cohort's patients, endpoints, strata, exact C and grammar hash, and check the reference
re-imager reproduces the truth layer on one real region. Run: make summary (after make data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # python -I drops the cwd

import numpy as np  # noqa: E402

from core import cohorts  # noqa: E402
from core.grammar import grammar, grammar_hash, n_features  # noqa: E402
from core.reimage_ref import reimage_region  # noqa: E402
from core.truth import neighbours, truth_region  # noqa: E402

for coh in cohorts.COHORTS:
    reg = cohorts.load_regions(coh)
    pts = cohorts.load_patients(reg)
    cells = cohorts.load_cells(coh)
    g = grammar(cells["type_names"])
    K = len(g["type_names"])
    print(f"== {coh}: {len(reg)} regions, {len(cells['cell_type']):,} cells, {len(pts)} patients, "
          f"{int(np.nansum(pts.event))} events, {int(pts.time_days.isna().sum())} without an endpoint")
    if coh == "upmc":
        print(f"   tuple+case sensitivity: {len(cohorts.load_patients(cohorts.load_regions(coh, identity='tuple+case')))} patients")
    print(f"   coverslips {sorted(reg.coverslip.unique())}; patients on >1 coverslip: {(pts.n_coverslips > 1).sum()}")
    print(f"   types K={K}, tumour focal T={len(g['focal_types'])}: {[g['type_names'][t] for t in g['focal_types']]}")
    print(f"   C = {n_features(g)}   grammar sha256 = {grammar_hash(g)}")
    p = cells["region_ptr"]
    xy, ty = cells["xy_um"][p[0]:p[1]], cells["cell_type"][p[0]:p[1]]
    nbr = neighbours(xy)
    S0, m0 = truth_region(xy, ty, g["focal_types"], K)
    S, m = reimage_region(nbr, xy, ty, g["focal_types"], K, crop=None, eps=0.0)
    print(f"   {cells['region_ids'][0]}: {len(xy)} cells, mean neighbours at 25/50/100 um "
          f"{np.round((nbr['rad_end'] - nbr['ptr'][:-1, None]).mean(0), 1).tolist()}, "
          f"re-imager == truth: {np.array_equal(S, S0) and np.array_equal(m, m0)}")
