"""Print each cohort's patients, endpoints, strata, exact C and grammar hash, and check the reference
re-imager reproduces the truth layer on one real region. Run: make summary (after make data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # python -I drops the cwd

import numpy as np  # noqa: E402

from core import cohorts  # noqa: E402
from core.grammar import grammar, grammar_hash, n_features  # noqa: E402
from core.reimage_ref import reimage  # noqa: E402
from core.truth import truth  # noqa: E402

FULL = {"patients": None, "regions_per_patient": None, "fov_um": None, "eps": 0.0, "seed": 0}
check = "--full" in sys.argv  # re-image every patient and compare with the whole truth layer (minutes)

for coh in cohorts.COHORTS:
    c = cohorts.load(coh)
    g = grammar(c["type_names"])
    K, P = len(g["type_names"]), len(c["time"])
    print(f"== {coh} (patient level, verified identity): {P} patients, {len(c['patient_of_region'])} regions, "
          f"{len(c['cell_type']):,} cells, {int(c['event'].sum())} events, strata {np.bincount(c['strata']).tolist()}")
    if coh == "upmc":
        for ident in ("tuple", "tuple+case"):
            pts = cohorts.load_patients(cohorts.load_regions(coh, identity=ident))
            print(f"   descriptive only, identity={ident}: {len(pts)} patients, {int(pts.event.sum())} events, "
                  f"{int((pts.stratum < 0).sum())} spanning both coverslip groups")
    print(f"   types K={K}, tumour focal T={len(g['focal_types'])}: {[g['type_names'][t] for t in g['focal_types']]}")
    print(f"   C = {n_features(g)}   grammar sha256 = {grammar_hash(g)}")
    sub = {k: c[k] for k in ("cell_type", "x", "y", "region_id", "type_names")}
    first = c["region_id"] == 0
    sub = {k: (v[first] if k != "type_names" else v) for k, v in sub.items()}
    sub["patient_of_region"] = np.zeros(1, np.int32)
    same = np.array_equal(reimage(sub, FULL, 0), truth(sub), equal_nan=True)
    print(f"   {c['region_names'][0]}: re-imager == truth at full field, eps=0: {same}")
    if check:
        X0 = truth(c)
        print(f"   all {P} patients: re-imager == truth: {np.array_equal(reimage(c, FULL, 0), X0, equal_nan=True)}; "
              f"NaN cells in X: {int(np.isnan(X0).sum())} of {X0.size}")
