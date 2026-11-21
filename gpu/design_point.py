"""One design point, the speed unit: S simulated Charville studies end to end, by the lab's way and by ours.

Lab way, per study: re-image with fresh cKDTree recounts (bench.lab_today.tree_region, the 7.52 s/study path
in cpu-ladder.json), then a 1,000-shuffle conditional maxT test (stats.permutation_test); call = p <= alpha.

Our way, per design: re-image with the CSR kernel (CPU or GPU), score each study once, score one independent
null study per study (the same re-imaged X with outcomes shuffled within strata, so X carries no signal), take
one threshold from the S null maxima (stats.threshold), and call = maxT > threshold.

Both use the drawn patients' real Charville outcomes, so their calls can be compared study by study.
"""
from __future__ import annotations

import numpy as np

from bench.lab_today import tree_region
from core.reimage_ref import draw, reimage
from stats import maxT, permutation_test, threshold

ALPHA = 0.05
LAB_RESAMPLES = 1000


def outcomes(cells: dict, design: dict, seed: int):
    p = draw(cells, design, seed)["patients"]
    return cells["time"][p], cells["event"][p].astype(float), cells["strata"][p]


def null_order(strata, seed: int) -> np.ndarray:
    """A within-strata shuffle of patient rows, seeded apart from the study's own draws."""
    rng = np.random.default_rng([seed, 7])
    order = np.arange(len(strata))
    for g in np.unique(strata):
        ix = np.flatnonzero(strata == g)
        order[ix] = rng.permutation(ix)
    return order


def lab_study(cells: dict, design: dict, seed: int, n_resamples: int = LAB_RESAMPLES) -> tuple[float, float]:
    """(maxT, p) the lab's way. `cells` must carry `_nbr` = {region: None}: the tree path builds no CSR."""
    X = reimage(cells, design, seed, region_kernel=tree_region)
    t, e, s = outcomes(cells, design, seed)
    r = permutation_test(X, t, e, s, n_resamples=n_resamples, seed=seed)
    return r["maxT"], r["p"]


def our_study(cells: dict, design: dict, seed: int, region_kernel, score) -> tuple[np.ndarray, np.ndarray]:
    """(z, null z) for one study; score(X, time, event, strata) -> z[C]."""
    X = reimage(cells, design, seed, region_kernel=region_kernel)
    t, e, s = outcomes(cells, design, seed)
    o = null_order(s, seed)
    return score(X, t, e, s), score(X, t[o], e[o], s)


def our_calls(z, null_z, alpha: float = ALPHA) -> tuple[np.ndarray, float]:
    """One threshold for the whole design from the null studies; a call per study."""
    cut = threshold({"alpha": alpha}, np.asarray(null_z))
    return maxT(np.asarray(z)) > cut, cut


def agreement(lab_p, ours, alpha: float = ALPHA) -> dict:
    lab = np.asarray(lab_p) <= alpha
    ours = np.asarray(ours, bool)
    return {"same_call": int((lab == ours).sum()), "of": len(lab), "both_positive": int((lab & ours).sum()),
            "lab_only": int((lab & ~ours).sum()), "ours_only": int((~lab & ours).sum()), "alpha": alpha}
