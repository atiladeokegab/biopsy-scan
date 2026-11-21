"""The design-point unit on invented tissue: both methods run, one threshold per design, calls compared."""
import numpy as np

from bench.fleet import synthetic_cohort
from core.reimage_ref import reimage_region
from gpu import design_point as dp
from stats import score


def _cohort(patients=8):
    cells, _ = synthetic_cohort(n_patients=patients, regions_per_patient=1, cells=300, side=300.0, seed=1)
    rng = np.random.default_rng(0)
    cells.update(time=rng.exponential(size=patients), event=rng.random(patients) < 0.6,
                 strata=np.repeat([0, 1], patients // 2))
    return cells, {"patients": patients, "regions_per_patient": 1, "fov_um": 250.0, "eps": 0.1, "seed": 0}


def test_null_order_shuffles_only_within_strata():
    strata = np.array([0, 0, 1, 1, 1, 2])
    order = dp.null_order(strata, seed=3)
    assert sorted(order) == list(range(6)) and np.array_equal(strata[order], strata)


def test_one_threshold_per_design_and_calls_compared_study_by_study():
    cells, design = _cohort()
    z_score = lambda X, t, e, s: np.nan_to_num(score(X, t, e, s)["z"])  # noqa: E731
    pairs = [dp.our_study(cells, design, seed, reimage_region, z_score) for seed in range(6)]
    z, null_z = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    assert z.shape == null_z.shape == (6, z.shape[1]) and not np.array_equal(z, null_z)
    calls, cut = dp.our_calls(z, null_z, alpha=0.2)
    assert calls.shape == (6,) and np.isfinite(cut)
    tree_cells = {**cells, "_nbr": {r: None for r in cells["_nbr"]}}
    lab = [dp.lab_study(tree_cells, design, seed, n_resamples=19) for seed in range(6)]
    assert all(0 < p <= 1 for _, p in lab)
    agree = dp.agreement([p for _, p in lab], calls, alpha=0.2)
    assert agree["of"] == 6 and agree["same_call"] + agree["lab_only"] + agree["ours_only"] == 6
