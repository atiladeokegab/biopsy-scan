"""bench.external_baseline: same cells as our re-imager, grammar-ordered squidpy features, the Bonferroni call.

squidpy and lifelines live only in the Modal image (lifelines needs pandas<3, #59): their tests skip here
and run in that image's pins (bench/modal_external.py PINS).
"""

import numpy as np
import pytest

from bench import external_baseline as eb

NAMES = ["B cell", "Stroma", "Tumor 1", "Tumor 2"]  # focal: the two tumour types
DESIGN = {"patients": 5, "regions_per_patient": 2, "fov_um": 150.0, "eps": 0.2, "seed": 0}


def _cells(seed=0, P=5, per=160):
    rng = np.random.default_rng(seed)
    R = 2 * P
    return {"cell_type": rng.integers(0, len(NAMES), R * per).astype(np.int8),
            "x": rng.uniform(0, 300, R * per).astype(np.float32), "y": rng.uniform(0, 300, R * per).astype(np.float32),
            "region_id": np.repeat(np.arange(R), per).astype(np.int32),
            "patient_of_region": np.repeat(np.arange(P), 2).astype(np.int32), "type_names": NAMES}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_study_cells_give_the_re_imagers_features_bit_for_bit(seed):
    from core.reimage_ref import reimage
    from core.truth import layout
    cells = _cells()
    L = layout(cells)
    sc = eb.study_cells(cells, DESIGN, seed)
    assert len(sc) == DESIGN["patients"] and all(len(looks) == 2 for looks in sc)
    assert sum(len(t) for looks in sc for _, _, t in looks) < len(cells["x"])  # cropped: fewer cells than the cohort
    X = eb.our_features(sc, L["focal"], L["K"])
    assert np.array_equal(X, reimage(_cells(), DESIGN, seed), equal_nan=True)


def test_bonferroni_calls_on_min_p_times_m_and_survives_no_fits():
    p = np.array([np.nan, 0.01, 1e-4, 0.5])
    assert eb.bonferroni(p, 630) == {"min_p_adj": pytest.approx(0.063), "top": 2, "call": False}
    assert eb.bonferroni(np.array([1e-5]), 630)["call"]
    assert eb.bonferroni(np.full(3, np.nan), 630) == {"min_p_adj": None, "top": None, "call": False}


def test_agreement_is_perfect_on_identical_inputs_and_counts_nan_disagreement():
    rng = np.random.default_rng(0)
    X = rng.random((20, 12))
    z = rng.normal(size=12)
    a = eb.agreement(X, X, z, z)
    assert a["fraction_pearson"] == pytest.approx(1) and a["enrichment_spearman"] == pytest.approx(1)
    assert a["top1_same"] and a["top10_overlap"] == 10 and a["nan_only_squidpy"] == a["nan_only_ours"] == 0
    Xs = X.copy()
    Xs[0, 1] = np.nan
    assert eb.agreement(Xs, X, z, z)["nan_only_squidpy"] == 1


def test_squidpy_fraction_is_the_pooled_edge_share_in_grammar_order():
    pytest.importorskip("squidpy")
    from core.grammar import RADII_UM
    from core.truth import layout, truth_counts
    cells = _cells()
    L = layout(cells)
    looks = eb.study_cells(cells, DESIGN, 4)[0]
    f = eb.squidpy_features(looks, NAMES, L["focal"], seed=0).reshape(len(L["focal"]), L["K"], len(RADII_UM), 2)
    pooled = sum(np.stack([truth_counts(xy, ty, L["K"])[ty == t].sum(0) for t in L["focal"]]) for _, xy, ty in looks)
    share = pooled / pooled.sum(-1, keepdims=True)  # (T, R, K): edges from focal t to j over all edges from t
    assert np.allclose(f[..., 0], share.transpose(0, 2, 1))
    assert np.isfinite(f[..., 1]).all()


def test_lifelines_screen_finds_a_planted_feature_and_skips_constant_columns():
    pytest.importorskip("lifelines")
    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(size=(n, 3))
    X[:, 2] = 1.0
    X[:5, 1] = np.nan
    time = rng.exponential(np.exp(-0.8 * X[:, 0]))
    event = (rng.random(n) < 0.8).astype(float)
    out = eb.lifelines_screen(X, time, event, np.repeat([0, 1], n // 2))
    assert out["z"][0] > 4 and abs(out["z"][1]) < 4 and np.isnan(out["p"][2]) and out["fitted"] == 2


def test_ratios_carry_both_arms_the_cross_run_label_and_the_different_statistic_wording():
    r = eb.ratios(eb.LAB_S_PER_STUDY["value"], 8, 1)
    assert r["ratio_vs_lab"] == pytest.approx(1) and r["ratio_vs_fused"] == pytest.approx(0.8809 / 0.0419)
    assert r["settings"]["cores"] == 8 and r["settings"]["squidpy.gr.nhood_enrichment"]["n_perms"] == 1000
    h = eb.headline({**r, "seconds_per_study": 0.8809, "studies": 50, "derived_off_the_shelf_s_per_study": 0.7,
                     "worker_study_s": {"median": 7, "min": 6, "max": 9}})
    assert "cross-run, same container class" in h and "not Preflight's speedup" in h and "range 6-9 s" in h
    assert "MEASURED, instrumented workflow, pool wall" in h and "0.70 s/study, DERIVED" in h and "not a wall-clock" in h


def test_failed_fits_are_excluded_from_agreement_ranks_and_counted():
    rng = np.random.default_rng(2)
    X = rng.random((20, 12))
    zo = np.arange(12.0)[::-1]  # our top feature is column 0
    none = eb.agreement(X, X, np.full(12, np.nan), zo)  # every lifelines fit failed
    assert none["top1_same"] is None and none["top10_overlap"] is None and none["z_spearman_abs"] is None
    assert none["z_excluded"] == 12
    zs = zo.copy()
    zs[:3] = np.nan  # three failed fits, ours ranks them highest
    part = eb.agreement(X, X, zs, zo)
    # each method ranks its own finite features: ours's best (0) is invalid for lifelines, whose best is 3
    assert part["z_excluded"] == 3 and part["top_lifelines"] == 3 and part["top_ours"] == 0
    assert part["top1_same"] is False
    assert part["top10_overlap"] == 7  # lifelines' top 10 is its 9 finite (3..11); ours is 0..9
    s = eb.summarize({"records": [_rec(none), _rec(part)], "pool_wall_s": 10.0, "workers": 1, "plant_feature": 0})
    assert s["top1_same"] == {"k": 0, "of": 1} and s["z_excluded"] == 15


def test_headline_per_study_is_the_measured_pool_wall_and_worker_time_is_a_separate_derived_rate():
    recs = [_rec(eb.agreement(np.ones((3, 2)), np.ones((3, 2)), np.ones(2), np.ones(2)), study_s=s) for s in (8.0, 12.0)]
    s = eb.summarize({"records": recs, "pool_wall_s": 30.0, "workers": 2, "plant_feature": 0})
    assert s["seconds_per_study"] == pytest.approx(15.0)  # pool wall / studies: measured
    assert s["derived_off_the_shelf_s_per_study"] == pytest.approx(20.0 / 2 / 2)  # summed worker time / workers / studies
    assert s["ratio_vs_lab"] == pytest.approx(15.0 / eb.LAB_S_PER_STUDY["value"])
    assert "DERIVED" in s["derived_basis"] and "MEASURED" in s["seconds_per_study_basis"]


def _rec(agreement, study_s=1.0):
    return {"stream": "alternative", "i": 0, "study_s": study_s, "cells_s": 0.0, "squidpy_s": study_s / 2,
            "lifelines_s": study_s / 2, "fitted": 0, "failed": 0, "lab_call": False, "lab_top": None, "ours_top": 0,
            "agreement": agreement}


def test_same_host_summary_takes_median_and_range_per_arm_and_ratio_per_repeat():
    s = eb.same_host_summary([{"external_s_per_study": 24.0, "fused_s_per_study": 0.04},
                              {"external_s_per_study": 30.0, "fused_s_per_study": 0.05},
                              {"external_s_per_study": 20.0, "fused_s_per_study": 0.02}])
    assert s["external_s_per_study"] == {"median": 24.0, "min": 20.0, "max": 30.0, "n": 3}
    assert s["fused_s_per_study"] == {"median": 0.04, "min": 0.02, "max": 0.05, "n": 3}
    assert s["ratio_per_repeat"] == pytest.approx([600.0, 600.0, 1000.0])  # per repeat, not median / median
    assert s["ratio"]["median"] == pytest.approx(600.0) and s["ratio"]["max"] == pytest.approx(1000.0)
