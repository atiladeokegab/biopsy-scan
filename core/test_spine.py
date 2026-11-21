"""Spine checks on a tiny invented fixture (no real data, offline)."""
import zipfile

import numpy as np
import pytest

from core import cohorts
from core.grammar import feature_names, grammar, grammar_hash, n_features
from core.reimage_ref import confusion_matrix, crop_mask, miscall, random_crop, reimage_region
from core.truth import neighbours, truth_counts, truth_region

TYPES = ["B cell", "Stroma", "Tumor", "Tumor (Ki67+)"]
K, FOCAL = 4, [2, 3]


def region(n=400, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 300.0
    xy[1] = xy[0]                      # duplicate position
    xy[2] = xy[0] + [25.0, 0.0]        # exactly on the 25 um boundary
    xy[3] = xy[0] + [0.0, 50.0]        # exactly on the 50 um boundary
    return xy, rng.integers(0, K, n).astype(np.int16)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_crop_eps0_reproduces_truth_exactly(seed):
    xy, ty = region(seed=seed)
    nbr = neighbours(xy)
    S0, m0 = truth_region(xy, ty, FOCAL, K)
    S, m = reimage_region(nbr, xy, ty, FOCAL, K, crop=None, eps=0.0, rng=np.random.default_rng(9))
    assert np.array_equal(S, S0) and np.array_equal(m, m0)
    assert m0.sum() == np.isin(ty, FOCAL).sum() and S0.any()


def test_crop_equals_truth_of_the_visible_cells():
    xy, ty = region(seed=3)
    crop = random_crop(xy, 120.0, np.random.default_rng(4))
    vis = crop_mask(xy, crop)
    assert 0 < vis.sum() < len(xy)
    S, m = reimage_region(neighbours(xy), xy, ty, FOCAL, K, crop=crop, eps=0.0)
    S1, m1 = truth_region(xy[vis], ty[vis], FOCAL, K)
    assert np.array_equal(S, S1) and np.array_equal(m, m1)


def test_neighbour_lists_sorted_and_radius_offsets():
    xy, _ = region(seed=5)
    nbr = neighbours(xy)
    for i in range(0, len(xy), 37):
        d = nbr["dist"][nbr["ptr"][i]:nbr["ptr"][i + 1]]
        assert np.all(np.diff(d) >= 0) and d.max(initial=0) <= 100
        for r, rad in enumerate((25.0, 50.0, 100.0)):
            assert nbr["rad_end"][i, r] - nbr["ptr"][i] == (d <= rad).sum()
    assert truth_counts(xy, np.zeros(len(xy), np.int16), K)[0, 0, 0] >= 2  # duplicate and boundary cell count


def test_miscall_rate_and_identity():
    ty = np.random.default_rng(0).integers(0, K, 200_000).astype(np.int16)
    u = np.random.default_rng(1).random(len(ty))
    assert np.array_equal(miscall(ty, confusion_matrix(K, 0.0), u), ty)
    obs = miscall(ty, confusion_matrix(K, 0.25), u)
    assert abs((obs != ty).mean() - 0.25) < 0.005
    assert np.allclose(confusion_matrix(K, 0.25).sum(1), 1)


def test_grammar_count_and_hash_are_frozen():
    g = grammar(TYPES)
    assert n_features(g) == 2 * 4 * 3 * 2 == len(feature_names(g))
    assert grammar_hash(g) == grammar_hash(grammar(list(TYPES)))
    assert grammar_hash(g) != grammar_hash(grammar(TYPES + ["Tumor 9"]))


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "upmc_labels.csv").write_text(
        "region_id,status,primary_outcome,recurred,hpvstatus,survival_status,survival_day,acquisition_id_visualizer,sample_label_visualizer\n"
        "UPMC_c001_v001_r001_reg001,DOD,0,1,0,1,831,a,NA\n"
        "UPMC_c002_v001_r001_reg002,DOD,0,1,0,1,831,a,11345 A_G1\n"
        "UPMC_c001_v001_r001_reg003,NED,1,0,1,0,1952,a,HN14-7849 1A\n"
        "UPMC_c001_v001_r001_reg004,NED,1,0,1,0,1952,a,HN15-8151 2B\n")
    (tmp_path / "charville_labels.csv").write_text(
        "region_id,primary_outcome,recurrence,alive_or_deceased,type_of_first_recurrence,surgery_date,first_recurrence_date,last_contact_date,length_of_disease_free_survival,grade_differentiation,acquisition_id_visualizer,sample_label_visualizer\n"
        "Charville_c001_v001_r001_reg001,0,0,Dead,X,1/1/2010,NONE,1/11/2010,1,2,a,24219_040715\n"
        "Charville_c001_v001_r001_reg002,0,0,Dead,X,1/1/2010,NONE,1/11/2010,1,2,a,24219_040716\n"
        "Charville_c002_v001_r001_reg003,0,0,Alive,X,,NONE,,1,2,a,24136_020201\n")
    with zipfile.ZipFile(tmp_path / "upmc_raw_data.zip", "w") as z:
        for k in range(1, 5):
            r = f"UPMC_c00{1 + (k == 2)}_v001_r001_reg00{k}"
            z.writestr(f"raw_data/{r}.cell_data.csv", "CELL_ID,X,Y\n7,0,0\n9,100,0\n")
            z.writestr(f"raw_data/{r}.cell_types.csv", "CELL_ID,CELL_TYPE\n9,Tumor\n7,B cell\n")
    return tmp_path


def test_upmc_identity_endpoints_and_strata(data_dir):
    reg = cohorts.load_regions("upmc", data_dir)
    pts = cohorts.load_patients(reg)
    assert len(pts) == 2 and pts.event.tolist() == [1.0, 0.0]
    assert pts.n_coverslips.tolist() == [2, 1] and pts.stratum.tolist() == ["c001", "c001"]
    assert len(cohorts.load_patients(cohorts.load_regions("upmc", data_dir, "tuple+case"))) == 3


def test_charville_identity_and_missing_dates(data_dir):
    pts = cohorts.load_patients(cohorts.load_regions("charville", data_dir))
    assert pts.patient_id.tolist() == ["24136", "24219"]
    assert np.isnan(pts.time_days[0]) and np.isnan(pts.event[0])
    assert pts.time_days[1] == 10 and pts.event[1] == 1


def test_cells_join_on_cell_id_and_scale(data_dir):
    c = cohorts.load_cells("upmc", ["UPMC_c001_v001_r001_reg003"], data_dir)
    assert c["type_names"] == ["B cell", "Tumor"]
    assert c["cell_type"].tolist() == [0, 1] and np.allclose(c["xy_um"][1], [100 * cohorts.PIXEL_UM, 0])
    assert c["region_ptr"].tolist() == [0, 2]
