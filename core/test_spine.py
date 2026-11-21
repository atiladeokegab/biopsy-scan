"""Spine checks on a tiny invented fixture (no real data, offline)."""
import zipfile

import numpy as np
import pytest

from core import cohorts
from core.grammar import feature_names, grammar, grammar_hash, n_features
from core.reimage_ref import confusion_matrix, crop_mask, draw, miscall, random_crop, reimage, reimage_region
from core.truth import neighbours, truth, truth_counts, truth_region

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
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cohorts, "LOCK_FILE", None)  # invented regions, not the real cohort: no held-out lock
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
        "Charville_c002_v001_r001_reg003,0,0,Alive,X,,NONE,,1,2,a,24136_020201\n"
        "Charville_c002_v001_r001_reg004,0,0,Alive,X,1/1/2010,NONE,1/1/2011,1,2,a,24180_1\n"
        "Charville_c003_v001_r001_reg005,0,0,Alive,X,1/1/2010,NONE,1/1/2011,1,2,a,24180_2\n")
    with zipfile.ZipFile(tmp_path / "upmc_raw_data.zip", "w") as z:
        for k in range(1, 5):
            r = f"UPMC_c00{1 + (k == 2)}_v001_r001_reg00{k}"
            z.writestr(f"raw_data/{r}.cell_data.csv", "CELL_ID,X,Y\n7,0,0\n9,100,0\n")
            z.writestr(f"raw_data/{r}.cell_types.csv", "CELL_ID,CELL_TYPE\n9,Tumor\n7,B cell\n")
    return tmp_path


def test_upmc_identity_endpoints_and_strata(data_dir):
    pts = cohorts.load_patients(cohorts.load_regions("upmc", data_dir))  # verified case numbers only
    assert pts.patient_id.tolist() == ["11345", "HN14-7849", "HN15-8151"]
    assert pts.stratum.tolist() == [0, 0, 0] and pts.event.tolist() == [1.0, 0.0, 0.0]
    tup = cohorts.load_patients(cohorts.load_regions("upmc", data_dir, "tuple"))
    assert len(tup) == 2 and tup.n_coverslips.tolist() == [2, 1]
    assert tup.stratum.tolist() == [0, 0]  # c001 and c002 are both group 0
    assert len(cohorts.load_patients(cohorts.load_regions("upmc", data_dir, "tuple+case"))) == 3
    with pytest.raises(ValueError):
        cohorts.load_regions("charville", data_dir, "tuple")


def test_charville_identity_missing_dates_and_ambiguous_stratum(data_dir):
    pts = cohorts.load_patients(cohorts.load_regions("charville", data_dir))
    assert pts.patient_id.tolist() == ["24136", "24180", "24219"]
    assert np.isnan(pts.time_days[0]) and np.isnan(pts.event[0])
    assert pts.stratum.tolist() == [1, -1, 0] and pts.usable.tolist() == [False, False, True]
    assert pts.time_days[2] == 10 and pts.event[2] == 1


def test_load_contract_arrays(data_dir):
    c = cohorts.load("upmc", data_dir)
    assert c["cell_type"].dtype == np.int8 and c["x"].dtype == np.float32 and c["region_id"].dtype == np.int32
    assert c["patient_of_region"].tolist() == [0, 1, 2] and c["region_id"].tolist() == [0, 0, 1, 1, 2, 2]
    assert c["event"].dtype == bool and c["event"].tolist() == [True, False, False]
    assert c["strata"].dtype == np.int32 and c["time"].dtype == np.float64
    assert np.isclose(c["x"][1], 100 * cohorts.PIXEL_UM)


def test_cells_join_on_cell_id_and_scale(data_dir):
    c = cohorts.load_cells("upmc", ["UPMC_c001_v001_r001_reg003"], data_dir)
    assert c["type_names"] == ["B cell", "Tumor"]
    assert c["cell_type"].tolist() == [0, 1] and np.allclose(c["xy_um"][1], [100 * cohorts.PIXEL_UM, 0])
    assert c["region_ptr"].tolist() == [0, 2]


def study_cells(seed=0):
    """Three patients over five invented regions; patient 2 has no Tumor (Ki67+) cell."""
    xs, ys, ts, rid = [], [], [], []
    for r in range(5):
        xy, ty = region(n=150, seed=seed + r)
        if r == 4:
            ty[ty == 3] = 2
        xs.append(xy[:, 0]); ys.append(xy[:, 1]); ts.append(ty); rid.append(np.full(len(ty), r))
    return {"cell_type": np.concatenate(ts).astype(np.int8), "x": np.concatenate(xs).astype(np.float32),
            "y": np.concatenate(ys).astype(np.float32), "region_id": np.concatenate(rid).astype(np.int32),
            "patient_of_region": np.array([0, 0, 1, 1, 2], np.int32), "type_names": TYPES}


FULL = {"patients": None, "regions_per_patient": None, "fov_um": None, "eps": 0.0, "seed": 0}


def test_reimage_full_field_eps0_equals_truth_with_nan():
    cells = study_cells()
    X0 = truth(cells)
    X = reimage(cells, FULL, seed=123)
    assert X.shape == (3, 2 * 4 * 3 * 2) and X.dtype == np.float64
    assert np.array_equal(X, X0, equal_nan=True)
    assert np.isnan(X0[2, 24:]).all() and np.isfinite(X0[:2]).all()  # no Ki67+ focal: NaN, not 0


def test_reimage_design_draws_are_seeded_and_change_x():
    cells = study_cells()
    design = {"patients": 5, "regions_per_patient": 1, "fov_um": 120.0, "eps": 0.1, "seed": 0}
    a, b = reimage(cells, design, 7), reimage(cells, design, 7)
    assert a.shape[0] == 5 and np.array_equal(a, b, equal_nan=True)
    assert not np.array_equal(a, reimage(cells, design, 8), equal_nan=True)
    d = draw(cells, design, 7)
    assert all(len(row) == 1 and row[0][1] is not None for row in d["looks"])
    assert len(draw(cells, dict(design, patients=[2, 0]), 7)["patients"]) == 2


def test_injected_region_kernel_gives_identical_x_in_draw_order():
    cells = study_cells()
    design = {"patients": 4, "regions_per_patient": 1, "fov_um": 150.0, "eps": 0.25, "seed": 0}
    seen = []

    def kernel(nbr, xy, types, focal, K, crop=None, M=None, u=None):
        seen.append((len(xy), crop, u[0]))
        return reimage_region(nbr, xy, types, focal, K, crop=crop, M=M, u=u)

    X = reimage(cells, design, 11, region_kernel=kernel)
    assert np.array_equal(X, reimage(cells, design, 11), equal_nan=True)
    looks = [look for row in draw(cells, design, 11)["looks"] for look in row]
    assert [(c, u[0]) for _, c, u in looks] == [(c, u0) for _, c, u0 in seen]


def test_single_type_cohort_identity_k1():
    c = {"cell_type": np.zeros(2, np.int8), "x": np.array([0, 10], np.float32),
         "y": np.zeros(2, np.float32), "region_id": np.zeros(2, np.int32),
         "patient_of_region": np.zeros(1, np.int32)}
    X0 = truth(c)
    assert np.array_equal(X0, np.ones((1, 6)))
    assert np.array_equal(reimage(c, {}, 0), X0)
    assert np.array_equal(reimage(c, {"eps": 0.3}, 0), X0)
    assert np.array_equal(confusion_matrix(1, 0.3), np.eye(1))
