"""The fused region kernel gives X bit-identical (sha256) to the CSR Numba path it replaces."""

import hashlib

import numpy as np
import pytest

from bench.fleet import synthetic_cohort
from core import reimage_ref
from core.reimage_ref import reimage
from reimage import cpu, fused


def _sha(X):
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()


def _kernel(module, cells, K):
    prepared = {id(nbr): module.prepare(nbr, K) for nbr in cells["_nbr"].values()}
    return lambda nbr, xy, types, focal, K, **kw: module.reimage_region(nbr, xy, types, focal, K,
                                                                       prepared=prepared[id(nbr)], **kw)


def test_two_hundred_studies_give_bit_identical_X():
    cells, _ = synthetic_cohort(n_patients=12, regions_per_patient=2, cells=900, side=300.0, seed=7)
    K = len(cells["type_names"])
    old, new = _kernel(cpu, cells, K), _kernel(fused, cells, K)
    design = dict(patients=10, regions_per_patient=2, fov_um=150.0, eps=0.1)
    diff = [s for s in range(200) if _sha(reimage(cells, design, s, region_kernel=old))
            != _sha(reimage(cells, design, s, region_kernel=new))]
    assert diff == []


@pytest.mark.parametrize("crop,eps,focal", [(None, 0, [0, 2]), ((0, 0, 50), 0.25, [0, 2]),
                                            ((500, 500, 10), 1, [0, 2]), (None, 0.5, [1])])
def test_region_matches_reference_exactly(crop, eps, focal):
    xy = np.array([[0, 0], [0, 0], [25, 0], [50, 0], [100, 0], [110, 0], [12, 30]], dtype=np.float64)
    types = np.array([0, 1, 2, 1, 0, 2, 0], dtype=np.int16)
    nbr, u = reimage_ref.neighbours(xy), np.random.default_rng(397).random(len(types))
    expected = reimage_ref.reimage_region(nbr, xy, types, focal, 3, crop=crop, eps=eps, u=u)
    actual = fused.reimage_region(nbr, xy, types, focal, 3, crop=crop, eps=eps, u=u)
    assert expected[0].tobytes() == actual[0].tobytes() and np.array_equal(expected[1], actual[1])


def test_bad_inputs_are_refused_before_the_kernel():
    xy = np.zeros((3, 2))
    nbr = reimage_ref.neighbours(xy)
    with pytest.raises(ValueError):
        fused.reimage_region(nbr, xy, np.array([0, 1, 5]), [0], 3, u=np.zeros(3))
    with pytest.raises(ValueError):
        fused.reimage_region(nbr, xy, np.array([0, 1, 2]), [0], 3, u=np.ones(3))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("x0", [1 + 1e-8, 1.0, 1 - 1e-8, np.nextafter(1.0, 2.0)])
def test_crop_boundaries_select_the_cells_crop_mask_selects(dtype, x0):
    """Review #120: float32 xy compared against a float64 crop edge, at and around the boundary."""
    xy = np.array([[1, 0], [1, 0], [2, 0]], dtype=dtype)
    types, u = np.array([0, 1, 0]), np.zeros(3)
    nbr = reimage_ref.neighbours(xy)
    for crop in ((x0, 0, 1), (0, 0, x0)):
        expected = cpu.reimage_region(nbr, xy, types, [0, 1], 2, crop=crop, u=u)
        actual = fused.reimage_region(nbr, xy, types, [0, 1], 2, crop=crop, u=u)
        assert expected[0].tobytes() == actual[0].tobytes() and np.array_equal(expected[1], actual[1])


def test_adjacency_of_another_size_is_refused_with_or_without_preparation():
    small = reimage_ref.neighbours(np.zeros((1, 2)))
    xy, types, u = np.zeros((2, 2)), np.array([0, 1]), np.zeros(2)
    with pytest.raises(ValueError):
        fused.reimage_region(small, xy, types, [0, 1], 2, u=u)
    with pytest.raises(ValueError):
        fused.reimage_region(small, xy, types, [0, 1], 2, u=u, prepared=fused.prepare(small, 2))
