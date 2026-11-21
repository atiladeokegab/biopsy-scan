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
