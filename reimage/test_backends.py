"""Native-boundary regressions and reference agreement; GPU cases need CUDA."""

import numpy as np
import pytest
from numba import cuda

from core import reimage_ref
from core.truth import neighbours
from reimage import run
from reimage import cpu, gpu


@pytest.fixture
def region_case():
    xy = np.array([[0, 0], [0, 0], [25, 0], [50, 0], [100, 0], [110, 0]], dtype=np.float64)
    types = np.array([0, 1, 2, 1, 0, 2], dtype=np.int16)
    return neighbours(xy), xy, types


@pytest.mark.parametrize("backend", [cpu, gpu])
@pytest.mark.parametrize("crop,eps", [(None, 0), ((0, 0, 50), 0.25), ((500, 500, 10), 1)])
def test_region_matches_reference(region_case, backend, crop, eps):
    """Shared acquisitions agree exactly, including truth reproduction and empty crops."""
    if backend is gpu and not cuda.is_available():
        pytest.skip("no CUDA device; CUDA simulator is checked in a separate process")
    nbr, xy, types = region_case
    u = np.random.default_rng(397).random(len(types))
    args = (nbr, xy, types, [0, 2], 3)
    expected = reimage_ref.reimage_region(*args, crop=crop, eps=eps, u=u)
    actual = backend.reimage_region(*args, crop=crop, eps=eps, u=u)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("bad", ["label", "index", "offset", "mask", "K"])
def test_invalid_native_indices_fail_before_counting(region_case, bad):
    """Malformed arrays cannot reach unchecked Numba/CUDA indexing."""
    nbr, _, types = region_case
    nbr = {key: value.copy() for key, value in nbr.items()}
    visible, K = np.ones(len(types), dtype=bool), 3
    if bad == "label":
        types = types.copy()
        types[0] = -1
    elif bad == "index":
        nbr["idx"][0] = len(types)
    elif bad == "offset":
        nbr["rad_end"][0, 0] = nbr["ptr"][1] + 1
    elif bad == "mask":
        visible = visible[:-1]
    else:
        K = True
    for backend in (cpu, gpu):
        with pytest.raises(ValueError):
            backend.reimage_counts(nbr, types, visible, K)


@pytest.mark.parametrize("backend", [cpu, gpu])
def test_prepared_adjacency_reuses_only_topology(region_case, backend):
    """A reused upload still consumes each acquisition's new labels and crop mask."""
    if backend is gpu and not cuda.is_available():
        pytest.skip("no CUDA device; CUDA simulator is checked in a separate process")
    nbr, _, types = region_case
    prepared = backend.prepare(nbr, 3)
    for offset in (0, 1):
        observed = (types + offset) % 3
        visible = np.arange(len(types)) >= offset
        expected = reimage_ref.reimage_counts(nbr, observed, visible, 3)
        np.testing.assert_array_equal(prepared.reimage_counts(observed, visible), expected)


def test_prepared_region_rejects_other_topology(region_case):
    """Equal-sized regions must not silently share the wrong cached adjacency."""
    nbr, xy, types = region_case
    other = neighbours(xy * 10)
    prepared = cpu.prepare(nbr, 3)
    with pytest.raises(ValueError, match="different region"):
        cpu.reimage_region(other, xy * 10, types, [0], 3, prepared=prepared)


def test_gpu_selection_never_reports_cpu_fallback(region_case):
    """Requesting GPU without a device must fail explicitly rather than fake a GPU rung."""
    if cuda.is_available():
        pytest.skip("this check requires CUDA to be unavailable")
    nbr, _, types = region_case
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        gpu.reimage_counts(nbr, types, np.ones(len(types), dtype=bool), 3)


@pytest.mark.parametrize("backend", ["numpy", "cpu", "gpu"])
@pytest.mark.parametrize("patients,k,fov,eps", [(None, None, None, 0), ([1, 0, 1], 1, 40, 0.25)])
def test_study_batch_matches_shared_draws(region_case, backend, patients, k, fov, eps):
    """Patient order, region cap, crop and mis-call draws are identical across backends."""
    if backend == "gpu" and not cuda.is_available():
        pytest.skip("no CUDA device; CUDA simulator is checked in a separate process")
    _, xy, types = region_case
    cells = dict(
        x=np.tile(xy[:, 0], 4).astype(np.float32),
        y=np.tile(xy[:, 1], 4).astype(np.float32),
        cell_type=np.tile(types, 4).astype(np.int8),
        region_id=np.repeat(np.arange(4), len(types)).astype(np.int32),
        patient_of_region=np.array([0, 0, 1, 1], dtype=np.int32),
        type_names=["Tumor A", "Other B", "Other C"],
    )
    design = dict(cells=cells, backend=backend, patients=patients, regions_per_patient=k,
                  fov_um=fov, eps=eps, seed=37)
    seeds = [37, 61]
    expected = np.stack([reimage_ref.reimage(cells, design, seed) for seed in seeds])
    actual = run(design, seeds)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("seeds", [[], [1.5], [-1], [[1]], [True]])
def test_invalid_study_seeds_fail_before_sampling(seeds):
    """Invalid seed batches cannot consume the shared study random stream."""
    with pytest.raises(ValueError, match="seeds"):
        run({}, seeds)
