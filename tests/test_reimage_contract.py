"""Direct geometry oracles; opt in accelerated count hooks with --reimage-counts."""

from importlib import import_module

import numpy as np
import pytest

from core.grammar import RADII_UM
from core.reimage_ref import crop_mask, miscall, reimage_region
from core.truth import neighbours


def direct_counts(xy, observed, visible, K):
    """Count ordered pairs directly; never consume the implementation's adjacency."""
    out = np.zeros((len(xy), len(RADII_UM), K), dtype=np.int64)
    for i in range(len(xy)):
        for j in range(len(xy)):
            if i == j or not (visible[i] and visible[j]):
                continue
            distance = np.sqrt(np.sum((xy[i].astype(float) - xy[j]) ** 2))
            for r, radius in enumerate(RADII_UM):
                if distance <= radius:
                    out[i, r, observed[j]] += 1
    return out


def direct_region(xy, observed, visible, focal, K):
    """Literal focal-cell means, with visible-cell composition for enrichment."""
    counts = direct_counts(xy, observed, visible, K)
    S = np.zeros((len(focal), K, len(RADII_UM), 2), dtype=np.float64)
    m = np.zeros(len(focal), dtype=np.int64)
    if not visible.any():
        return S, m
    for t, kind in enumerate(focal):
        for i in np.flatnonzero(visible & (observed == kind)):
            m[t] += 1
            for r in range(len(RADII_UM)):
                total = counts[i, r].sum()
                for j in range(K):
                    fraction = counts[i, r, j] / total if total else 0
                    share = np.mean(observed[visible] == j)
                    S[t, j, r, 0] += fraction
                    S[t, j, r, 1] += fraction / share if share else 0
    return S, m


@pytest.mark.parametrize("region", range(5))
@pytest.mark.parametrize("crop", [None, (0, 0, 25), (10, 10, 35)])
@pytest.mark.parametrize("seed", [317, 911])
def test_cell_counts_match_direct_geometry(tiny_cells, count_kernel, region, crop, seed):
    """Five regions × three crops × two label draws agree cell for cell."""
    take = tiny_cells["region_id"] == region
    xy = np.column_stack([tiny_cells["x"][take], tiny_cells["y"][take]])
    original = tiny_cells["cell_type"][take]
    rng = np.random.default_rng(seed)
    observed = (original + rng.integers(0, 3, len(original))).astype(np.int8) % 3
    visible = np.ones(len(xy), dtype=bool) if crop is None else (
        (xy[:, 0] >= crop[0]) & (xy[:, 0] < crop[0] + crop[2])
        & (xy[:, 1] >= crop[1]) & (xy[:, 1] < crop[1] + crop[2])
    )
    module, name = count_kernel.split(":")
    kernel = getattr(import_module(module), name)
    actual = kernel(neighbours(xy), observed, visible, 3)
    np.testing.assert_array_equal(actual, direct_counts(xy, observed, visible, 3))
    assert np.asarray(actual).dtype.kind in "iu"


@pytest.mark.parametrize("crop", [None, (0, 0, 25), (500, 500, 10)])
def test_region_sums_match_focal_cell_oracle(tiny_cells, crop):
    """Full, cropped and empty fields use visible neighbours and focal-cell weighting."""
    take = tiny_cells["region_id"] == 0
    xy = np.column_stack([tiny_cells["x"][take], tiny_cells["y"][take]])
    types = tiny_cells["cell_type"][take]
    visible = crop_mask(xy, crop)
    expected = direct_region(xy, types, visible, [0, 2], 3)
    actual = reimage_region(neighbours(xy), xy, types, [0, 2], 3,
                            crop=crop, eps=0, u=np.zeros(len(types)))
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)
    np.testing.assert_array_equal(actual[1], expected[1])


def test_focal_cells_are_relabelled_too():
    """At 100% binary mis-call the old neighbours become the new focal cells."""
    xy = np.array([[0, 0], [10, 0], [20, 0]], dtype=np.float32)
    types = np.array([0, 1, 1], dtype=np.int8)
    actual = reimage_region(neighbours(xy), xy, types, [0], 2,
                            eps=1, u=np.array([0.2, 0.5, 0.8]))
    expected = direct_region(xy, 1 - types, np.ones(3, dtype=bool), [0], 2)
    np.testing.assert_array_equal(actual[1], [2])
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-12)


def test_crop_is_half_open():
    """The lower boundary is visible; cells on the upper x or y edge are outside."""
    xy = np.array([[0, 0], [25, 0], [0, 25], [24, 24], [-1, 0]], dtype=float)
    np.testing.assert_array_equal(crop_mask(xy, (0, 0, 25)), [1, 0, 0, 1, 0])


def test_radius_includes_boundary_but_excludes_self(count_kernel):
    """Coincident distinct cells count; self does not; each radius is inclusive."""
    xs = [0, 0, 25, 50, 100, np.nextafter(np.float32(100), np.float32(np.inf))]
    xy = np.column_stack([xs, np.zeros(6)]).astype(np.float32)
    observed = np.array([0, 1, 1, 1, 1, 1], dtype=np.int8)
    module, name = count_kernel.split(":")
    kernel = getattr(import_module(module), name)
    actual = kernel(neighbours(xy), observed, np.ones(6, dtype=bool), 2)
    np.testing.assert_array_equal(actual[0, :, 1], [2, 3, 4])
    np.testing.assert_array_equal(actual, direct_counts(xy, observed, np.ones(6, dtype=bool), 2))


def test_miscall_cdf_boundaries():
    """Shared per-cell draws choose the next type exactly at a CDF boundary."""
    M = np.array([[0.25, 0.5, 0.25], [1, 0, 0], [0, 0, 1]])
    observed = miscall(np.array([0, 0, 0, 0, 1, 2], dtype=np.int8), M,
                       np.array([0, 0.25, 0.749, 0.75, 0.9, 0.1]))
    np.testing.assert_array_equal(observed, [0, 1, 1, 2, 0, 2])
