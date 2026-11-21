"""Invented tissue only: smoke never reads the cancer cohorts or needs a GPU."""

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--reimage-counts", action="append", default=[], metavar="MODULE:FUNCTION",
        help="Also check a backend's (nbr, observed_types, visible, K) cell counts.",
    )


def pytest_generate_tests(metafunc):
    if "count_kernel" in metafunc.fixturenames:
        kernels = ["core.reimage_ref:reimage_counts"]
        kernels.extend(metafunc.config.getoption("--reimage-counts"))
        metafunc.parametrize("count_kernel", kernels)


@pytest.fixture
def tiny_cells():
    """216 cells, three patients, six different-density regions and three types."""
    grid = np.arange(36)
    spacing = np.repeat(np.arange(10, 40, 5), 36)
    return {
        "cell_type": ((np.tile(grid % 3, 6) + np.repeat(np.arange(6), 36)) % 3).astype(np.int8),
        "x": (np.tile(grid % 6, 6) * spacing).astype(np.float32),
        "y": (np.tile(grid // 6, 6) * spacing).astype(np.float32),
        "region_id": np.repeat(np.arange(6), 36).astype(np.int32),
        "patient_of_region": np.repeat(np.arange(3), 2).astype(np.int32),
    }
