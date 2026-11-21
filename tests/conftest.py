"""Invented tissue only: smoke never reads the cancer cohorts or needs a GPU."""

import numpy as np
import pytest


@pytest.fixture
def tiny_cells():
    """216 cells, three patients, two regions each, three interleaved types."""
    grid = np.arange(36)
    return {
        "cell_type": np.tile(grid % 3, 6).astype(np.int8),
        "x": np.tile((grid % 6) * 10, 6).astype(np.float32),
        "y": np.tile((grid // 6) * 10, 6).astype(np.float32),
        "region_id": np.repeat(np.arange(6), 36).astype(np.int32),
        "patient_of_region": np.repeat(np.arange(3), 2).astype(np.int32),
    }
