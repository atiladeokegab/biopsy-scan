"""Event-count calibration of the frozen design-race statistic; evaluation streams stay separate."""

import math

import numpy as np

from stats.reference import threshold

ALPHA = 0.05
SPLIT = 38  # predeclared from 159 * the source event fraction (38/159), before evaluating calls


def fit_thresholds(calibration, mode="two_bins"):
    """Initial unconditional baseline; the regression specifies the event-conditioned rule."""
    cut = threshold({"alpha": ALPHA}, np.array([[r["maxT"]] for r in calibration]))
    return {"mode": mode, "bins": {"low": {"n": len(calibration), "threshold": cut},
                                    "high": {"n": len(calibration), "threshold": cut}}}
