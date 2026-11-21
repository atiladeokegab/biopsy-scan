"""Float64 survival-search reference statistics.

``z`` is a randomisation-standardised Breslow score, not a Cox Wald z.
Conditional permutation tests assume joint exchangeability of (time, event)
within strata and non-informative censoring. Design thresholds instead require
independent noisy null studies from the same design; they give unconditional
model-based calibration, never exact per-study error control.
"""

from .reference import maxT, omnibus, permutation_test, score, threshold

__all__ = ["maxT", "omnibus", "permutation_test", "score", "threshold"]
