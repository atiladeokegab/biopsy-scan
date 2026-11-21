"""Float64 survival-search reference statistics.

``reference.score``'s ``z`` is a randomisation-standardised Breslow score, not a
Cox Wald z. ``screen`` (R-15) is the clinical-null Cox score statistic with
nuisance-adjusted information and a clinical-null parametric bootstrap maxT; see
``stats/screen.py``.
Conditional permutation tests assume joint exchangeability of (time, event)
within strata and non-informative censoring. Design thresholds instead require
independent noisy null studies from the same design; they give unconditional
model-based calibration, never exact per-study error control.
"""

from .reference import maxT, omnibus, permutation_test, score, threshold
from .screen import permutation_maxT, screen, screen_batch

__all__ = ["maxT", "omnibus", "permutation_maxT", "permutation_test", "score", "screen", "screen_batch", "threshold"]
