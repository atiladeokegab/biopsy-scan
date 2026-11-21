"""Weibull PH outcomes, source-scaled effects and reverse-KM censoring."""

import numpy as np
from scipy.optimize import brentq


def reverse_km(time, event):
    """Censoring atoms and probabilities, retaining unresolved mass at +infinity.

    Original failures precede censors at tied times, matching event=(T<=C).
    Reference: scikit-survival's kaplan_meier_estimator(reverse=True):
    https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.nonparametric.kaplan_meier_estimator.html
    """
    time, event = np.asarray(time, float), np.asarray(event)
    if time.ndim != 1 or not len(time) or event.shape != time.shape:
        raise ValueError("time/event must be nonempty aligned vectors")
    if not np.isfinite(time).all() or (time < 0).any() or not np.isin(event, [0, 1]).all():
        raise ValueError("invalid source times or event flags")
    atoms, mass, remaining = [], [], 1.0
    for t in np.unique(time):
        at_time = time == t
        deaths = int(np.sum(event[at_time]))
        censored = int(at_time.sum()) - deaths
        if censored:
            hazard = censored / (int(np.sum(time >= t)) - deaths)
            atoms.append(float(t))
            mass.append(remaining * hazard)
            remaining *= 1 - hazard
    return np.array([*atoms, np.inf]), np.array([*mass, remaining])


def _event_probability(log_scale, eta, strata, censoring, shape):
    p = np.zeros(len(eta))
    for group, distribution in censoring.items():
        ix = strata == group
        if not ix.any():
            continue
        c, weights = distribution["time"], distribution["mass"]
        with np.errstate(divide="ignore"):
            log_hazard = eta[ix, None] + shape * (np.log(c)[None, :] - log_scale)
        # Once log hazard is 7 the event CDF is indistinguishable from one.
        cdf = -np.expm1(-np.exp(np.minimum(log_hazard, 7)))
        p[ix] = cdf @ weights
    return float(p.mean())


def fit_model(cells, reference, *, feature=None, hr=2.0, shape=1.5, target=None):
    """Freeze a source estimand and tune expected event fraction, once per effect.

    The plant is chosen without using outcomes: first finite, nonconstant feature
    with within-stratum variation. Its mean and population SD (ddof=0) are frozen
    across sample sizes and imaging designs. No absent focal value is imputed.

    Censoring uses stratum-specific reverse KM. Unresolved tail mass is placed at
    the stratum's maximum observed follow-up (an explicit administrative-horizon
    assumption, not an identified censoring tail). Weibull scale solves the exact
    expectation over source patients and these censoring atoms, not observed
    simulated events. Null and alternative models each hit the same event target.
    """
    X = np.asarray(reference, np.float64)
    time, event, strata = (np.asarray(cells[k]) for k in ("time", "event", "strata"))
    if X.ndim != 2 or not all(X.shape) or time.shape != (len(X),) or event.shape != time.shape or strata.shape != time.shape:
        raise ValueError("reference and source outcomes must have aligned patient rows")
    reverse_km(time, event)  # Validate endpoints even before grouping.
    if not np.isfinite(strata).all() or (strata < 0).any():
        raise ValueError("source strata must be finite and unambiguous")
    if not np.isfinite([hr, shape]).all() or hr <= 0 or shape <= 0:
        raise ValueError("hr and Weibull shape must be positive and finite")
    candidates = np.flatnonzero(np.isfinite(X).all(axis=0))
    if feature is not None:
        if not isinstance(feature, (int, np.integer)) or not 0 <= feature < X.shape[1]:
            raise ValueError("feature must be a valid integer column index")
        candidates = np.array([feature])
    selected = None
    for j in candidates:
        x = X[:, j]
        if not np.isfinite(x).all():
            continue
        within = any(np.ptp(x[strata == s]) > 0 for s in np.unique(strata))
        if np.ptp(x) > 0 and within:
            selected = int(j)
            break
    if selected is None:
        raise ValueError("plant needs a finite source feature with within-stratum variation")
    x = X[:, selected]
    mu, sd = float(x.mean()), float(x.std(ddof=0))
    standardized = (x - mu) / sd
    eta = np.log(hr) * standardized
    target = float(event.mean()) if target is None else float(target)
    if not np.isfinite(target) or not 0 < target < 1:
        raise ValueError("target event fraction must lie strictly between zero and one")
    censoring = {}
    for s in np.unique(strata):
        ix = strata == s
        atoms, mass = reverse_km(time[ix], event[ix])
        tau = float(time[ix].max())
        censoring[int(s)] = {"time": np.minimum(atoms, tau), "mass": mass,
                             "horizon": tau, "unresolved_tail_mass": float(mass[-1])}
    positive = time[time > 0]
    center = float(np.log(np.median(positive))) if len(positive) else 0.0
    lo, hi = center - 32 / shape, center + 32 / shape
    def probability(value):
        return _event_probability(value, eta, strata, censoring, shape)
    if probability(lo) < target or probability(hi) > target:
        raise ValueError("target event fraction is outside the declared censoring model's support")
    log_scale = brentq(lambda value: probability(value) - target, lo, hi, xtol=1e-12)
    return {"feature": selected, "source_mean": mu, "source_sd": sd,
            "source_z": standardized, "strata": strata, "hr": float(hr), "shape": float(shape),
            "log_scale": float(log_scale), "target_event_fraction": target,
            "expected_event_fraction": probability(log_scale), "censoring": censoring,
            "censoring_tail_policy": "administrative horizon = stratum maximum observed follow-up"}


def outcomes(model, patients, survival_seed, censor_seed):
    """Generate aligned observed time/event vectors for resampled source indices."""
    patients = np.asarray(patients)
    if patients.ndim != 1 or not len(patients) or patients.dtype.kind not in "iu":
        raise ValueError("patients must be a nonempty integer index vector")
    if (patients < 0).any() or (patients >= len(model["source_z"])).any():
        raise ValueError("patient index is outside the source population")
    rng_t = np.random.default_rng(survival_seed)
    rng_c = np.random.default_rng(censor_seed)
    eta = np.log(model["hr"]) * model["source_z"][patients]
    log_t = model["log_scale"] + (np.log(rng_t.exponential(size=len(patients))) - eta) / model["shape"]
    strata = model["strata"][patients]
    censor = np.empty(len(patients))
    for group, distribution in model["censoring"].items():
        ix = strata == group
        censor[ix] = rng_c.choice(distribution["time"], size=int(ix.sum()), p=distribution["mass"])
    with np.errstate(divide="ignore"):
        log_c = np.log(censor)
    event = log_t <= log_c
    return np.exp(np.minimum(log_t, log_c)), event, strata.copy()
