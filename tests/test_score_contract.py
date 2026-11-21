"""Independent score checks; expected values follow Breslow risk sets, not a fit."""

from itertools import permutations

import numpy as np
import pytest

import stats


def breslow_residual(time, event, strata):
    """Tiny, deliberately direct oracle: tied events share the same risk set."""
    residual = np.asarray(event, dtype=np.float64).copy()
    for group in np.unique(strata):
        members = strata == group
        for death_time in np.unique(time[members & event]):
            risk = members & (time >= death_time)
            deaths = np.count_nonzero(members & event & (time == death_time))
            residual[risk] -= deaths / np.count_nonzero(risk)
    return residual


@pytest.fixture
def adverse_score_case():
    # Tied deaths, a censor at an event time, an event-free stratum, and a singleton.
    return (
        np.array([[0, 7], [1, 7], [2, 7], [3, 7], [10, 7], [12, 7],
                  [100, 7], [-100, 7], [900, 7]], dtype=np.float64),
        np.array([1, 1, 2, 3, 1, 2, 1, 2, 1], dtype=np.float64),
        np.array([1, 1, 0, 1, 1, 0, 0, 0, 1], dtype=bool),
        np.array([5, 5, 5, 5, 11, 11, 20, 20, 31], dtype=np.int32),
    )


def test_ties_censoring_and_degenerate_strata(adverse_score_case):
    """Breslow U and exact randomisation V stay finite for uninformative strata."""
    result = stats.score(*adverse_score_case)
    np.testing.assert_allclose(result["U"], [-3, 0], atol=1e-12)
    np.testing.assert_allclose(result["V"], [8 / 3, 0], atol=1e-12)
    np.testing.assert_allclose(result["z"], [-3 / np.sqrt(8 / 3), 0], atol=1e-12)
    for key in ("U", "V", "z"):
        assert np.asarray(result[key]).dtype == np.float64


def test_variance_is_exhaustive_permutation_variance(adverse_score_case):
    """V is the population variance of U over every within-stratum permutation."""
    X, time, event, strata = adverse_score_case
    residual = breslow_residual(time, event, strata)
    scores = []
    for first in permutations(range(4)):
        for second in permutations([4, 5]):
            order = [*first, *second, 6, 7, 8]
            scores.append(X[order].T @ residual)
    np.testing.assert_allclose(np.mean(scores, axis=0), [0, 0], atol=1e-12)
    np.testing.assert_allclose(stats.score(X, time, event, strata)["V"],
                               np.var(scores, axis=0), atol=1e-12)


def test_batch_and_patient_order_preserve_scores(adverse_score_case):
    """The batch axis and arbitrary patient order cannot change the statistic."""
    X, time, event, strata = adverse_score_case
    batch = np.stack([X, -X, X * 2])
    result = stats.score(batch, time, event, strata)
    order = np.array([8, 2, 5, 0, 7, 1, 4, 6, 3])
    shuffled = stats.score(batch[:, order], time[order], event[order], strata[order])
    for key in ("U", "V", "z"):
        assert np.asarray(result[key]).shape == (3, 2)
        np.testing.assert_allclose(shuffled[key], result[key], atol=1e-12)
        for i in range(3):
            single = stats.score(batch[i], time, event, strata)
            np.testing.assert_allclose(result[key][i], single[key], atol=1e-12)
    np.testing.assert_allclose(result["z"][1], -result["z"][0], atol=1e-12)
    np.testing.assert_allclose(result["z"][2], result["z"][0], atol=1e-12)


def test_every_patient_censored_has_no_evidence():
    """An entirely event-free cohort produces zeros, never NaN or a discovery."""
    X = np.arange(24, dtype=np.float64).reshape(8, 3)
    result = stats.score(X, np.arange(1, 9, dtype=np.float64),
                         np.zeros(8, dtype=bool), np.zeros(8, dtype=np.int32))
    for key in ("U", "V", "z"):
        np.testing.assert_array_equal(result[key], np.zeros(3))


def test_planted_ordered_signal_is_found():
    """A strong planted within-group survival ordering dominates irrelevant columns."""
    rng = np.random.default_rng(731)
    time = np.tile(np.arange(1, 41), 2).astype(np.float64)
    event = np.ones(80, dtype=bool)
    strata = np.repeat([3, 9], 40).astype(np.int32)
    X = np.column_stack([-time, rng.normal(size=(80, 2))])
    result = stats.score(X, time, event, strata)
    assert result["z"][0] > 6
    assert np.argmax(np.abs(result["z"])) == 0
    np.testing.assert_allclose(stats.maxT(result["z"]), abs(result["z"][0]))


def test_batch_confounding_is_a_hard_negative():
    """Huge between-group survival differences alone must not become a spatial hit."""
    time = np.r_[np.arange(1, 21), np.arange(101, 121)].astype(np.float64)
    event = np.ones(40, dtype=bool)
    strata = np.repeat([7, 19], 20).astype(np.int32)
    X = np.repeat([[0.0, 900.0], [1000.0, -900.0]], 20, axis=0)
    pooled = stats.score(X, time, event, np.zeros(40, dtype=np.int32))
    assert stats.maxT(pooled["z"]) > 4  # The trap is strongly associated when pooled.
    stratified = stats.score(X, time, event, strata)
    np.testing.assert_allclose(stratified["z"], [0, 0], atol=1e-12)


def test_maxT_reduces_candidates_and_keeps_studies():
    """The multiplicity statistic uses both tails and never mixes study batches."""
    z = np.array([[1.0, -8.0, 2.0], [-3.0, 0.0, 2.0]])
    np.testing.assert_array_equal(stats.maxT(z), [8, 3])


def test_null_threshold_uses_independent_maxima():
    """The design threshold uses the max absolute candidate from independent nulls."""
    null_z = np.column_stack([np.arange(1, 101), -2 * np.arange(1, 101)]).astype(float)
    design = dict(patients=80, regions_per_patient=2, fov_um=100, eps=0, seed=731)
    threshold = stats.threshold(design, null_z)
    assert 190 <= threshold <= 192
    assert np.isscalar(threshold) and np.isfinite(threshold)
