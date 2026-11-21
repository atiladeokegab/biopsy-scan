"""Reference checks kept in stats/ because issue #4 owns only this area."""

from itertools import permutations, product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from statsmodels.duration.hazard_regression import PHReg

from stats import maxT, omnibus, permutation_test, score, threshold


@pytest.fixture
def tiny():
    return (
        np.array([[0, 0], [1, 3], [4, 1], [2, 0], [1, 2], [4, 0]], dtype=float),
        np.array([1, 1, 1, 2, 1, 2], dtype=float),
        np.array([1, 1, 0, 1, 1, 0]),
        np.array([0, 0, 0, 0, 1, 1]),
    )


@pytest.mark.parametrize("seed", range(4))
def test_score_matches_phreg_breslow_with_ties(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(30, 4))
    time = rng.integers(0, 7, 30).astype(float)
    event = rng.integers(0, 2, 30)
    strata = np.repeat([11, 27, 99], 10)
    expected = PHReg(time, X, status=event, strata=strata, ties="breslow").score(np.zeros(4))
    assert_allclose(score(X, time, event, strata)["U"], expected, atol=2e-14)


def test_hand_computed_ties_and_joint_maxT(tiny):
    result = permutation_test(*tiny, exact=True)
    assert_allclose(result["U"], [-4, 2])
    assert_allclose(result["V"], [31 / 6, 3])
    assert result["maxT"] == pytest.approx(4 / np.sqrt(31 / 6))
    assert result["p"] == pytest.approx(1 / 6)
    assert_allclose(result["adjusted_p"], [1 / 6, 2 / 3])
    assert result["null_z"].shape == (48, 2)


def test_variance_and_maxT_against_outcome_pair_enumeration(tiny):
    X, time, event, strata = tiny
    # Independent oracle: build beta=0 scores directly from event risk sets.
    U = []
    for left, right in product(permutations(range(4)), permutations(range(4, 6))):
        order = np.array(left + right)
        t, e = time[order], event[order]
        u = np.zeros(X.shape[1])
        for s in np.unique(strata):
            ix = strata == s
            for value in np.unique(t[ix & (e == 1)]):
                deaths = ix & (t == value) & (e == 1)
                risk = ix & (t >= value)
                u += X[deaths].sum(axis=0) - deaths.sum() * X[risk].mean(axis=0)
        U.append(u)
    U = np.array(U)
    result = permutation_test(*tiny, exact=True)
    assert_allclose(U.mean(axis=0), 0, atol=1e-15)
    assert_allclose(U.var(axis=0, ddof=0), result["V"], atol=2e-15)
    expected_max = np.max(np.abs(U / np.sqrt(U.var(axis=0))), axis=1)
    assert_allclose(maxT(result["null_z"]), expected_max, atol=2e-15)


def test_batch_and_patient_order_invariance(tiny):
    X, time, event, strata = tiny
    batch = np.stack([X, 2 * X + 100, -X])
    result = score(batch, time, event, strata)
    assert_allclose(result["z"], np.stack([score(x, time, event, strata)["z"] for x in batch]))
    assert result["U"].dtype == result["V"].dtype == result["z"].dtype == np.float64
    order = np.array([5, 2, 0, 4, 1, 3])
    shuffled = score(batch[:, order], time[order], event[order], strata[order])
    for key in result:
        assert_allclose(shuffled[key], result[key], atol=2e-15)


def test_event_free_singleton_and_constant_columns(tiny):
    X, time, event, strata = tiny
    X = np.vstack([X, [[100, -100], [1000, -300], [400, 200]]])
    time = np.r_[time, [1, 3, 0]]
    event = np.r_[event, [0, 0, 1]]
    strata = np.r_[strata, [2, 2, 3]]
    expected = score(*tiny)
    actual = score(X, time, event, strata)
    for key in expected:
        assert_allclose(actual[key], expected[key])
    X = np.column_stack([X, np.full(len(X), 0.1), strata * 0.1])
    result = score(X, time, event, strata)
    for value in result.values():
        assert_allclose(value[-2:], 0, atol=0)
    for value in score(X, time, np.zeros(len(X)), strata).values():
        assert_allclose(value, 0, atol=0)


def test_monte_carlo_reproducible_and_plus_one(tiny):
    result = permutation_test(*tiny, n_resamples=41, seed=7)
    other = permutation_test(*tiny, n_resamples=41, seed=7)
    assert_allclose(result["null_z"], other["null_z"])
    expected = (1 + np.sum(maxT(result["null_z"]) >= result["maxT"] - 1e-14)) / 42
    assert result["p"] == pytest.approx(expected)
    assert result["omnibus"]["p"] >= 1 / 42


def test_unconditional_threshold_finite_calibration_and_strict_ties():
    design = {"patients": 6, "regions_per_patient": 2, "fov_um": 250, "eps": 0.1, "seed": 8}
    z = np.arange(1, 20, dtype=float)[:, None]
    assert threshold(design, z) == 19
    assert threshold(design, {"z": z}) == 19
    assert np.isinf(threshold(design, z[:-1]))
    # Every possible held-out study among ten exchangeable rank values.
    maxima = np.arange(10, dtype=float)
    decisions = [value > threshold({"alpha": 0.2}, np.delete(maxima, i)[:, None])
                 for i, value in enumerate(maxima)]
    assert np.mean(decisions) <= 0.2
    assert threshold({"alpha": 0.2}, np.ones((9, 2))) == 1


def test_omnibus_calibrates_choice_jointly(tiny):
    rows = permutation_test(*tiny, exact=True)["null_z"]
    p = []
    for i, row in enumerate(rows):
        result = omnibus(row, np.delete(rows, i, axis=0))
        assert set(result["statistics"]) == {"maxT", "sum_z2", "HC"}
        assert result["p"] >= min(result["component_p"].values())
        p.append(result["p"])
    for alpha in [0.05, 0.1, 0.25, 0.5]:
        assert np.mean(np.array(p) <= alpha) <= alpha
    result = omnibus(np.zeros(3), np.zeros((19, 3)))
    assert result["p"] == 1
    assert result["statistics"] == {"maxT": 0, "sum_z2": 0, "HC": 0}


def test_exact_omnibus_preserves_mathematical_ties_after_row_reorder():
    X = np.array([[1, 4, 2], [3, 2, 1], [3, 4, 1], [3, 1, 3], [3, 2, 3]])
    time, event, strata = np.array([1, 0, 0, 2, 4]), np.array([0, 0, 1, 0, 1]), np.zeros(5)
    result = permutation_test(X, time, event, strata, exact=True)
    reverse = permutation_test(X[::-1], time[::-1], event[::-1], strata, exact=True)
    assert result["omnibus"]["p"] == reverse["omnibus"]["p"]
    assert result["omnibus"]["component_p"] == reverse["omnibus"]["component_p"]


def test_event_free_stratum_never_squares_uninformative_features():
    for value in score([[1e155], [-1e155]], [1, 2], [0, 0], [0, 0]).values():
        assert_allclose(value, 0, atol=0)


@pytest.mark.parametrize("field,value", [
    ("X", [[np.nan]]), ("X", [[np.inf]]), ("X", []),
    ("time", [-1]), ("time", [np.nan]), ("time", [1, 2]),
    ("event", [2]), ("event", [np.nan]), ("strata", [np.nan]),
])
def test_rejects_invalid_score_inputs(field, value):
    args = {"X": [[1]], "time": [1], "event": [1], "strata": [0]}
    args[field] = value
    with pytest.raises(ValueError):
        score(**args)


def test_rejects_invalid_calibration_and_unbounded_enumeration(tiny):
    for z in [[], [np.nan], [[np.inf]]]:
        with pytest.raises(ValueError):
            maxT(z)
    for alpha in [0, 1, np.nan]:
        with pytest.raises(ValueError):
            threshold({"alpha": alpha}, np.ones((20, 2)))
    with pytest.raises(ValueError):
        threshold({}, np.ones(20))
    with pytest.raises(ValueError):
        permutation_test(*tiny, exact=True, max_exact=47)
    with pytest.raises(ValueError):
        permutation_test(*tiny, n_resamples=0)
    with pytest.raises(ValueError):
        omnibus([0, 1], np.zeros((3, 1)))
