import numpy as np
import pytest
from numpy.testing import assert_allclose

from atlas.simulation import fit_model, outcomes, reverse_km


def source():
    return {"time": np.full(4, 5.0), "event": np.array([1, 0, 0, 0]),
            "strata": np.zeros(4, dtype=int)}, np.arange(4.0)[:, None]


def test_reverse_km_event_first_ties_and_unresolved_tail():
    time, mass = reverse_km([1, 1, 2, 3], [1, 0, 0, 1])
    assert_allclose(time, [1, 2, np.inf])
    assert_allclose(mass, [1 / 3, 1 / 3, 1 / 3])
    time, mass = reverse_km([2, 4, 6], [0, 1, 1])
    assert_allclose(time, [2, np.inf])
    assert_allclose(mass, [1 / 3, 2 / 3])


def test_weibull_scale_matches_closed_form_with_constant_censoring():
    cells, X = source()
    model = fit_model(cells, X, hr=1, shape=2, target=0.4)
    expected_scale = 5 / np.sqrt(-np.log(0.6))
    assert_allclose(np.exp(model["log_scale"]), expected_scale, rtol=1e-12)
    assert model["expected_event_fraction"] == pytest.approx(0.4, abs=1e-13)
    assert_allclose(model["censoring"][0]["time"], [5, 5])
    assert_allclose(model["censoring"][0]["mass"], [1, 0])


def test_source_sd_frozen_and_hr_quantile_identity():
    cells, X = source()
    model = fit_model(cells, X, hr=2, shape=1.5)
    assert model["source_mean"] == 1.5
    assert model["source_sd"] == pytest.approx(np.sqrt(1.25))
    model["log_scale"] = -10  # Avoid censoring for this paired quantile check.
    t1, e1, _ = outcomes(model, [0], 17, 19)
    t2, e2, _ = outcomes(model, [1], 17, 19)
    expected_ratio = 2 ** (-(X[1, 0] - X[0, 0]) / model["source_sd"] / 1.5)
    assert e1[0] and e2[0]
    assert t2[0] / t1[0] == pytest.approx(expected_ratio)


def test_independent_generated_events_match_analytic_target():
    cells, X = source()
    model = fit_model(cells, X, hr=2.5, shape=1.2, target=0.35)
    patients = np.tile(np.arange(4), 25000)
    t, event, strata = outcomes(model, patients, 123, 456)
    assert np.isfinite(t).all() and (t >= 0).all()
    assert abs(event.mean() - 0.35) < 0.006
    assert_allclose(strata, 0)
    other = outcomes(model, patients, 123, 456)
    assert_allclose(other[0], t)
    assert_allclose(other[1], event)


def test_tail_mass_becomes_declared_administrative_horizon():
    cells = {"time": np.array([2, 4, 6]), "event": np.array([0, 1, 1]), "strata": np.zeros(3, int)}
    model = fit_model(cells, np.arange(3.0)[:, None], hr=1)
    dist = model["censoring"][0]
    assert_allclose(dist["time"], [2, 6])
    assert_allclose(dist["mass"], [1 / 3, 2 / 3])
    assert dist["unresolved_tail_mass"] == pytest.approx(2 / 3)


def test_plant_selection_never_uses_outcomes_or_missing_value_imputation():
    cells, X = source()
    X = np.column_stack([np.full(4, 0.1), [0, np.nan, 2, 3], X])
    model = fit_model(cells, X)
    assert model["feature"] == 2
    changed = {**cells, "event": np.array([0, 1, 0, 1])}
    assert fit_model(changed, X)["feature"] == 2
    with pytest.raises(ValueError, match="plant"):
        fit_model(cells, X, feature=1)
    cells["strata"] = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="plant"):
        fit_model(cells, np.array([[0.1], [0.1], [0.2], [0.2]]))


@pytest.mark.parametrize("hr,shape,target", [(0, 1, 0.2), (1, 0, 0.2), (1, 1, 0), (1, 1, 1), (np.nan, 1, 0.2)])
def test_invalid_models_fail_clearly(hr, shape, target):
    cells, X = source()
    with pytest.raises(ValueError):
        fit_model(cells, X, hr=hr, shape=shape, target=target)


def test_invalid_patient_indices_and_impossible_event_target():
    cells, X = source()
    model = fit_model(cells, X)
    for indices in [[], [-1], [4], [0.5]]:
        with pytest.raises(ValueError):
            outcomes(model, indices, 1, 2)
    cells["time"] = np.zeros(4)
    with pytest.raises(ValueError, match="support"):
        fit_model(cells, X, target=0.2)
