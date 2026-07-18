import numpy as np

from data.config import Regime
from data.trajectory_gen import generate_ground_truth, measure


def test_ground_truth_has_no_nans_and_switches_modes():
    rng = np.random.default_rng(0)
    small = Regime(name="test_small", T=300)
    states, labels = generate_ground_truth(rng, small)
    assert states.shape == (300, 9)
    assert np.all(np.isfinite(states))
    assert set(np.unique(labels)).issubset({0, 1, 2})
    assert len(np.unique(labels)) >= 2  # switches modes within 300 steps


def test_generation_is_reproducible_given_same_seed():
    regime = Regime(name="test_repro", T=100)
    s1, l1 = generate_ground_truth(np.random.default_rng(42), regime)
    s2, l2 = generate_ground_truth(np.random.default_rng(42), regime)
    assert np.array_equal(s1, s2)
    assert np.array_equal(l1, l2)


def test_measurements_stay_within_wrapped_azimuth_range():
    rng = np.random.default_rng(1)
    regime = Regime(name="test_meas", T=200)
    states, _ = generate_ground_truth(rng, regime)
    z = measure(np.random.default_rng(2), states, regime)
    assert np.all(z[:, 0] >= -np.pi) and np.all(z[:, 0] < np.pi)
    assert np.all(z[:, 2] > 0)  # range stays positive for these initial conditions


def test_heavy_tailed_regime_has_larger_measurement_spread():
    T = 2000
    gaussian = Regime(name="g", T=T, heavy_tailed=False)
    heavy = Regime(name="h", T=T, heavy_tailed=True, heavy_tailed_dof=3.0)

    rng = np.random.default_rng(7)
    states, _ = generate_ground_truth(rng, gaussian)

    z_gauss = measure(np.random.default_rng(8), states, gaussian)
    z_heavy = measure(np.random.default_rng(8), states, heavy)

    from filters.measurement_model import h
    resid_gauss = np.array([z_gauss[t, 2] - h(states[t])[2] for t in range(T)])
    resid_heavy = np.array([z_heavy[t, 2] - h(states[t])[2] for t in range(T)])
    assert np.std(resid_heavy) > np.std(resid_gauss)
