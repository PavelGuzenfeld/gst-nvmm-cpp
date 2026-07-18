import numpy as np
from scipy.stats import chi2

from filters.metrics import (
    anees_with_bounds,
    empirical_coverage,
    fraction_inside_bounds,
    nees,
    nis,
    position_rmse,
    velocity_rmse,
)


def test_position_and_velocity_rmse_known_values():
    x_true = np.zeros((2, 9))
    x_est = np.zeros((2, 9))
    x_est[0, 0] = 3.0
    x_est[0, 1] = 4.0  # position error norm 5 at step 0, 0 at step 1
    rmse = position_rmse(x_true, x_est)
    assert np.isclose(rmse, np.sqrt((25 + 0) / 2))

    x_est_v = np.zeros((2, 9))
    x_est_v[:, 3] = 3.0
    x_est_v[:, 4] = 4.0
    assert np.isclose(velocity_rmse(x_true, x_est_v), 5.0)


def test_nees_matches_hand_computation_for_identity_covariance():
    e = np.array([1.0, 2.0, 3.0])
    x_true = e
    x_est = np.zeros(3)
    P = np.eye(3)
    value = nees(x_true, x_est, P, idx_slice=slice(None))
    assert np.isclose(value, 1.0 + 4.0 + 9.0)


def test_nis_matches_hand_computation():
    y = np.array([2.0, 0.0])
    S = np.diag([4.0, 1.0])
    value = nis(y, S)
    assert np.isclose(value, 1.0)  # 2^2/4 + 0


def test_anees_bounds_match_scipy_chi2_directly():
    # A well-calibrated filter has E[NEES] = dim (the chi-square mean), not 1
    # -- use that as the "perfectly consistent" synthetic value.
    N, dim = 10, 6
    nees_values = np.full((N, 5), float(dim))
    anees, (low, high) = anees_with_bounds(nees_values, dim=dim, mc_axis=0)
    assert np.allclose(anees, float(dim))
    assert np.isclose(low, chi2.ppf(0.025, df=N * dim) / N)
    assert np.isclose(high, chi2.ppf(0.975, df=N * dim) / N)
    assert low < dim < high  # a perfectly consistent filter should pass


def test_fraction_inside_bounds_counts_correctly():
    values = np.array([0.5, 1.0, 5.0, 10.0])
    frac = fraction_inside_bounds(values, (0.9, 5.5))
    assert np.isclose(frac, 0.5)  # 1.0 and 5.0 are inside


def test_empirical_coverage_all_inside_gives_full_coverage():
    dim = 3
    threshold_95 = chi2.ppf(0.95, df=dim)
    values = np.full(100, threshold_95 * 0.5)
    cov = empirical_coverage(values, dim=dim, levels=(0.68, 0.95))
    assert cov[0.95] == 1.0
