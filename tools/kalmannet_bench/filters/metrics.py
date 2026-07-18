"""Shared evaluation metrics -- identical code path for every filter, per the
brief's fairness requirement.

NEES/NIS use single-run chi-square bounds only as building blocks; the
headline consistency numbers are ANEES/ANIS (averaged over N Monte Carlo
runs), tested against the chi-square interval for N*dim degrees of freedom
scaled by N -- single-run NEES bounds are too loose to discriminate a good
filter from an overconfident one.
"""

import numpy as np
from scipy.stats import chi2

from filters.state import POS, VEL


def position_rmse(x_true, x_est):
    """x_true, x_est: [..., 9]. RMSE over the position sub-state."""
    err = x_true[..., POS] - x_est[..., POS]
    return np.sqrt(np.mean(np.sum(err**2, axis=-1)))


def velocity_rmse(x_true, x_est):
    err = x_true[..., VEL] - x_est[..., VEL]
    return np.sqrt(np.mean(np.sum(err**2, axis=-1)))


def per_step_rmse(x_true, x_est, idx_slice):
    """RMSE at each time step, averaged over the leading (trajectory/MC) axes.
    x_true, x_est: [..., T, 9]."""
    err = x_true[..., idx_slice] - x_est[..., idx_slice]
    sq = np.sum(err**2, axis=-1)  # [..., T]
    return np.sqrt(np.mean(sq, axis=tuple(range(sq.ndim - 1))))


def nees(x_true, x_est, P, idx_slice=slice(None)):
    """Normalized Estimation Error Squared, one value per (leading-axis,
    time) entry. x_true, x_est: [..., n]; P: [..., n, n]."""
    e = (x_true - x_est)[..., idx_slice]
    P_sub = P[..., idx_slice, idx_slice]
    Pinv_e = np.linalg.solve(P_sub, e[..., None])[..., 0]
    return np.sum(e * Pinv_e, axis=-1)


def nis(y, S):
    """Normalized Innovation Squared. y: [..., m]; S: [..., m, m]."""
    Sinv_y = np.linalg.solve(S, y[..., None])[..., 0]
    return np.sum(y * Sinv_y, axis=-1)


def anees_with_bounds(nees_values, dim, mc_axis, confidence=0.95):
    """nees_values: array with a Monte-Carlo axis (mc_axis) of size N and a
    time axis; averages over mc_axis to get ANEES per time step, and returns
    the [low, high] interval for the chi-square distribution with N*dim
    degrees of freedom, scaled by 1/N (per the brief's headline consistency
    test)."""
    N = nees_values.shape[mc_axis]
    anees = np.mean(nees_values, axis=mc_axis)
    alpha = 1 - confidence
    low = chi2.ppf(alpha / 2, df=N * dim) / N
    high = chi2.ppf(1 - alpha / 2, df=N * dim) / N
    return anees, (low, high)


def fraction_inside_bounds(values, bounds):
    low, high = bounds
    return float(np.mean((values >= low) & (values <= high)))


def empirical_coverage(nees_values, dim, levels=(0.68, 0.95)):
    """Fraction of (run, step) NEES/NIS entries falling inside the nominal
    single-draw chi-square ellipsoid at each confidence level -- the
    calibration curve: empirical coverage vs. nominal."""
    out = {}
    for p in levels:
        threshold = chi2.ppf(p, df=dim)
        out[p] = float(np.mean(nees_values <= threshold))
    return out
