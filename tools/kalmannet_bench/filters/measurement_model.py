"""Nonlinear azimuth/elevation/range measurement model, sensor at the origin.

z = h(x) + v,  h: Cartesian -> spherical. Shared by every filter so the
comparison in the brief is apples-to-apples: EKF/UKF/IMM/KalmanNet/consistency
variants all see identical measurements and identical innovation code.
"""

import numpy as np

from filters.state import PX, PY, PZ, STATE_DIM

# Floor on the horizontal/full range to avoid the true singularity of
# spherical coordinates at rho=0 (target directly overhead) or r=0 (target at
# the sensor) -- both are measure-zero in practice but would otherwise divide
# by zero and NaN the whole filter.
_EPS = 1e-6


def wrap_angle(a):
    """Wrap angle(s) in radians to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def h(x):
    """Cartesian state -> [azimuth, elevation, range] measurement."""
    px, py, pz = x[PX], x[PY], x[PZ]
    rho = np.hypot(px, py)
    r = np.hypot(rho, pz)
    az = np.arctan2(py, px)
    el = np.arctan2(pz, rho)
    return np.array([az, el, r])


def H_jacobian(x):
    """d h / d x at x, zero in the velocity/acceleration columns (h depends
    only on position)."""
    px, py, pz = x[PX], x[PY], x[PZ]
    rho2 = max(px * px + py * py, _EPS)
    rho = np.sqrt(rho2)
    r2 = max(rho2 + pz * pz, _EPS)
    r = np.sqrt(r2)

    H = np.zeros((3, STATE_DIM))
    # d(azimuth)/d(px, py)
    H[0, PX] = -py / rho2
    H[0, PY] = px / rho2
    # d(elevation)/d(px, py, pz)
    H[1, PX] = -pz * px / (r2 * rho)
    H[1, PY] = -pz * py / (r2 * rho)
    H[1, PZ] = rho / r2
    # d(range)/d(px, py, pz)
    H[2, PX] = px / r
    H[2, PY] = py / r
    H[2, PZ] = pz / r
    return H


def innovation(z_meas, x_pred):
    """z_meas - h(x_pred), with azimuth/elevation components wrapped to
    (-pi, pi] so a target sitting across the +-pi boundary doesn't produce a
    spurious ~2*pi innovation."""
    z_pred = h(x_pred)
    innov = z_meas - z_pred
    innov[0] = wrap_angle(innov[0])
    innov[1] = wrap_angle(innov[1])
    return innov
