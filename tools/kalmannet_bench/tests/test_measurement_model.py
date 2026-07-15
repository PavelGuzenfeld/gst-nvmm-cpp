"""Angle-wrap is the #1 bug in angle-measurement filters (per the brief) --
this is the first executable test in the whole benchmark, written and passing
before the data generator, before any filter, before anything else."""

import numpy as np

from filters.measurement_model import h, innovation, wrap_angle
from filters.state import STATE_DIM


def _state_at(az, el, r):
    px = r * np.cos(el) * np.cos(az)
    py = r * np.cos(el) * np.sin(az)
    pz = r * np.sin(el)
    x = np.zeros(STATE_DIM)
    x[0], x[1], x[2] = px, py, pz
    return x


def test_innovation_wraps_across_pi_boundary():
    r, el = 100.0, 0.0
    x_true = _state_at(az=np.pi - 0.01, el=el, r=r)
    x_pred = _state_at(az=-np.pi + 0.01, el=el, r=r)

    z_meas = h(x_true)
    naive_diff = z_meas[0] - h(x_pred)[0]
    wrapped = innovation(z_meas, x_pred)

    # True azimuth separation is only 0.02 rad; a naive subtraction across
    # the +-pi seam produces a spurious ~2*pi difference instead.
    assert abs(naive_diff) > 6.0
    assert abs(wrapped[0]) < 0.05


def test_wrap_angle_is_idempotent_and_bounded():
    angles = np.linspace(-10 * np.pi, 10 * np.pi, 4001)
    wrapped = wrap_angle(angles)
    assert np.all(wrapped >= -np.pi) and np.all(wrapped < np.pi)
    # wrapping twice must be a no-op
    assert np.allclose(wrap_angle(wrapped), wrapped)


def test_measurement_jacobian_matches_finite_difference():
    from filters.measurement_model import H_jacobian

    rng = np.random.default_rng(0)
    for _ in range(20):
        x = np.zeros(STATE_DIM)
        x[0:3] = rng.uniform(-500, 500, size=3)
        x[2] = rng.uniform(50, 500)  # keep pz away from the rho=0 singularity path
        H = H_jacobian(x)

        eps = 1e-4
        H_fd = np.zeros((3, STATE_DIM))
        for j in range(3):  # h only depends on position columns
            dx = np.zeros(STATE_DIM)
            dx[j] = eps
            H_fd[:, j] = (h(x + dx) - h(x - dx)) / (2 * eps)

        assert np.allclose(H[:, :3], H_fd[:, :3], atol=1e-3)
        assert np.allclose(H[:, 3:], 0.0)
