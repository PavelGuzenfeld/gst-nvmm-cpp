import numpy as np

from filters.motion_models import cv_model, ca_model, ct_model, step
from filters.state import STATE_DIM, PX, PY, VX, VY


def test_cv_position_update_matches_constant_velocity():
    dt = 0.02
    F, Q = cv_model(dt)
    x = np.array([0.0, 0.0, 100.0, 10.0, -5.0, 0.0, 0.0, 0.0, 0.0])
    x1 = step(x, F)
    assert np.isclose(x1[PX], x[PX] + x[VX] * dt, atol=1e-3)
    assert np.isclose(x1[PY], x[PY] + x[VY] * dt, atol=1e-3)
    assert np.all(np.linalg.eigvalsh(Q) > -1e-12)


def test_ct_zero_omega_matches_cv_horizontal_block():
    dt = 0.02
    F_cv, _ = cv_model(dt)
    F_ct, _ = ct_model(dt, omega=0.0)
    h_idx = [PX, VX, PY, VY]
    assert np.allclose(F_ct[np.ix_(h_idx, h_idx)], F_cv[np.ix_(h_idx, h_idx)])


def test_ct_rotation_preserves_horizontal_speed_noiseless():
    dt = 0.02
    omega = np.deg2rad(15.0)
    F, _ = ct_model(dt, omega=omega)
    x = np.zeros(STATE_DIM)
    x[VX], x[VY] = 20.0, 0.0
    speed0 = np.hypot(x[VX], x[VY])
    for _ in range(100):
        x = step(x, F)
    speed1 = np.hypot(x[VX], x[VY])
    assert np.isclose(speed0, speed1, rtol=1e-6)


def test_ca_has_larger_stationary_accel_variance_than_cv():
    dt = 0.02
    _, Q_cv = cv_model(dt)
    _, Q_ca = ca_model(dt)
    from filters.state import A_X
    assert Q_ca[A_X, A_X] > Q_cv[A_X, A_X]


def test_all_process_noise_covariances_are_psd():
    dt = 0.02
    for F, Q in (cv_model(dt), ca_model(dt), ct_model(dt, omega=np.deg2rad(10))):
        assert np.allclose(Q, Q.T)
        assert np.all(np.linalg.eigvalsh(Q) > -1e-10)
