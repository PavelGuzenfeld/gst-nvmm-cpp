import numpy as np

from data.config import Regime
from data.trajectory_gen import measure
from filters.ekf import EKF
from filters.motion_models import cv_model, step
from filters.ukf import UKF
from filters.state import STATE_DIM


def _pure_cv_trajectory(rng, regime, T):
    """Ground truth generated directly from the CV model, bypassing the
    random segment-kind draw in data.trajectory_gen -- these tests want the
    filter's own model assumption to be exactly correct, not whatever motion
    kind the segment sampler happens to pick for a given seed."""
    F, Q = cv_model(regime.dt, sigma_a=regime.cv_sigma_a)
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(STATE_DIM))
    x = np.zeros(STATE_DIM)
    x[:3] = [3000.0, 500.0, 200.0]
    x[3:6] = [50.0, -20.0, 5.0]
    states = np.zeros((T, STATE_DIM))
    for t in range(T):
        states[t] = x
        x = step(x, F) + L @ rng.standard_normal(STATE_DIM)
    return states


def _run_filter(flt, states, measurements):
    T = states.shape[0]
    x = states[0].copy()
    x[3:] += np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0])  # perturbed init
    P = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])

    xs, Ps, innovs, Ss = [], [], [], []
    for t in range(T):
        if t == 0:
            x_pred, P_pred = x, P
            if isinstance(flt, UKF):
                pts_pred = flt._sigma_points(x_pred, P_pred)
                x_upd, P_upd, y, S = flt.update(x_pred, P_pred, pts_pred, measurements[t])
            else:
                x_upd, P_upd, y, S = flt.update(x_pred, P_pred, measurements[t])
        else:
            x_upd, P_upd, y, S = flt.step(xs[-1], Ps[-1], measurements[t])
        xs.append(x_upd)
        Ps.append(P_upd)
        innovs.append(y)
        Ss.append(S)
    return np.array(xs), np.array(Ps), np.array(innovs), np.array(Ss)


def test_ekf_tracks_cv_trajectory_with_bounded_error_and_valid_covariance():
    regime = Regime(name="cv_only", T=200)
    states = _pure_cv_trajectory(np.random.default_rng(3), regime, regime.T)
    z = measure(np.random.default_rng(4), states, regime)

    ekf = EKF(regime.dt, regime.R())
    xs, Ps, _, _ = _run_filter(ekf, states, z)

    pos_err = np.linalg.norm(xs[-20:, :3] - states[-20:, :3], axis=-1)
    assert np.mean(pos_err) < 20.0  # converges to sub-20m tracking error

    for P in Ps:
        assert np.allclose(P, P.T, atol=1e-6)
        assert np.all(np.linalg.eigvalsh(P) > -1e-8)


def test_ukf_matches_ekf_closely_on_mildly_nonlinear_case():
    regime = Regime(name="cv_only2", T=150)
    states = _pure_cv_trajectory(np.random.default_rng(5), regime, regime.T)
    z = measure(np.random.default_rng(6), states, regime)

    ekf = EKF(regime.dt, regime.R())
    ukf = UKF(regime.dt, regime.R())
    xs_ekf, _, _, _ = _run_filter(ekf, states, z)
    xs_ukf, Ps_ukf, _, _ = _run_filter(ukf, states, z)

    assert np.allclose(xs_ekf[-1, :3], xs_ukf[-1, :3], atol=5.0)
    for P in Ps_ukf:
        assert np.allclose(P, P.T, atol=1e-4)


def test_ukf_handles_target_crossing_the_azimuth_wrap_boundary():
    """Construct sigma points that straddle +-pi and confirm the UKF's
    circular-mean measurement update doesn't diverge."""
    regime = Regime(name="wrap", T=1)
    dt = regime.dt
    ukf = UKF(dt, regime.R())

    x = np.zeros(STATE_DIM)
    r = 1000.0
    x[0] = r * np.cos(np.pi - 0.005)
    x[1] = r * np.sin(np.pi - 0.005)  # azimuth just under +pi
    x[3] = 50.0  # moving so sigma points spread across the seam
    P = np.diag([200.0, 200.0, 50.0, 20.0, 20.0, 5.0, 1.0, 1.0, 1.0])

    x_pred, P_pred, pts_pred = ukf.predict(x, P)
    from filters.measurement_model import h
    z = h(x_pred)
    x_upd, P_upd, y, S = ukf.update(x_pred, P_pred, pts_pred, z)

    assert np.all(np.isfinite(x_upd))
    assert abs(y[0]) < 0.1  # innovation on a noiseless self-consistent measurement stays tiny
    assert np.all(np.linalg.eigvalsh(P_upd) > -1e-6)
