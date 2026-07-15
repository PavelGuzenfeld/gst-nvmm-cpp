import numpy as np

from data.config import Regime
from data.trajectory_gen import measure
from filters.ekf import EKF
from filters.imm import IMM
from filters.motion_models import ct_model, ca_model, step
from filters.state import STATE_DIM


def _ct_trajectory(rng, regime, T, omega):
    F, Q = ct_model(regime.dt, omega=omega, sigma_a=regime.cv_sigma_a)
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(STATE_DIM))
    x = np.zeros(STATE_DIM)
    x[:3] = [3000.0, 500.0, 200.0]
    x[3:6] = [60.0, 0.0, 0.0]
    states = np.zeros((T, STATE_DIM))
    for t in range(T):
        states[t] = x
        x = step(x, F) + L @ rng.standard_normal(STATE_DIM)
    return states


def _ca_trajectory(rng, regime, T, sigma_a, tau):
    F, Q = ca_model(regime.dt, sigma_a=sigma_a, tau=tau)
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(STATE_DIM))
    x = np.zeros(STATE_DIM)
    x[:3] = [3000.0, 500.0, 200.0]
    x[3:6] = [60.0, 0.0, 0.0]
    states = np.zeros((T, STATE_DIM))
    for t in range(T):
        states[t] = x
        x = step(x, F) + L @ rng.standard_normal(STATE_DIM)
    return states


def _run_imm(imm, states, measurements):
    x0 = states[0].copy()
    x0[3:] += np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    P0 = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    state = imm.init_state(x0, P0)

    xs, Ps, mus = [], [], []
    for t in range(states.shape[0]):
        x_out, P_out, state, _, _ = imm.step(state, measurements[t])
        xs.append(x_out)
        Ps.append(P_out)
        mus.append(state["mu"].copy())
    return np.array(xs), np.array(Ps), np.array(mus)


def test_mode_probabilities_stay_valid_throughout():
    regime = Regime(name="imm_ct", T=200)
    states = _ct_trajectory(np.random.default_rng(1), regime, regime.T, omega=np.deg2rad(15.0))
    z = measure(np.random.default_rng(2), states, regime)

    imm = IMM(regime.dt, regime.R())
    _, Ps, mus = _run_imm(imm, states, z)

    assert np.allclose(np.sum(mus, axis=1), 1.0)
    assert np.all(mus >= -1e-9) and np.all(mus <= 1 + 1e-9)
    for P in Ps:
        assert np.allclose(P, P.T, atol=1e-4)
        assert np.all(np.linalg.eigvalsh(P) > -1e-6)


def test_imm_assigns_majority_probability_to_ct_during_sustained_turn():
    regime = Regime(name="imm_ct2", T=300)
    omega = np.deg2rad(15.0)
    states = _ct_trajectory(np.random.default_rng(3), regime, regime.T, omega=omega)
    z = measure(np.random.default_rng(4), states, regime)

    imm = IMM(regime.dt, regime.R(), omega_ct=omega)
    _, _, mus = _run_imm(imm, states, z)

    ct_mass = mus[-1, 1] + mus[-1, 2]  # ct_pos + ct_neg
    assert ct_mass > 0.5


def test_imm_prefers_ca_on_average_during_sustained_acceleration():
    # Mode probabilities are a noisy, single-step recursive estimate over a
    # genuinely random (Singer) acceleration realization -- any single step
    # can swing on one unlucky innovation, so assert on the tail-averaged
    # mass rather than the literal last step.
    regime = Regime(name="imm_ca", T=300)
    states = _ca_trajectory(np.random.default_rng(5), regime, regime.T, sigma_a=6.0, tau=3.0)
    z = measure(np.random.default_rng(6), states, regime)

    imm = IMM(regime.dt, regime.R())
    _, _, mus = _run_imm(imm, states, z)

    mean_mu = mus[-100:].mean(axis=0)
    assert np.argmax(mean_mu) == 3  # ca is the single most-favored mode on average
    assert mean_mu[3] > mean_mu[0]  # and clearly beats cv


def test_imm_beats_single_model_ekf_during_a_turn():
    regime = Regime(name="imm_vs_ekf", T=250)
    omega = np.deg2rad(20.0)
    states = _ct_trajectory(np.random.default_rng(7), regime, regime.T, omega=omega)
    z = measure(np.random.default_rng(8), states, regime)

    imm = IMM(regime.dt, regime.R(), omega_ct=omega)
    xs_imm, _, _ = _run_imm(imm, states, z)

    ekf = EKF(regime.dt, regime.R())
    x = states[0].copy()
    x[3:] += np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    P = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    xs_ekf = []
    for t in range(states.shape[0]):
        if t == 0:
            x, P, _, _ = ekf.update(x, P, z[t])
        else:
            x, P, _, _ = ekf.step(x, P, z[t])
        xs_ekf.append(x)
    xs_ekf = np.array(xs_ekf)

    tail = slice(-50, None)
    err_imm = np.linalg.norm(xs_imm[tail, :3] - states[tail, :3], axis=-1).mean()
    err_ekf = np.linalg.norm(xs_ekf[tail, :3] - states[tail, :3], axis=-1).mean()
    assert err_imm < err_ekf
