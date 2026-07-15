"""KalmanNet trains through torch h()/f(); the classical filters evaluate
through the numpy versions. If the two diverge even slightly, KalmanNet
trains against a different model than it's judged against -- verify
agreement directly, and verify the wrap fix survives in the torch path (the
brief's #1 bug, again, on a second code path)."""

import numpy as np
import torch

from filters.measurement_model import H_jacobian, h, innovation, wrap_angle
from filters.state import STATE_DIM
from filters.torch_ops import CVMotionModel, H_jacobian_torch, h_torch, innovation_torch, wrap_angle_torch
from filters.motion_models import cv_model, step


def test_wrap_angle_torch_matches_numpy():
    angles = np.linspace(-10 * np.pi, 10 * np.pi, 2001)
    np_wrapped = wrap_angle(angles)
    torch_wrapped = wrap_angle_torch(torch.tensor(angles)).numpy()
    assert np.allclose(np_wrapped, torch_wrapped, atol=1e-10)


def test_h_torch_matches_numpy_on_random_states():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = np.zeros(STATE_DIM)
        x[:3] = rng.uniform(-2000, 2000, size=3)
        z_np = h(x)
        z_torch = h_torch(torch.tensor(x)).numpy()
        assert np.allclose(z_np, z_torch, atol=1e-8)


def test_innovation_torch_matches_numpy_including_wrap_boundary():
    rng = np.random.default_rng(1)
    for _ in range(50):
        x_true = np.zeros(STATE_DIM)
        x_true[:3] = rng.uniform(-2000, 2000, size=3)
        x_pred = np.zeros(STATE_DIM)
        x_pred[:3] = rng.uniform(-2000, 2000, size=3)
        z = h(x_true)

        innov_np = innovation(z.copy(), x_pred)
        innov_torch = innovation_torch(torch.tensor(z), torch.tensor(x_pred)).numpy()
        assert np.allclose(innov_np, innov_torch, atol=1e-8)

    # explicit wrap-boundary case, mirroring test_measurement_model.py
    r, el = 100.0, 0.0
    az_true, az_pred = np.pi - 0.01, -np.pi + 0.01
    x_true = np.array([
        r * np.cos(el) * np.cos(az_true), r * np.cos(el) * np.sin(az_true), r * np.sin(el),
        0, 0, 0, 0, 0, 0,
    ])
    x_pred = np.array([
        r * np.cos(el) * np.cos(az_pred), r * np.cos(el) * np.sin(az_pred), r * np.sin(el),
        0, 0, 0, 0, 0, 0,
    ])
    z = h(x_true)
    innov_torch = innovation_torch(torch.tensor(z), torch.tensor(x_pred)).numpy()
    assert abs(innov_torch[0]) < 0.05  # not a spurious ~2*pi jump


def test_H_jacobian_torch_matches_numpy_batched():
    rng = np.random.default_rng(2)
    xs = np.zeros((10, STATE_DIM))
    xs[:, :3] = rng.uniform(-2000, 2000, size=(10, 3))
    xs[:, 2] = rng.uniform(50, 500, size=10)  # keep pz away from the rho=0 singularity

    H_np = np.stack([H_jacobian(xs[i]) for i in range(10)])
    H_t = H_jacobian_torch(torch.tensor(xs)).numpy()
    assert np.allclose(H_np, H_t, atol=1e-8)


def test_cv_motion_model_matches_numpy_step():
    dt = 0.02
    F_np, _ = cv_model(dt)
    x = np.array([100.0, 200.0, 50.0, 10.0, -5.0, 1.0, 0.0, 0.0, 0.0])

    x1_np = step(x, F_np)
    torch_model = CVMotionModel(dt)
    x1_torch = torch_model(torch.tensor(x, dtype=torch.float64)).numpy()
    assert np.allclose(x1_np, x1_torch, atol=1e-10)
