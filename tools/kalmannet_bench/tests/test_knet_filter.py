import numpy as np
import torch

from filters.knet.gain_network import KalmanGainGRU
from filters.knet.knet_filter import KalmanNetFilter
from filters.measurement_model import h
from filters.state import MEAS_DIM, STATE_DIM


def _priors():
    prior_Q = torch.eye(STATE_DIM, dtype=torch.float64) * 0.1
    prior_Sigma = torch.eye(STATE_DIM, dtype=torch.float64) * 10.0
    prior_S = torch.eye(MEAS_DIM, dtype=torch.float64) * 0.01
    return prior_Q, prior_Sigma, prior_S


def test_knet_forward_pass_is_finite_and_correctly_shaped():
    torch.manual_seed(0)
    net = KalmanGainGRU().double()
    flt = KalmanNetFilter(net, dt=0.02)

    batch = 3
    x0 = torch.zeros(batch, STATE_DIM, dtype=torch.float64)
    x0[:, 0] = 1000.0
    x0[:, 1] = 200.0
    x0[:, 2] = 50.0
    prior_Q, prior_Sigma, prior_S = _priors()
    flt.init_sequence(x0, prior_Q, prior_Sigma, prior_S)

    for t in range(5):
        z = torch.tensor(
            np.stack([h(x0[b].numpy() + t) for b in range(batch)]), dtype=torch.float64
        )
        x_upd = flt.step(z)
        assert x_upd.shape == (batch, STATE_DIM)
        assert torch.all(torch.isfinite(x_upd))


def test_gradient_flows_through_truncated_bptt_window():
    torch.manual_seed(1)
    net = KalmanGainGRU().double()
    flt = KalmanNetFilter(net, dt=0.02)

    batch = 2
    x0 = torch.zeros(batch, STATE_DIM, dtype=torch.float64)
    x0[:, 0] = 1500.0
    x0[:, 2] = 100.0
    prior_Q, prior_Sigma, prior_S = _priors()
    flt.init_sequence(x0, prior_Q, prior_Sigma, prior_S)

    loss = torch.tensor(0.0, dtype=torch.float64)
    x_true = x0.clone()
    for _ in range(4):
        x_true = x_true.clone()
        x_true[:, 0] += 20.0 * 0.02
        z = torch.stack([
            torch.tensor(h(x_true[b].detach().numpy()), dtype=torch.float64) for b in range(batch)
        ])
        x_upd = flt.step(z)
        loss = loss + torch.mean((x_upd - x_true) ** 2)

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in net.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(np.isfinite(g) for g in grad_norms)
    assert any(g > 0 for g in grad_norms)


def test_knet_survives_azimuth_wrap_boundary_without_a_spurious_correction():
    """Mirrors tests/test_measurement_model.py's wrap test, but through the
    KalmanNet filter path -- the reference implementation this was ported
    from never wraps angles, so this is the one place a regression could
    silently reappear."""
    torch.manual_seed(2)
    net = KalmanGainGRU().double()
    flt = KalmanNetFilter(net, dt=0.02)

    r, el = 100.0, 0.0

    def state_at(az):
        x = np.zeros(STATE_DIM)
        x[0] = r * np.cos(el) * np.cos(az)
        x[1] = r * np.cos(el) * np.sin(az)
        x[2] = r * np.sin(el)
        return x

    x0 = torch.tensor(state_at(np.pi - 0.01), dtype=torch.float64).unsqueeze(0)
    prior_Q, prior_Sigma, prior_S = _priors()
    flt.init_sequence(x0, prior_Q, prior_Sigma, prior_S)

    # First measurement near +pi, second just past -pi -- a true azimuth
    # change of only ~0.02 rad, but a naive difference would show ~2*pi.
    z0 = torch.tensor(h(state_at(np.pi - 0.01)), dtype=torch.float64).unsqueeze(0)
    z1 = torch.tensor(h(state_at(-np.pi + 0.01)), dtype=torch.float64).unsqueeze(0)

    x1 = flt.step(z0)
    x2 = flt.step(z1)

    assert torch.all(torch.isfinite(x1))
    assert torch.all(torch.isfinite(x2))
    # The position update should stay physically small (meters), not the
    # huge jump a ~2*pi-corrupted gain-times-innovation term would produce.
    assert torch.norm(x2[:, :2] - x1[:, :2]).item() < 50.0
