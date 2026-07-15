import numpy as np
import torch

from filters.knet.rkn_network import RKNGainCovNetwork
from filters.knet.rkn_filter import RKNFilter
from filters.measurement_model import h
from filters.state import STATE_DIM


def _p0(batch):
    return torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0).repeat(batch, 1, 1) * 100.0


def test_rkn_forward_pass_is_finite_and_p_stays_valid():
    torch.manual_seed(0)
    net = RKNGainCovNetwork().double()
    flt = RKNFilter(net, dt=0.02)

    batch = 3
    x0 = torch.zeros(batch, STATE_DIM, dtype=torch.float64)
    x0[:, 0] = 1000.0
    x0[:, 1] = 200.0
    x0[:, 2] = 50.0
    flt.init_sequence(x0, _p0(batch))

    for t in range(5):
        z = torch.tensor(
            np.stack([h(x0[b].numpy() + t) for b in range(batch)]), dtype=torch.float64
        )
        x_upd, P_upd = flt.step(z)
        assert x_upd.shape == (batch, STATE_DIM)
        assert P_upd.shape == (batch, STATE_DIM, STATE_DIM)
        assert torch.all(torch.isfinite(x_upd))
        assert torch.all(torch.isfinite(P_upd))
        for b in range(batch):
            P_b = P_upd[b].detach().numpy()
            assert np.allclose(P_b, P_b.T, atol=1e-6)
            assert np.all(np.linalg.eigvalsh(P_b) > -1e-6)


def test_gradient_flows_through_both_networks():
    torch.manual_seed(1)
    net = RKNGainCovNetwork().double()
    flt = RKNFilter(net, dt=0.02)

    batch = 2
    x0 = torch.zeros(batch, STATE_DIM, dtype=torch.float64)
    x0[:, 0] = 1500.0
    x0[:, 2] = 100.0
    flt.init_sequence(x0, _p0(batch))

    loss = torch.tensor(0.0, dtype=torch.float64)
    x_true = x0.clone()
    for _ in range(4):
        x_true = x_true.clone()
        x_true[:, 0] += 20.0 * 0.02
        z = torch.stack([
            torch.tensor(h(x_true[b].detach().numpy()), dtype=torch.float64) for b in range(batch)
        ])
        x_upd, P_upd = flt.step(z)
        e = (x_upd - x_true).unsqueeze(-1)
        nll = torch.mean(torch.bmm(torch.bmm(e.transpose(1, 2), torch.linalg.inv(P_upd)), e))
        loss = loss + nll

    loss.backward()
    grad_norms_K = [p.grad.norm().item() for p in net.rnn_K.parameters() if p.grad is not None]
    grad_norms_cov = [p.grad.norm().item() for p in net.rnn_cov.parameters() if p.grad is not None]
    assert len(grad_norms_K) > 0 and len(grad_norms_cov) > 0
    assert all(np.isfinite(g) for g in grad_norms_K + grad_norms_cov)
    assert any(g > 0 for g in grad_norms_K)
    assert any(g > 0 for g in grad_norms_cov)


def test_rkn_survives_azimuth_wrap_boundary():
    """Mirrors the equivalent vanilla-KalmanNet test -- the reference this
    was ported from never wraps angles, so this is where a regression could
    silently reappear on RKN's own feature computation (z_diff)."""
    torch.manual_seed(2)
    net = RKNGainCovNetwork().double()
    flt = RKNFilter(net, dt=0.02)

    r, el = 100.0, 0.0

    def state_at(az):
        x = np.zeros(STATE_DIM)
        x[0] = r * np.cos(el) * np.cos(az)
        x[1] = r * np.cos(el) * np.sin(az)
        x[2] = r * np.sin(el)
        return x

    x0 = torch.tensor(state_at(np.pi - 0.01), dtype=torch.float64).unsqueeze(0)
    flt.init_sequence(x0, _p0(1))

    z0 = torch.tensor(h(state_at(np.pi - 0.01)), dtype=torch.float64).unsqueeze(0)
    z1 = torch.tensor(h(state_at(-np.pi + 0.01)), dtype=torch.float64).unsqueeze(0)

    x1, _ = flt.step(z0)
    x2, P2 = flt.step(z1)

    assert torch.all(torch.isfinite(x2))
    assert torch.all(torch.isfinite(P2))
    assert torch.norm(x2[:, :2] - x1[:, :2]).item() < 50.0
