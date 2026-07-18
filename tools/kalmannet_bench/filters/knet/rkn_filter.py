"""Recursive KalmanNet filter recursion -- ported from the reference's
`step()` / `get_features()` / `update_cov()` (Algo/KalmanFilter.py,
Algo/RecursiveKalmanNet.py), adapted to our nonlinear az/el/range h()/
H_jacobian() (the reference assumes a linear H matrix multiplying x
directly). The angle-wrap fix applied to vanilla KalmanNet applies here too,
and in one more place: the raw z_diff feature (z_t - z_previous) also
contains azimuth and must be wrapped, not just the innovation.
"""

import torch

from filters.state import STATE_DIM
from filters.torch_ops import CVMotionModel, H_jacobian_torch, h_torch, wrap_measurement_diff


class RKNFilter:
    def __init__(self, network, dt, device="cpu"):
        self.net = network
        self.f = CVMotionModel(dt, device=device)
        self.device = device

    def init_sequence(self, x0, P0):
        batch = x0.shape[0]
        self.hidden = self.net.init_hidden(batch, self.device)
        self.x = x0
        self.P = P0
        # Mirrors the reference's init_sequence: treat the "previous prior"
        # as x0 propagated once, and "previous observation" as h(x0).
        self.x_prior_previous = self.f(x0)
        self.z_previous = h_torch(x0)

    def step(self, z):
        x_prior = self.f(self.x)
        H_t = H_jacobian_torch(x_prior)
        y = wrap_measurement_diff(z - h_torch(x_prior))

        correction = self.x - self.x_prior_previous
        z_diff = wrap_measurement_diff(z - self.z_previous)

        self.z_previous = z
        self.x_prior_previous = x_prior

        features = torch.cat([y, z_diff, H_t.reshape(H_t.shape[0], -1), correction], dim=-1)
        features = features ** 2

        K, B, self.hidden = self.net(features, self.hidden)

        x_upd = x_prior + torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)

        I = torch.eye(STATE_DIM, dtype=x_prior.dtype, device=x_prior.device).unsqueeze(0)
        IKH = I - K @ H_t
        F = self.f.F.unsqueeze(0)
        A = IKH @ F @ self.P @ F.transpose(1, 2) @ IKH.transpose(1, 2)
        P_upd = A + B

        self.x = x_upd
        self.P = P_upd
        return x_upd, P_upd

    def detach_state(self):
        """Detach recurrent state from the autograd graph at a truncated-BPTT boundary."""
        self.x = self.x.detach()
        self.P = self.P.detach()
        self.x_prior_previous = self.x_prior_previous.detach()
        self.z_previous = self.z_previous.detach()
        self.hidden = tuple(h.detach() for h in self.hidden)
