"""KalmanNet filter recursion: architecture #2's gain network driving the
same predict/update structure as the classical filters, but with NO explicit
covariance propagation -- Q and R are exactly what the learned gain replaces
(NOTES.md sec. 1c). Ported from the reference KNet_step/step_prior/
step_KGain_est, with one correctness fix the reference never needed: every
angular quantity (the observation difference, the innovation, and the
update itself) is wrapped to (-pi, pi] before use. The reference was built
for the Lorenz-attractor / linear-canonical cases, which have no angle
measurement at all -- ported verbatim, a target crossing the +-pi seam would
hand the network a spurious ~2*pi feature and a ~2*pi correction.
"""

import torch
import torch.nn.functional as F

from filters.torch_ops import CVMotionModel, h_torch, wrap_measurement_diff


class KalmanNetFilter:
    def __init__(self, gain_net, dt, device="cpu"):
        self.net = gain_net
        self.f = CVMotionModel(dt, device=device)
        self.device = device

    def init_sequence(self, x0, prior_Q, prior_Sigma, prior_S):
        batch = x0.shape[0]
        self.hidden = self.net.init_hidden(batch, prior_Q, prior_Sigma, prior_S, self.device)
        self.x_posterior = x0
        self.x_posterior_prev = x0
        self.x_prior_prev = x0
        self.y_prev = h_torch(x0)

    def step(self, z):
        x_prior = self.f(self.x_posterior)
        y_pred = h_torch(x_prior)

        obs_diff = wrap_measurement_diff(z - self.y_prev)
        innov = wrap_measurement_diff(z - y_pred)
        fw_evol_diff = self.x_posterior - self.x_posterior_prev
        fw_update_diff = self.x_posterior - self.x_prior_prev

        obs_diff = F.normalize(obs_diff, p=2, dim=-1, eps=1e-12)
        obs_innov_diff = F.normalize(innov, p=2, dim=-1, eps=1e-12)
        fw_evol_diff = F.normalize(fw_evol_diff, p=2, dim=-1, eps=1e-12)
        fw_update_diff = F.normalize(fw_update_diff, p=2, dim=-1, eps=1e-12)

        gain, self.hidden = self.net(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff, self.hidden)

        x_upd = x_prior + torch.bmm(gain, innov.unsqueeze(-1)).squeeze(-1)

        self.x_posterior_prev = self.x_posterior
        self.x_prior_prev = x_prior
        self.x_posterior = x_upd
        self.y_prev = z
        return x_upd

    def detach_state(self):
        """Detach recurrent state from the autograd graph at a truncated-BPTT boundary."""
        self.x_posterior = self.x_posterior.detach()
        self.x_posterior_prev = self.x_posterior_prev.detach()
        self.x_prior_prev = self.x_prior_prev.detach()
        self.y_prev = self.y_prev.detach()
        self.hidden = tuple(h.detach() for h in self.hidden)
