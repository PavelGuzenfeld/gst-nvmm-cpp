"""KalmanNet architecture #2 (Revach et al., arXiv:2107.10043) -- three GRUs
(Q, Sigma, S) plus FC pre/post layers, ported from the reference
KNet/KalmanNet_nn.py (see NOTES.md sec. 1). Kept as close to the reference's
data flow as possible, including the non-standard "backward flow" where FC3
-> FC4 overwrites the Sigma-GRU's hidden state each step.

Tensors use the reference's seq-first GRU convention (seq_len=1, batch,
feature) rather than batch_first -- not a style choice: the backward-flow
output (shape [1, batch, hidden]) is fed straight back in as the Sigma-GRU's
next hidden state (shape [num_layers=1, batch, hidden]), and only lines up
because both leading dims are 1. Switching to batch_first would silently
transpose batch and hidden on that feedback path for batch_size > 1.
"""

import torch
import torch.nn as nn

from filters.state import MEAS_DIM, STATE_DIM


class KalmanGainGRU(nn.Module):
    def __init__(self, m=STATE_DIM, n=MEAS_DIM, in_mult=5, out_mult=40):
        super().__init__()
        self.m, self.n = m, n

        self.d_hidden_Q = m * m
        self.GRU_Q = nn.GRU(m * in_mult, self.d_hidden_Q)

        self.d_hidden_Sigma = m * m
        self.GRU_Sigma = nn.GRU(self.d_hidden_Q + m * in_mult, self.d_hidden_Sigma)

        self.d_hidden_S = n * n
        self.GRU_S = nn.GRU(n * n + 2 * n * in_mult, self.d_hidden_S)

        self.FC1 = nn.Sequential(nn.Linear(self.d_hidden_Sigma, n * n), nn.ReLU())

        d_in_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        d_out_FC2 = n * m
        self.FC2 = nn.Sequential(
            nn.Linear(d_in_FC2, d_in_FC2 * out_mult),
            nn.ReLU(),
            nn.Linear(d_in_FC2 * out_mult, d_out_FC2),
        )
        self.FC3 = nn.Sequential(nn.Linear(self.d_hidden_S + d_out_FC2, m * m), nn.ReLU())
        self.FC4 = nn.Sequential(nn.Linear(self.d_hidden_Sigma + m * m, self.d_hidden_Sigma), nn.ReLU())
        self.FC5 = nn.Sequential(nn.Linear(m, m * in_mult), nn.ReLU())
        self.FC6 = nn.Sequential(nn.Linear(m, m * in_mult), nn.ReLU())
        self.FC7 = nn.Sequential(nn.Linear(2 * n, 2 * n * in_mult), nn.ReLU())

        # A freshly initialized gain matrix has no restoring-force guarantee
        # the way an analytic Kalman gain does -- closed-loop (I-KH)F can
        # easily have spectral radius > 1, so an untrained network diverges
        # from step one. Start the gain near zero (x_upd ~= x_prior, pure CV
        # prediction, which is itself stable) and let training grow it.
        final_linear = self.FC2[-1]
        nn.init.uniform_(final_linear.weight, -1e-3, 1e-3)
        nn.init.zeros_(final_linear.bias)

    def init_hidden(self, batch_size, prior_Q, prior_Sigma, prior_S, device):
        def rep(prior, hidden_dim):
            return prior.flatten().reshape(1, 1, hidden_dim).repeat(1, batch_size, 1).to(device)
        return (
            rep(prior_Q, self.d_hidden_Q),
            rep(prior_Sigma, self.d_hidden_Sigma),
            rep(prior_S, self.d_hidden_S),
        )

    def forward(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff, hidden):
        """All four *_diff args: [batch, dim]. hidden: (h_Q, h_Sigma, h_S),
        each [1, batch, hidden_dim]. Returns (gain [batch, m, n], new hidden)."""
        h_Q, h_Sigma, h_S = hidden
        obs_diff = obs_diff.unsqueeze(0)
        obs_innov_diff = obs_innov_diff.unsqueeze(0)
        fw_evol_diff = fw_evol_diff.unsqueeze(0)
        fw_update_diff = fw_update_diff.unsqueeze(0)

        out_FC5 = self.FC5(fw_update_diff)
        out_Q, h_Q = self.GRU_Q(out_FC5, h_Q)

        out_FC6 = self.FC6(fw_evol_diff)
        in_Sigma = torch.cat((out_Q, out_FC6), dim=-1)
        out_Sigma, h_Sigma = self.GRU_Sigma(in_Sigma, h_Sigma)

        out_FC1 = self.FC1(out_Sigma)

        in_FC7 = torch.cat((obs_diff, obs_innov_diff), dim=-1)
        out_FC7 = self.FC7(in_FC7)

        in_S = torch.cat((out_FC1, out_FC7), dim=-1)
        out_S, h_S = self.GRU_S(in_S, h_S)

        in_FC2 = torch.cat((out_Sigma, out_S), dim=-1)
        out_FC2 = self.FC2(in_FC2)

        in_FC3 = torch.cat((out_S, out_FC2), dim=-1)
        out_FC3 = self.FC3(in_FC3)

        in_FC4 = torch.cat((out_Sigma, out_FC3), dim=-1)
        out_FC4 = self.FC4(in_FC4)
        h_Sigma = out_FC4  # backward flow: overwrite Sigma-GRU's hidden state

        gain_flat = out_FC2.squeeze(0)  # [batch, n*m]
        gain = gain_flat.view(-1, self.m, self.n)
        return gain, (h_Q, h_Sigma, h_S)
