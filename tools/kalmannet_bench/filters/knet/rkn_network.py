"""Recursive KalmanNet (Mortada, Falcon, Kahil, Clavaud, Michel, EUSIPCO 2025,
arXiv:2506.11639) -- the consistency-corrected KalmanNet variant, adapted
from the official reference (`github.com/ixblue/RecursiveKalmanNet`,
`Algo/RecursiveKalmanNet.py`) onto our az/el/range measurement model. Two
independent GRU networks, matching the reference's `GRUNetwork` structure
(FC pre-layer -> GRU -> FC post-layer -> linear output):

  - rnn_K:   outputs the flattened Kalman gain, K in R^{m x n}.
  - rnn_cov: outputs the p = m(m+1)/2 free lower-triangular entries of a
             Cholesky factor L; B_t = L_t L_t^T is the learned half of the
             Joseph-form covariance (NOTES.md sec. 2). The other half, A_t
             (the propagated (I-K_tH_t)F_t P_{t-1} F_t^T (I-K_tH_t)^T term),
             is computed in closed form -- this is what distinguishes RKN
             from vanilla KalmanNet, Split-KalmanNet, and Cholesky-KalmanNet.

Both networks share the same 4 input features (innovation, observation
difference, Jacobian H_t, state correction), squared elementwise -- per the
reference and NOTES.md's extraction ("covariance is a second-order/quadratic
quantity").
"""

import torch
import torch.nn as nn

from filters.state import MEAS_DIM, STATE_DIM

_M, _N = STATE_DIM, MEAS_DIM
_FEATURE_DIM = 2 * _N + _M * _N + _M  # y, z_diff, H_t (flattened), correction


class GRUNetwork(nn.Module):
    """FC1 -> GRU -> FC2 -> linear output, matching the reference
    `Algo.RecursiveKalmanNet.GRUNetwork` (one FC layer each side, default
    multipliers 10/10/20 -- kept faithful to the reference's defaults)."""

    def __init__(self, output_size, fc1_mult=10, hidden_mult=10, fc2_mult=20):
        super().__init__()
        self.output_size = output_size
        hidden_size = output_size * hidden_mult

        self.fc1 = nn.Sequential(nn.Linear(_FEATURE_DIM, _FEATURE_DIM * fc1_mult), nn.ReLU())
        self.gru = nn.GRU(_FEATURE_DIM * fc1_mult, hidden_size)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * fc2_mult),
            nn.ReLU(),
            nn.Linear(hidden_size * fc2_mult, output_size),
        )
        self.hidden_size = hidden_size

        # Same rationale as vanilla KalmanNet's gain network (see
        # gain_network.py): a fresh recurrent gain/covariance output has no
        # stability guarantee, so start near zero (K~0 => trust the CV
        # prediction; L~0 => P starts as just the closed-form A_t term).
        final_linear = self.fc2[-1]
        nn.init.uniform_(final_linear.weight, -1e-3, 1e-3)
        nn.init.zeros_(final_linear.bias)

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, dtype=torch.float64, device=device)

    def forward(self, features, hidden):
        """features: [batch, feature_dim]. Returns (output [batch,
        output_size], new_hidden)."""
        x = features.unsqueeze(0)  # [1, batch, feature_dim], seq-first for GRU
        x = self.fc1(x)
        x, hidden = self.gru(x, hidden)
        x = self.fc2(x)
        return x.squeeze(0), hidden


class RKNGainCovNetwork(nn.Module):
    def __init__(self, m=STATE_DIM, n=MEAS_DIM, fc1_mult=10, hidden_mult=10, fc2_mult=20):
        """Multiplier defaults (10, 10, 20) match the reference exactly. At
        our m=9 (the reference's own examples use smaller m), that default
        sizing produces a ~7.8M-parameter network whose forward pass alone
        measures ~3.3 ms/step single-threaded -- already over the brief's
        2 ms budget before any accelerator is involved (see REPORT.md sec.
        3). Pass smaller multipliers for a latency-competitive trained
        variant; the reference-default size is reported separately as a
        deployment-audit data point, not silently discarded."""
        super().__init__()
        self.m, self.n = m, n
        self.rnn_K = GRUNetwork(output_size=m * n, fc1_mult=fc1_mult, hidden_mult=hidden_mult, fc2_mult=fc2_mult)
        self.rnn_cov = GRUNetwork(output_size=m * (m + 1) // 2, fc1_mult=fc1_mult, hidden_mult=hidden_mult, fc2_mult=fc2_mult)
        self._tril_rows, self._tril_cols = torch.tril_indices(m, m)

    def init_hidden(self, batch_size, device):
        return self.rnn_K.init_hidden(batch_size, device), self.rnn_cov.init_hidden(batch_size, device)

    def forward(self, features, hidden):
        h_K, h_cov = hidden
        K_flat, h_K = self.rnn_K(features, h_K)
        C, h_cov = self.rnn_cov(features, h_cov)

        batch = features.shape[0]
        K = K_flat.view(batch, self.m, self.n)

        L = torch.zeros(batch, self.m, self.m, dtype=features.dtype, device=features.device)
        L[:, self._tril_rows, self._tril_cols] = C
        B = L @ L.transpose(1, 2)  # PSD by construction for any real L

        return K, B, (h_K, h_cov)
