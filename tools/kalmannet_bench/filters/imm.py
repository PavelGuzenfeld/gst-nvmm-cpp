"""Interacting Multiple Model filter (Blom & Bar-Shalom 1988; standard
mixing / mode-matched-filtering / combination cycle as in Bar-Shalom, Li &
Kirubarajan, "Estimation with Applications to Tracking and Navigation," ch.
11) over four modes: CV, CT+, CT- (symmetric turn-rate hypotheses -- the
filter's CT mode assumes a fixed nominal |omega|, not the true per-segment
turn rate, which is randomized in the ground-truth generator; this
model/reality mismatch is deliberate and feeds the robustness sweep), and CA
(Singer). Each mode runs the same az/el/range EKF update as the single-model
sanity floor, so the only extra machinery is the mixing/combination steps.
"""

import numpy as np

from filters.measurement_model import H_jacobian, innovation
from filters.motion_models import ca_model, ct_model, cv_model
from filters.state import MEAS_DIM, STATE_DIM


def _gaussian_log_likelihood(y, S):
    m = y.shape[0]
    _, logdet = np.linalg.slogdet(S)
    return -0.5 * (m * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(S, y))


class IMM:
    def __init__(self, dt, R, omega_ct=np.deg2rad(15.0), p_stay=0.97,
                 cv_sigma_a=0.05, ct_sigma_a=0.3, ca_sigma_a=3.0, ca_tau=4.0):
        self.R = R
        self.mode_names = ["cv", "ct_pos", "ct_neg", "ca"]
        self.FQ = [
            cv_model(dt, sigma_a=cv_sigma_a),
            ct_model(dt, omega=omega_ct, sigma_a=ct_sigma_a),
            ct_model(dt, omega=-omega_ct, sigma_a=ct_sigma_a),
            ca_model(dt, sigma_a=ca_sigma_a, tau=ca_tau),
        ]
        r = len(self.mode_names)
        # tpm[i, j] = P(mode j at k | mode i at k-1). High self-transition
        # probability: mode switches are slow relative to the 50 Hz cycle.
        self.tpm = np.full((r, r), (1 - p_stay) / (r - 1))
        np.fill_diagonal(self.tpm, p_stay)

    @property
    def n_modes(self):
        return len(self.mode_names)

    def init_state(self, x0, P0):
        r = self.n_modes
        return {
            "x": [x0.copy() for _ in range(r)],
            "P": [P0.copy() for _ in range(r)],
            "mu": np.full(r, 1.0 / r),
        }

    def step(self, state, z):
        r = self.n_modes
        x_prev, P_prev, mu_prev = state["x"], state["P"], state["mu"]

        # 1. Mixing / interaction.
        c = self.tpm.T @ mu_prev  # predicted mode probabilities c[j]
        mix_w = (self.tpm * mu_prev[:, None]) / c[None, :]  # mix_w[i, j] = mu_{i|j}

        x0, P0 = [], []
        for j in range(r):
            xj0 = sum(mix_w[i, j] * x_prev[i] for i in range(r))
            Pj0 = np.zeros((STATE_DIM, STATE_DIM))
            for i in range(r):
                d = x_prev[i] - xj0
                Pj0 += mix_w[i, j] * (P_prev[i] + np.outer(d, d))
            x0.append(xj0)
            P0.append(Pj0)

        # 2. Mode-matched EKF predict + update.
        x_new, P_new, log_likelihoods = [], [], []
        y_list, S_list = [], []
        for j in range(r):
            F, Q = self.FQ[j]
            x_pred = F @ x0[j]
            P_pred = F @ P0[j] @ F.T + Q
            H = H_jacobian(x_pred)
            y = innovation(z, x_pred)
            S = H @ P_pred @ H.T + self.R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_upd = x_pred + K @ y
            I_KH = np.eye(STATE_DIM) - K @ H
            P_upd = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
            x_new.append(x_upd)
            P_new.append(P_upd)
            log_likelihoods.append(_gaussian_log_likelihood(y, S))
            y_list.append(y)
            S_list.append(S)

        # 3. Mode-probability update, in log-space (softmax over
        # log-likelihood + log prior). A large outlier innovation --
        # entirely plausible under the heavy-tailed-noise robustness-sweep
        # regime -- can drive every mode's raw Gaussian likelihood to
        # literally 0.0 in floating point (exp() underflow), which would
        # make mu = likelihoods*c sum to zero and divide-by-zero into NaN
        # that then poisons every subsequent step. Subtracting the max
        # log-weight before exponentiating keeps this numerically stable
        # regardless of how extreme the innovation is.
        log_weights = np.array(log_likelihoods) + np.log(np.maximum(c, 1e-300))
        log_weights -= np.max(log_weights)
        mu = np.exp(log_weights)
        mu /= np.sum(mu)

        # 4. Combination (output estimate).
        x_out = sum(mu[j] * x_new[j] for j in range(r))
        P_out = np.zeros((STATE_DIM, STATE_DIM))
        for j in range(r):
            d = x_new[j] - x_out
            P_out += mu[j] * (P_new[j] + np.outer(d, d))

        # IMM has no single native innovation/S (each mode predicts its own);
        # for NIS reporting, combine the same way state/covariance are
        # combined above -- probability-weighted mean and spread-of-means.
        y_out = sum(mu[j] * y_list[j] for j in range(r))
        S_out = np.zeros((MEAS_DIM, MEAS_DIM))
        for j in range(r):
            dy = y_list[j] - y_out
            S_out += mu[j] * (S_list[j] + np.outer(dy, dy))

        return x_out, P_out, {"x": x_new, "P": P_new, "mu": mu}, y_out, S_out
