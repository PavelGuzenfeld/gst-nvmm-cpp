"""Single-model Extended Kalman Filter: constant-velocity process, nonlinear
az/el/range measurement. The brief's sanity floor -- every other filter (IMM,
KalmanNet, consistency variants) has to beat this to justify its complexity.
"""

import numpy as np

from filters.measurement_model import H_jacobian, innovation
from filters.motion_models import cv_model
from filters.state import STATE_DIM


class EKF:
    def __init__(self, dt, R, sigma_a=0.05, tau=0.05):
        self.F, self.Q = cv_model(dt, sigma_a=sigma_a, tau=tau)
        self.R = R

    def predict(self, x, P):
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z):
        H = H_jacobian(x_pred)
        y = innovation(z, x_pred)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_upd = x_pred + K @ y
        # Joseph form: guarantees a symmetric, PSD P_upd regardless of
        # numerical error in K, which matters here because P_upd's validity
        # is the exact thing NEES/NIS are testing.
        I_KH = np.eye(STATE_DIM) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        return x_upd, P_upd, y, S

    def step(self, x, P, z):
        x_pred, P_pred = self.predict(x, P)
        return self.update(x_pred, P_pred, z)
