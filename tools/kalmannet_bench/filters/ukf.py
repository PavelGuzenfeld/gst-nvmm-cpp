"""Unscented Kalman Filter: same CV process model and az/el/range measurement
as the EKF sanity floor, using sigma points instead of a Jacobian. Azimuth
needs care here too -- the sigma-point measurement mean and every residual
must use circular (wrapped) arithmetic, or sigma points straddling the +-pi
seam corrupt the predicted mean and covariance the same way a naive
innovation would.
"""

import numpy as np

from filters.measurement_model import h, wrap_angle
from filters.motion_models import cv_model
from filters.state import STATE_DIM


class UKF:
    def __init__(self, dt, R, sigma_a=0.05, tau=0.05, alpha=1e-3, beta=2.0, kappa=0.0):
        self.F, self.Q = cv_model(dt, sigma_a=sigma_a, tau=tau)
        self.R = R
        n = STATE_DIM
        self.n = n
        self.lam = alpha**2 * (n + kappa) - n
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + self.lam)))
        self.Wc = np.full(2 * n + 1, 1.0 / (2 * (n + self.lam)))
        self.Wm[0] = self.lam / (n + self.lam)
        self.Wc[0] = self.lam / (n + self.lam) + (1 - alpha**2 + beta)

    def _sigma_points(self, x, P):
        n = self.n
        S = np.linalg.cholesky((n + self.lam) * (P + 1e-12 * np.eye(n)))
        pts = np.zeros((2 * n + 1, n))
        pts[0] = x
        for i in range(n):
            pts[1 + i] = x + S[:, i]
            pts[1 + n + i] = x - S[:, i]
        return pts

    def predict(self, x, P):
        pts = self._sigma_points(x, P)
        pts_pred = pts @ self.F.T
        x_pred = self.Wm @ pts_pred
        diff = pts_pred - x_pred
        P_pred = (diff.T * self.Wc) @ diff + self.Q
        return x_pred, P_pred, pts_pred

    def update(self, x_pred, P_pred, pts_pred, z):
        Z = np.array([h(p) for p in pts_pred])

        # Circular mean for azimuth (column 0); ordinary weighted mean for
        # elevation/range.
        az_mean = np.arctan2(self.Wm @ np.sin(Z[:, 0]), self.Wm @ np.cos(Z[:, 0]))
        z_mean = np.array([az_mean, self.Wm @ Z[:, 1], self.Wm @ Z[:, 2]])

        dz = Z - z_mean
        dz[:, 0] = wrap_angle(dz[:, 0])

        dx = pts_pred - x_pred
        S = (dz.T * self.Wc) @ dz + self.R
        Pxz = (dx.T * self.Wc) @ dz

        K = Pxz @ np.linalg.inv(S)
        y = z - z_mean
        y[0] = wrap_angle(y[0])

        x_upd = x_pred + K @ y
        P_upd = P_pred - K @ S @ K.T
        return x_upd, P_upd, y, S

    def step(self, x, P, z):
        x_pred, P_pred, pts_pred = self.predict(x, P)
        return self.update(x_pred, P_pred, pts_pred, z)
