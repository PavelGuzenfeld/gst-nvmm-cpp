#!/usr/bin/env python3
"""Reference KalmanFilter for KalmanBox (gst/common/kalman_box.hpp): reproduces
SAMURAI's kalman_filter.py math (SORT/ByteTrack lineage, adapted to w,h in
place of aspect ratio). Prints the mean/cov_diag values that
tests/test_kalman_box.cpp checks against — rerun after touching the port.
"""
import numpy as np

STD_WEIGHT_POSITION = 1.0 / 20
STD_WEIGHT_VELOCITY = 1.0 / 160


class KalmanFilterBox:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

    def initiate(self, cx, cy, w, h):
        mean = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        std = [
            2 * STD_WEIGHT_POSITION * w, 2 * STD_WEIGHT_POSITION * h,
            2 * STD_WEIGHT_POSITION * w, 2 * STD_WEIGHT_POSITION * h,
            10 * STD_WEIGHT_VELOCITY * w, 10 * STD_WEIGHT_VELOCITY * h,
            10 * STD_WEIGHT_VELOCITY * w, 10 * STD_WEIGHT_VELOCITY * h]
        self.mean = mean
        self.covariance = np.diag(np.square(std))

    def predict(self, dt):
        w, h = self.mean[2], self.mean[3]
        std = [
            STD_WEIGHT_POSITION * w, STD_WEIGHT_POSITION * h,
            STD_WEIGHT_POSITION * w, STD_WEIGHT_POSITION * h,
            STD_WEIGHT_VELOCITY * w, STD_WEIGHT_VELOCITY * h,
            STD_WEIGHT_VELOCITY * w, STD_WEIGHT_VELOCITY * h]
        motion_cov = np.diag(np.square(std))
        motion_mat = self._motion_mat.copy()
        motion_mat[0:4, 4:8] = np.eye(4) * dt
        self.mean = motion_mat @ self.mean
        self.covariance = motion_mat @ self.covariance @ motion_mat.T + motion_cov

    def _project(self):
        w, h = self.mean[2], self.mean[3]
        std = [STD_WEIGHT_POSITION * w, STD_WEIGHT_POSITION * h,
               STD_WEIGHT_POSITION * w, STD_WEIGHT_POSITION * h]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ self.mean
        covariance = self._update_mat @ self.covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, cx, cy, w, h):
        projected_mean, projected_cov = self._project()
        kalman_gain = np.linalg.solve(
            projected_cov, (self.covariance @ self._update_mat.T).T).T
        innovation = np.array([cx, cy, w, h]) - projected_mean
        self.mean = self.mean + kalman_gain @ innovation
        self.covariance = (self.covariance
                            - kalman_gain @ projected_cov @ kalman_gain.T)

    def gating_distance(self, cx, cy, w, h):
        projected_mean, projected_cov = self._project()
        d = np.array([cx, cy, w, h]) - projected_mean
        z = np.linalg.solve(np.linalg.cholesky(projected_cov), d)
        return float(z @ z)


def _tag(name, mean, cov):
    print(f"{name} mean     = {np.array2string(mean, precision=10, separator=', ')}")
    print(f"{name} cov_diag = {np.array2string(np.diag(cov), precision=10, separator=', ')}")


def main():
    kf = KalmanFilterBox()

    kf.initiate(100, 200, 14, 8)
    _tag("INIT", kf.mean, kf.covariance)

    kf.predict(1.0)
    _tag("PRED1", kf.mean, kf.covariance)

    kf.update(110, 205, 15, 9)
    _tag("UPD", kf.mean, kf.covariance)

    kf.predict(2.0)
    _tag("PRED2", kf.mean, kf.covariance)

    gd = kf.gating_distance(130, 212, 16, 9)
    print(f"GATING distance = {gd:.10g}")


if __name__ == "__main__":
    main()
