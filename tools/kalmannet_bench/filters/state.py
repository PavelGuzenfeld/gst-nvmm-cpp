"""Shared state-vector convention for every filter in this benchmark.

x = [px, py, pz, vx, vy, vz, ax, ay, az], sensor at the origin. All filters
(EKF, UKF, IMM's per-mode filters, KalmanNet, consistency variants) use this
same 9-dim layout so state, covariance, and metric code are directly
comparable across methods -- per the brief's fairness requirement.
"""

STATE_DIM = 9

PX, PY, PZ = 0, 1, 2
VX, VY, VZ = 3, 4, 5
A_X, A_Y, A_Z = 6, 7, 8

POS = slice(0, 3)
VEL = slice(3, 6)
ACC = slice(6, 9)

# Measurement-space indices. Named AZIMUTH (not AZ) to avoid colliding with
# the state's A_Z acceleration component above.
MEAS_DIM = 3
AZIMUTH, ELEVATION, RANGE = 0, 1, 2
