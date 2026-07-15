"""Exact continuous-to-discrete LTI conversion via Van Loan's method, so every
motion model's (F, Q) comes from one checkable derivation instead of
hand-copied closed-form formulas per model.

Given xdot = A x + G w, w continuous white noise with PSD Qc:
    M = dt * [[-A, G Qc G^T], [0, A^T]]
    expm(M) = [[Phi11, Phi12], [0, Phi22]]
    F = Phi22^T,  Q = F @ Phi12
"""

import numpy as np
from scipy.linalg import expm


def van_loan_discretize(A, G, Qc, dt):
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = G @ Qc @ G.T
    M[n:, n:] = A.T
    phi = expm(M * dt)
    F = phi[n:, n:].T
    Q = F @ phi[:n, n:]
    return F, Q
