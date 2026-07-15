"""Validate van_loan_discretize against the textbook closed-form discrete
white-noise-acceleration (DWNA) model (Bar-Shalom, Li & Kirubarajan,
"Estimation with Applications to Tracking and Navigation," sec. 6.2) before
trusting it for the Singer/CT motion models built on top of it."""

import numpy as np

from filters.discretize import van_loan_discretize


def test_matches_dwna_closed_form():
    dt, q = 0.25, 3.0
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    G = np.array([[0.0], [1.0]])
    Qc = np.array([[q]])

    F, Q = van_loan_discretize(A, G, Qc, dt)

    F_expected = np.array([[1.0, dt], [0.0, 1.0]])
    Q_expected = q * np.array([
        [dt**3 / 3, dt**2 / 2],
        [dt**2 / 2, dt],
    ])

    assert np.allclose(F, F_expected)
    assert np.allclose(Q, Q_expected)


def test_discrete_Q_is_symmetric_positive_semidefinite():
    dt = 0.02
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -0.2]])
    G = np.array([[0.0], [0.0], [1.0]])
    Qc = np.array([[1.5]])

    _, Q = van_loan_discretize(A, G, Qc, dt)

    assert np.allclose(Q, Q.T)
    eigvals = np.linalg.eigvalsh(Q)
    assert np.all(eigvals > -1e-12)
