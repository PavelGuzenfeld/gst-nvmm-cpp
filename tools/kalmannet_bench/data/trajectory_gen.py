"""Piecewise-switching ground-truth trajectory generator (CV/CT/CA segments)
and the az/el/range measurement model applied on top, per the brief's
maneuvering-target data spec. Segment durations, motion kind, and per-segment
parameters (turn rate, Singer accel/tau) are all randomized so the test set
contains maneuvers no filter was tuned to a single fixed instance of.
"""

import numpy as np

from filters.measurement_model import h, wrap_angle
from filters.motion_models import MOTION_LABELS, ca_model, ct_model, cv_model, step
from filters.state import PX, PY, PZ, STATE_DIM, VX, VY, VZ


def _sample_initial_state(rng, regime):
    r0 = rng.uniform(*regime.range0)
    az0 = rng.uniform(-np.pi, np.pi)
    el0 = rng.uniform(np.deg2rad(-10), np.deg2rad(60))
    px = r0 * np.cos(el0) * np.cos(az0)
    py = r0 * np.cos(el0) * np.sin(az0)
    pz = r0 * np.sin(el0)

    speed = rng.uniform(*regime.speed0)
    heading = rng.uniform(-np.pi, np.pi)
    climb = rng.uniform(np.deg2rad(-10), np.deg2rad(10))
    vx = speed * np.cos(climb) * np.cos(heading)
    vy = speed * np.cos(climb) * np.sin(heading)
    vz = speed * np.sin(climb)

    x0 = np.zeros(STATE_DIM)
    x0[PX], x0[PY], x0[PZ] = px, py, pz
    x0[VX], x0[VY], x0[VZ] = vx, vy, vz
    return x0


def _sample_segments(rng, T, regime):
    """List of (length, kind, params) covering all T steps."""
    segments = []
    remaining = T
    while remaining > 0:
        seg_len = int(rng.integers(regime.seg_len_range[0], regime.seg_len_range[1]))
        seg_len = min(seg_len, remaining)
        kind = rng.choice(["cv", "ct", "ca"], p=[0.4, 0.35, 0.25])
        if kind == "ct":
            mag = np.deg2rad(rng.uniform(*regime.ct_omega_deg_s_range))
            sign = rng.choice([-1.0, 1.0])
            params = {"omega": float(sign * mag)}
        elif kind == "ca":
            params = {"sigma_a": regime.ca_sigma_a, "tau": regime.ca_tau}
        else:
            params = {}
        segments.append((seg_len, kind, params))
        remaining -= seg_len
    return segments


def _fq_for(kind, params, dt, regime):
    if kind == "cv":
        return cv_model(dt, sigma_a=regime.cv_sigma_a)
    if kind == "ca":
        return ca_model(dt, sigma_a=params["sigma_a"], tau=params["tau"])
    if kind == "ct":
        return ct_model(dt, omega=params["omega"])
    raise ValueError(kind)


def generate_ground_truth(rng, regime):
    """Returns (states[T, 9] float64, labels[T] int8 in MOTION_LABELS)."""
    T, dt = regime.T, regime.dt
    x = _sample_initial_state(rng, regime)
    states = np.zeros((T, STATE_DIM))
    labels = np.zeros(T, dtype=np.int8)

    t = 0
    for seg_len, kind, params in _sample_segments(rng, T, regime):
        F, Q = _fq_for(kind, params, dt, regime)
        L = np.linalg.cholesky(Q + 1e-12 * np.eye(STATE_DIM))
        label = MOTION_LABELS[kind]
        for _ in range(seg_len):
            if t >= T:
                break
            states[t] = x
            labels[t] = label
            w = L @ rng.standard_normal(STATE_DIM)
            x = step(x, F) + w
            t += 1
    return states, labels


def measure(rng, states, regime):
    """Apply h() plus sensor noise to a ground-truth trajectory. Azimuth is
    re-wrapped after adding noise so a true target sitting near +-pi never
    leaks a measurement outside (-pi, pi]."""
    T = states.shape[0]
    z = np.zeros((T, 3))
    scale = np.array([regime.sigma_az_rad, regime.sigma_el_rad, regime.sigma_range_m])
    for t in range(T):
        z_true = h(states[t])
        if regime.heavy_tailed:
            noise = rng.standard_t(regime.heavy_tailed_dof, size=3) * scale
        else:
            noise = rng.normal(size=3) * scale
        z[t] = z_true + noise
        z[t, 0] = wrap_angle(z[t, 0])
    return z
