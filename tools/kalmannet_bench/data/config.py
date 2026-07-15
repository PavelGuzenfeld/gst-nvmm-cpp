"""Regime definitions for synthetic trajectory generation: the nominal
train/val/test regime, and the shifted regimes used only for the test-time
robustness sweep (higher maneuver intensity, higher/lower measurement noise,
heavier-tailed measurement noise) -- per the brief's data spec.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Regime:
    name: str
    dt: float = 0.02                        # 50 Hz, matches the tracker's cyclic rate
    T: int = 750                            # 15 s per trajectory
    seg_len_range: tuple = (50, 250)        # steps per motion segment (1-5 s)
    ct_omega_deg_s_range: tuple = (5.0, 30.0)   # CT turn-rate magnitude; sign randomized
    ca_sigma_a: float = 3.0                 # m/s^2, Singer stationary accel std
    ca_tau: float = 4.0                     # s, Singer correlation time
    cv_sigma_a: float = 0.05                # m/s^2, near-zero -- CV's accel state stays ~0
    sigma_az_rad: float = 0.001             # ~0.06 deg
    sigma_el_rad: float = 0.001
    sigma_range_m: float = 2.0
    heavy_tailed: bool = False              # Student-t measurement noise instead of Gaussian
    heavy_tailed_dof: float = 3.0
    range0: tuple = (500.0, 5000.0)         # initial slant range, m
    speed0: tuple = (20.0, 250.0)           # initial speed, m/s

    def R(self):
        """Measurement noise covariance every filter is told to assume.
        Deliberately the *nominal* Gaussian covariance even under the
        heavy-tailed regime -- the filters are not informed of the shift,
        which is the point of that robustness-sweep entry."""
        return np.diag([self.sigma_az_rad**2, self.sigma_el_rad**2, self.sigma_range_m**2])


NOMINAL = Regime(name="nominal")

HIGH_MANEUVER = Regime(
    name="high_maneuver",
    ct_omega_deg_s_range=(20.0, 60.0),
    ca_sigma_a=8.0,
    ca_tau=2.0,
)

HIGH_NOISE = Regime(
    name="high_noise",
    sigma_az_rad=0.004,
    sigma_el_rad=0.004,
    sigma_range_m=8.0,
)

LOW_NOISE = Regime(
    name="low_noise",
    sigma_az_rad=0.0003,
    sigma_el_rad=0.0003,
    sigma_range_m=0.5,
)

HEAVY_TAILED = Regime(
    name="heavy_tailed",
    heavy_tailed=True,
    heavy_tailed_dof=3.0,
)

REGIMES = {r.name: r for r in (NOMINAL, HIGH_MANEUVER, HIGH_NOISE, LOW_NOISE, HEAVY_TAILED)}
