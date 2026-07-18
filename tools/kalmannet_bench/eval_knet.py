#!/usr/bin/env python3
"""Evaluate the trained vanilla KalmanNet checkpoint on the same test.npz
(all mc realizations) and robustness-sweep sets the classical filters are
judged on, so its number in the results table is directly comparable. No
covariance is reported (see REPORT.md sec. 1.3 -- state dim 9 > measurement
dim 3, so vanilla KalmanNet has no recoverable Sigma): RMSE only.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from filters.knet.gain_network import KalmanGainGRU
from filters.knet.knet_filter import KalmanNetFilter
from filters.motion_models import cv_model
from filters.state import STATE_DIM

_INIT_OFFSET = torch.tensor([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
_BURN_IN = 20


def _priors(dt, R):
    _, Q = cv_model(dt)
    return (
        torch.tensor(Q, dtype=torch.float64),
        torch.eye(STATE_DIM, dtype=torch.float64) * 100.0,
        torch.tensor(R, dtype=torch.float64),
    )


def position_velocity_rmse(net, dt, R, states, measurements):
    """states: [N,T,9], measurements: [N,mc,T,3]. Returns (pos_rmse, vel_rmse)
    over all trajectories and mc realizations, burn-in excluded."""
    N, mc, T, _ = measurements.shape
    prior_Q, prior_Sigma, prior_S = _priors(dt, R)
    pos_sq, vel_sq, count = 0.0, 0.0, 0

    with torch.no_grad():
        for i in range(N):
            x_true = torch.tensor(states[i], dtype=torch.float64).unsqueeze(0)
            for m in range(mc):
                z = torch.tensor(measurements[i, m], dtype=torch.float64).unsqueeze(0)
                flt = KalmanNetFilter(net, dt)
                x0 = x_true[:, 0, :] + _INIT_OFFSET
                flt.init_sequence(x0, prior_Q, prior_Sigma, prior_S)
                for t in range(T):
                    x_upd = flt.step(z[:, t, :])
                    if t >= _BURN_IN:
                        err = (x_upd[0] - x_true[0, t]).numpy()
                        err = np.nan_to_num(err, nan=1e5, posinf=1e5, neginf=-1e5)
                        pos_sq += np.sum(err[:3] ** 2)
                        vel_sq += np.sum(err[3:6] ** 2)
                        count += 1
    return float(np.sqrt(pos_sq / count)), float(np.sqrt(vel_sq / count))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/splits")
    ap.add_argument("--checkpoint", default="results/knet_vanilla/best.pt")
    ap.add_argument("--out", default="results/knet_test_rmse.csv")
    args = ap.parse_args()

    torch.set_num_threads(1)
    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())

    net = KalmanGainGRU().double()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    net.eval()

    rows = []
    for regime_name in manifest["regimes"]:
        path = data_dir / ("test.npz" if regime_name == "nominal" else f"test_{regime_name}.npz")
        if not path.exists():
            continue
        d = np.load(path)
        states, measurements = d["states"], d["measurements"]
        r = manifest["regimes"][regime_name]
        R = np.diag([r["sigma_az_rad"] ** 2, r["sigma_el_rad"] ** 2, r["sigma_range_m"] ** 2])
        dt = r["dt"]

        pos_rmse, vel_rmse = position_velocity_rmse(net, dt, R, states, measurements)
        print(f"{regime_name}: pos_rmse={pos_rmse:.2f}m vel_rmse={vel_rmse:.2f}m/s", flush=True)
        rows.append({"filter": "KalmanNet-vanilla", "regime": regime_name, "pos_rmse": pos_rmse, "vel_rmse": vel_rmse})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filter", "regime", "pos_rmse", "vel_rmse"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
