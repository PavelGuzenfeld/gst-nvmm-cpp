#!/usr/bin/env python3
"""Truncated-BPTT training for vanilla KalmanNet (architecture #2), with
validation-based early stopping -- per NOTES.md sec. 1(b): per-step squared
error, time-averaged per trajectory (via the truncated-chunk mean), batch-
averaged, with L2 weight decay via Adam's `weight_decay` (not the reference
repo's optional, non-default composition loss). Train/val use only the
single (mc=0) noise realization generated for those splits; the test set's
extra Monte Carlo realizations are for evaluation only and are never touched
here.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from filters.knet.gain_network import KalmanGainGRU
from filters.knet.knet_filter import KalmanNetFilter
from filters.motion_models import cv_model
from filters.state import STATE_DIM

_INIT_PERTURB_STD = torch.tensor([30.0, 30.0, 30.0, 3.0, 3.0, 3.0, 0.5, 0.5, 0.5], dtype=torch.float64)

# Position (~km), velocity (~10 m/s), and acceleration (~1 m/s^2) live at very
# different physical scales. A raw sum-of-squared-error loss is numerically
# dominated by position alone, starving velocity/acceleration of gradient
# signal -- which indirectly hurts position too, since accurate position
# *prediction* depends on the network learning a good velocity correction.
# Weight each group by its inverse-variance (using _INIT_PERTURB_STD as the
# reference scale) so all three contribute comparably, the same normalization
# a Mahalanobis/NLL-style loss would apply automatically.
_LOSS_WEIGHT = (1.0 / _INIT_PERTURB_STD ** 2)


def _load_split(path):
    d = np.load(path)
    return d["states"], d["measurements"]


def _priors(dt, R):
    _, Q = cv_model(dt)
    prior_Q = torch.tensor(Q, dtype=torch.float64)
    prior_Sigma = torch.eye(STATE_DIM, dtype=torch.float64) * 100.0
    prior_S = torch.tensor(R, dtype=torch.float64)
    return prior_Q, prior_Sigma, prior_S


def run_epoch(net, dt, states, measurements, R, batch_size, trunc_len, rng, optimizer=None):
    """One pass over the dataset. With an optimizer: trains via truncated
    BPTT, detaching recurrent state at each chunk boundary. Without: a
    plain no_grad forward pass returning per-step position errors, for
    validation-time RMSE."""
    N, T, _ = states.shape
    training = optimizer is not None
    order = rng.permutation(N) if training else np.arange(N)
    prior_Q, prior_Sigma, prior_S = _priors(dt, R)

    total_loss, total_chunks = 0.0, 0
    all_errors = []

    for start in range(0, N, batch_size):
        idx = order[start:start + batch_size]
        x_true = torch.tensor(states[idx], dtype=torch.float64)       # [b, T, 9]
        z = torch.tensor(measurements[idx, 0], dtype=torch.float64)   # [b, T, 3]

        x0 = x_true[:, 0, :] + torch.randn(len(idx), STATE_DIM, dtype=torch.float64) * _INIT_PERTURB_STD

        flt = KalmanNetFilter(net, dt)
        flt.init_sequence(x0, prior_Q, prior_Sigma, prior_S)

        if not training:
            # No ground-truth reseed here: EKF/UKF/IMM never get one (they
            # don't diverge), so injecting truth into KalmanNet's state
            # would grade it on an easier protocol and mask exactly the
            # failure mode (divergence on maneuvers) this benchmark exists
            # to surface. nan_to_num is a pure numerical safety net -- it
            # only replaces actual inf/nan so one overflowed trajectory
            # doesn't NaN-poison the whole-batch RMSE aggregate -- it does
            # not touch merely-large-but-finite (i.e. genuinely diverged)
            # values.
            with torch.no_grad():
                errs = [flt.step(z[:, t, :]) - x_true[:, t, :] for t in range(T)]
                traj_err = torch.nan_to_num(torch.stack(errs, dim=1), nan=1e5, posinf=1e5, neginf=-1e5)
            all_errors.append(traj_err.numpy())
            continue

        loss_accum = torch.zeros((), dtype=torch.float64)
        chunk_steps = 0
        for t in range(T):
            x_upd = flt.step(z[:, t, :])
            weighted_sq_err = (x_upd - x_true[:, t, :]) ** 2 * _LOSS_WEIGHT
            loss_accum = loss_accum + torch.mean(torch.sum(weighted_sq_err, dim=-1))
            chunk_steps += 1

            if chunk_steps == trunc_len or t == T - 1:
                loss = loss_accum / chunk_steps
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

                # A freshly-updated recurrent gain has no stability guarantee
                # (unlike an analytic Kalman gain) -- an occasional chunk can
                # still diverge before its gradient correction lands. Skip
                # applying a non-finite update rather than let one poisoned
                # chunk corrupt every subsequent step's weights.
                if torch.isfinite(loss) and torch.isfinite(grad_norm):
                    optimizer.step()
                    # Gradient-norm clipping already bounds the actual
                    # weight update regardless of the raw loss magnitude;
                    # this clamp is purely so a rare diverged-chunk's huge
                    # squared error doesn't make the *logged* average
                    # uninterpretable.
                    total_loss += min(loss.item(), 1e6)
                    total_chunks += 1

                flt.detach_state()
                loss_accum = torch.zeros((), dtype=torch.float64)
                chunk_steps = 0

    if training:
        return total_loss / max(total_chunks, 1)
    errors = np.concatenate(all_errors, axis=0)  # [N, T, 9]
    return float(np.sqrt(np.mean(np.sum(errors[..., :3] ** 2, axis=-1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/splits")
    ap.add_argument("--out", default="results/knet_vanilla")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--trunc-len", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    dt = manifest["regimes"]["nominal"]["dt"]
    R = np.diag([
        manifest["regimes"]["nominal"]["sigma_az_rad"] ** 2,
        manifest["regimes"]["nominal"]["sigma_el_rad"] ** 2,
        manifest["regimes"]["nominal"]["sigma_range_m"] ** 2,
    ])

    train_states, train_meas = _load_split(data_dir / "train.npz")
    val_states, val_meas = _load_split(data_dir / "val.npz")

    net = KalmanGainGRU().double()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best_val, best_epoch, patience_left = float("inf"), -1, args.patience
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        train_loss = run_epoch(net, dt, train_states, train_meas, R, args.batch_size, args.trunc_len, rng, optimizer)
        val_rmse = run_epoch(net, dt, val_states, val_meas, R, args.batch_size, args.trunc_len, rng)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_pos_rmse": val_rmse})
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_pos_rmse={val_rmse:.4f}", flush=True)

        if val_rmse < best_val:
            best_val, best_epoch, patience_left = val_rmse, epoch, args.patience
            torch.save(net.state_dict(), out / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch} (best epoch {best_epoch}, val_pos_rmse={best_val:.4f})")
                break

    log = {
        "seed": args.seed,
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "data_manifest_seed": manifest["seed"],
        "hyperparams": vars(args),
        "best_epoch": best_epoch,
        "best_val_pos_rmse": best_val,
        "wall_time_s": time.time() - t0,
        "history": history,
    }
    (out / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"done: best epoch {best_epoch}, val_pos_rmse={best_val:.4f}, wall_time={log['wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
