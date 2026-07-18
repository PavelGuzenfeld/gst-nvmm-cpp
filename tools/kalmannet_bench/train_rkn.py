#!/usr/bin/env python3
"""Truncated-BPTT training for Recursive KalmanNet, with the tuning-free
Gaussian NLL loss (NOTES.md sec. 2, eq. 9; ported from the reference's
`GaussianLikelihoodLoss`, using `slogdet` instead of `log(det(...))` for
numerical stability): L_t = e_t^T P_t^{-1} e_t + logdet(P_t), batch+time-
averaged. Same protocol conventions as train_knet.py for a fair, direct
comparison: truncated BPTT, a finite-loss/finite-gradient skip-guard (never
a ground-truth state injection), validation-based early stopping.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from filters.knet.rkn_filter import RKNFilter
from filters.knet.rkn_network import RKNGainCovNetwork
from filters.state import STATE_DIM

_INIT_PERTURB_STD = torch.tensor([30.0, 30.0, 30.0, 3.0, 3.0, 3.0, 0.5, 0.5, 0.5], dtype=torch.float64)


def _load_split(path):
    d = np.load(path)
    return d["states"], d["measurements"]


def _gaussian_nll(e, P):
    """e: [batch, m], P: [batch, m, m]. Batch-averaged NLL."""
    e_col = e.unsqueeze(-1)
    quad = torch.bmm(e_col.transpose(1, 2), torch.linalg.solve(P, e_col)).squeeze(-1).squeeze(-1)
    _, logdet = torch.linalg.slogdet(P)
    return torch.mean(quad + logdet)


def run_epoch(net, dt, states, measurements, batch_size, trunc_len, rng, optimizer=None):
    N, T, _ = states.shape
    training = optimizer is not None
    order = rng.permutation(N) if training else np.arange(N)

    total_loss, total_chunks = 0.0, 0
    all_errors = []

    for start in range(0, N, batch_size):
        idx = order[start:start + batch_size]
        x_true = torch.tensor(states[idx], dtype=torch.float64)
        z = torch.tensor(measurements[idx, 0], dtype=torch.float64)

        x0 = x_true[:, 0, :] + torch.randn(len(idx), STATE_DIM, dtype=torch.float64) * _INIT_PERTURB_STD
        P0 = torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0).repeat(len(idx), 1, 1) * 100.0

        flt = RKNFilter(net, dt)
        flt.init_sequence(x0, P0)

        if not training:
            # No ground-truth reseed -- same fairness rationale as
            # train_knet.py: EKF/UKF/IMM never get one.
            with torch.no_grad():
                errs = [flt.step(z[:, t, :])[0] - x_true[:, t, :] for t in range(T)]
                traj_err = torch.nan_to_num(torch.stack(errs, dim=1), nan=1e5, posinf=1e5, neginf=-1e5)
            all_errors.append(traj_err.numpy())
            continue

        loss_accum = torch.zeros((), dtype=torch.float64)
        chunk_steps = 0
        for t in range(T):
            x_upd, P_upd = flt.step(z[:, t, :])
            e = x_upd - x_true[:, t, :]
            loss_accum = loss_accum + _gaussian_nll(e, P_upd)
            chunk_steps += 1

            if chunk_steps == trunc_len or t == T - 1:
                loss = loss_accum / chunk_steps
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

                if torch.isfinite(loss) and torch.isfinite(grad_norm):
                    optimizer.step()
                    total_loss += min(loss.item(), 1e6)
                    total_chunks += 1

                flt.detach_state()
                loss_accum = torch.zeros((), dtype=torch.float64)
                chunk_steps = 0

    if training:
        return total_loss / max(total_chunks, 1)
    errors = np.concatenate(all_errors, axis=0)
    return float(np.sqrt(np.mean(np.sum(errors[..., :3] ** 2, axis=-1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/splits")
    ap.add_argument("--out", default="results/rkn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--trunc-len", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--fc1-mult", type=int, default=3)
    ap.add_argument("--hidden-mult", type=int, default=3)
    ap.add_argument("--fc2-mult", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    dt = manifest["regimes"]["nominal"]["dt"]

    train_states, train_meas = _load_split(data_dir / "train.npz")
    val_states, val_meas = _load_split(data_dir / "val.npz")

    net = RKNGainCovNetwork(
        fc1_mult=args.fc1_mult, hidden_mult=args.hidden_mult, fc2_mult=args.fc2_mult,
    ).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best_val, best_epoch, patience_left = float("inf"), -1, args.patience
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        train_loss = run_epoch(net, dt, train_states, train_meas, args.batch_size, args.trunc_len, rng, optimizer)
        val_rmse = run_epoch(net, dt, val_states, val_meas, args.batch_size, args.trunc_len, rng)
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
        "param_count": sum(p.numel() for p in net.parameters()),
        "best_epoch": best_epoch,
        "best_val_pos_rmse": best_val,
        "wall_time_s": time.time() - t0,
        "history": history,
    }
    (out / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"done: best epoch {best_epoch}, val_pos_rmse={best_val:.4f}, wall_time={log['wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
