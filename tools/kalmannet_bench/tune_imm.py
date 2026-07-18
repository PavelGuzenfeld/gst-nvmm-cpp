#!/usr/bin/env python3
"""Grid search for IMM hyperparameters (transition-matrix self-stay
probability, nominal CT turn rate, CA/Singer sigma_a and tau) on the
validation set ONLY -- an under-tuned IMM is a strawman baseline, so this is
real tuning effort, not a rubber stamp. Every candidate and the final choice
are recorded to results/imm_tuning.json; the test set is never touched here.
"""

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

from filters.imm import IMM


def _load_split(path):
    d = np.load(path)
    return d["states"], d["measurements"]


def run_imm(states, measurements, dt, R, **imm_kwargs):
    """states: [N,T,9], measurements: [N,1,T,3] (single MC realization).
    Mean position RMSE over steps 20..T, skipping the initial-condition
    transient."""
    imm = IMM(dt, R, **imm_kwargs)
    N, T, _ = states.shape
    all_err = []
    for i in range(N):
        x0 = states[i, 0].copy()
        x0[3:] += np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        P0 = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        state = imm.init_state(x0, P0)
        errs = np.zeros(T)
        for t in range(T):
            x_out, _, state, _, _ = imm.step(state, measurements[i, 0, t])
            errs[t] = np.linalg.norm(x_out[:3] - states[i, t, :3])
        all_err.append(errs[20:])
    all_err = np.concatenate(all_err)
    return float(np.sqrt(np.mean(all_err ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/splits")
    ap.add_argument("--out", default="results/imm_tuning.json")
    ap.add_argument("--n-search", type=int, default=30, help="val trajectories used during grid search")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    dt = manifest["regimes"]["nominal"]["dt"]
    R = np.diag([
        manifest["regimes"]["nominal"]["sigma_az_rad"] ** 2,
        manifest["regimes"]["nominal"]["sigma_el_rad"] ** 2,
        manifest["regimes"]["nominal"]["sigma_range_m"] ** 2,
    ])

    val_states, val_meas = _load_split(data_dir / "val.npz")
    search_states, search_meas = val_states[: args.n_search], val_meas[: args.n_search]

    p_stay_grid = [0.90, 0.95, 0.97, 0.99]
    omega_ct_deg_grid = [10.0, 15.0, 20.0, 25.0]
    ca_sigma_a_grid = [2.0, 4.0, 6.0]
    ca_tau_grid = [2.0, 4.0]

    candidates = list(itertools.product(p_stay_grid, omega_ct_deg_grid, ca_sigma_a_grid, ca_tau_grid))
    print(f"grid search over {len(candidates)} candidates, {args.n_search} val trajectories each", flush=True)

    results = []
    t0 = time.time()
    for i, (p_stay, omega_deg, ca_sigma_a, ca_tau) in enumerate(candidates):
        rmse = run_imm(
            search_states, search_meas, dt, R,
            omega_ct=np.deg2rad(omega_deg), p_stay=p_stay,
            ca_sigma_a=ca_sigma_a, ca_tau=ca_tau,
        )
        results.append({
            "p_stay": p_stay, "omega_ct_deg": omega_deg,
            "ca_sigma_a": ca_sigma_a, "ca_tau": ca_tau, "val_pos_rmse": rmse,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(candidates)} done, {time.time() - t0:.0f}s elapsed", flush=True)

    results.sort(key=lambda r: r["val_pos_rmse"])
    best = results[0]
    print(f"search took {time.time() - t0:.1f}s")
    print(f"best (search subset): {best}")

    full_rmse = run_imm(
        val_states, val_meas, dt, R,
        omega_ct=np.deg2rad(best["omega_ct_deg"]), p_stay=best["p_stay"],
        ca_sigma_a=best["ca_sigma_a"], ca_tau=best["ca_tau"],
    )
    print(f"best config, full val set ({val_states.shape[0]} traj) RMSE: {full_rmse:.2f} m")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "n_search_trajectories": args.n_search,
        "n_full_val_trajectories": int(val_states.shape[0]),
        "grid": {
            "p_stay": p_stay_grid, "omega_ct_deg": omega_ct_deg_grid,
            "ca_sigma_a": ca_sigma_a_grid, "ca_tau": ca_tau_grid,
        },
        "all_candidates": results,
        "best": best,
        "best_full_val_pos_rmse": full_rmse,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
