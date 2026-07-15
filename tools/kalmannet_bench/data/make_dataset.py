#!/usr/bin/env python3
"""Generate train/val/test synthetic maneuvering-target trajectory splits and
a manifest.json recording every generation parameter, per the brief's
reproducibility requirement. Train/val/test use disjoint seeds (via
np.random.SeedSequence spawning, for genuinely independent streams -- not
just different-looking ones). Only the nominal regime is used for
train/val/test; the shifted regimes are generated test-only, for the
robustness sweep, and are never touched during IMM tuning or KalmanNet
training.
"""

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from data.config import REGIMES, NOMINAL
from data.trajectory_gen import generate_ground_truth, measure


def _git_commit():
    """GIT_COMMIT (set by docker_run.sh from the host, where .git is
    visible) takes precedence -- the container only mounts
    tools/kalmannet_bench, so a local `git rev-parse` here would silently
    resolve to nothing and every manifest would say "unknown"."""
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit:
        return env_commit
    try:
        root = Path(__file__).resolve().parents[3]
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    except Exception:
        return "unknown"


def generate_split(seed, n_traj, mc_per_traj, regime):
    parent = np.random.SeedSequence(seed)
    trajectories = []
    for traj_seed in parent.spawn(n_traj):
        child_seeds = traj_seed.spawn(1 + mc_per_traj)
        traj_rng = np.random.default_rng(child_seeds[0])
        states, labels = generate_ground_truth(traj_rng, regime)
        measurements = np.stack([
            measure(np.random.default_rng(child_seeds[1 + k]), states, regime)
            for k in range(mc_per_traj)
        ])
        trajectories.append({"states": states, "labels": labels, "measurements": measurements})
    return trajectories


def save_split(path, trajectories):
    np.savez_compressed(
        path,
        states=np.stack([t["states"] for t in trajectories]),
        measurements=np.stack([t["measurements"] for t in trajectories]),
        labels=np.stack([t["labels"] for t in trajectories]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-sweep", type=int, default=50, help="trajectories per shifted regime")
    ap.add_argument("--mc-test", type=int, default=5, help="noise realizations per test trajectory")
    ap.add_argument("--n-highmc", type=int, default=5, help="trajectories in the high-MC ANEES/NIS subset")
    ap.add_argument("--mc-highmc", type=int, default=100, help="noise realizations per high-MC trajectory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    save_split(out / "train.npz", generate_split(args.seed + 1, args.n_train, 1, NOMINAL))
    save_split(out / "val.npz", generate_split(args.seed + 2, args.n_val, 1, NOMINAL))
    save_split(out / "test.npz", generate_split(args.seed + 3, args.n_test, args.mc_test, NOMINAL))

    # Separate from the 200-trajectory test set used for RMSE confidence
    # intervals: few trajectories, MANY noise realizations each, so ANEES's
    # chi-square band (df = N*dim, N = MC count) is tight enough to actually
    # discriminate a well-calibrated filter from an overconfident one.
    save_split(out / "test_highmc.npz", generate_split(args.seed + 4, args.n_highmc, args.mc_highmc, NOMINAL))

    for idx, (name, regime) in enumerate(REGIMES.items()):
        if name == "nominal":
            continue
        sweep = generate_split(args.seed + 100 + idx, args.n_sweep, args.mc_test, regime)
        save_split(out / f"test_{name}.npz", sweep)

    manifest = {
        "seed": args.seed,
        "git_commit": _git_commit(),
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "n_sweep": args.n_sweep,
        "mc_test": args.mc_test,
        "n_highmc": args.n_highmc,
        "mc_highmc": args.mc_highmc,
        "motion_labels": {"cv": 0, "ct": 1, "ca": 2},
        "regimes": {name: asdict(r) for name, r in REGIMES.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote splits + manifest to {out}")


if __name__ == "__main__":
    main()
