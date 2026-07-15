#!/usr/bin/env python3
"""Per-step latency benchmark -- batch size 1, single-threaded CPU, >=100
warm-up steps, median/p99 over >=10k steps, per the brief's methodology
(per-step latency is what matters for a 20ms cyclic loop, not batched
throughput). Run this alone, with no other CPU-bound background jobs, or the
numbers are contention artifacts, not real per-step cost.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from filters.ekf import EKF
from filters.imm import IMM
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.knet_filter import KalmanNetFilter
from filters.knet.rkn_filter import RKNFilter
from filters.knet.rkn_network import RKNGainCovNetwork
from filters.state import STATE_DIM
from filters.ukf import UKF


def bench(step_fn, n_warmup=200, n_measure=10000):
    for _ in range(n_warmup):
        step_fn()
    times = np.empty(n_measure)
    for i in range(n_measure):
        t0 = time.perf_counter()
        step_fn()
        times[i] = time.perf_counter() - t0
    return {
        "median_ms": float(np.median(times) * 1000),
        "p99_ms": float(np.percentile(times, 99) * 1000),
        "mean_ms": float(np.mean(times) * 1000),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/splits")
    ap.add_argument("--imm-tuning", default="results/imm_tuning.json")
    ap.add_argument("--knet-checkpoint", default=None)
    ap.add_argument("--rkn-checkpoint", default=None)
    ap.add_argument("--rkn-fc1-mult", type=int, default=3)
    ap.add_argument("--rkn-hidden-mult", type=int, default=3)
    ap.add_argument("--rkn-fc2-mult", type=int, default=4)
    ap.add_argument("--bench-rkn-reference-default", action="store_true",
                     help="also benchmark RKN at the reference's default (10,10,20) sizing, "
                          "untrained -- forward-pass compute cost is independent of training")
    ap.add_argument("--n-warmup", type=int, default=200)
    ap.add_argument("--n-measure", type=int, default=10000)
    ap.add_argument("--out", default="results/latency.json")
    args = ap.parse_args()

    torch.set_num_threads(1)

    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text())
    dt = manifest["regimes"]["nominal"]["dt"]
    r = manifest["regimes"]["nominal"]
    R = np.diag([r["sigma_az_rad"] ** 2, r["sigma_el_rad"] ** 2, r["sigma_range_m"] ** 2])

    x = np.array([2000.0, 500.0, 300.0, 40.0, -10.0, 5.0, 0.0, 0.0, 0.0])
    P = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    z = np.array([0.2, 0.1, 2100.0])

    results = {}

    ekf = EKF(dt, R)
    results["EKF"] = bench(lambda: ekf.step(x, P, z), args.n_warmup, args.n_measure)

    ukf = UKF(dt, R)
    results["UKF"] = bench(lambda: ukf.step(x, P, z), args.n_warmup, args.n_measure)

    imm_best = json.loads(Path(args.imm_tuning).read_text())["best"]
    imm = IMM(
        dt, R, omega_ct=np.deg2rad(imm_best["omega_ct_deg"]), p_stay=imm_best["p_stay"],
        ca_sigma_a=imm_best["ca_sigma_a"], ca_tau=imm_best["ca_tau"],
    )
    imm_state = imm.init_state(x, P)
    results["IMM"] = bench(lambda: imm.step(imm_state, z), args.n_warmup, args.n_measure)

    if args.knet_checkpoint and Path(args.knet_checkpoint).exists():
        net = KalmanGainGRU().double()
        net.load_state_dict(torch.load(args.knet_checkpoint, map_location="cpu", weights_only=True))
        net.eval()
        flt = KalmanNetFilter(net, dt)
        x0 = torch.tensor(x, dtype=torch.float64).unsqueeze(0)
        prior_Q = torch.eye(STATE_DIM, dtype=torch.float64) * 0.1
        prior_Sigma = torch.eye(STATE_DIM, dtype=torch.float64) * 100.0
        prior_S = torch.tensor(R, dtype=torch.float64)
        flt.init_sequence(x0, prior_Q, prior_Sigma, prior_S)
        z_t = torch.tensor(z, dtype=torch.float64).unsqueeze(0)
        param_count = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            results["KalmanNet-CPU"] = bench(lambda: flt.step(z_t), args.n_warmup, args.n_measure)
        results["KalmanNet-CPU"]["param_count"] = param_count

    def _bench_rkn(net, label, x0_np, P0_np, z_np):
        rkn_flt = RKNFilter(net, dt)
        x0_t = torch.tensor(x0_np, dtype=torch.float64).unsqueeze(0)
        P0_t = torch.tensor(P0_np, dtype=torch.float64).unsqueeze(0)
        rkn_flt.init_sequence(x0_t, P0_t)
        z_t2 = torch.tensor(z_np, dtype=torch.float64).unsqueeze(0)
        with torch.no_grad():
            results[label] = bench(lambda: rkn_flt.step(z_t2), args.n_warmup, args.n_measure)
        results[label]["param_count"] = sum(p.numel() for p in net.parameters())

    if args.rkn_checkpoint and Path(args.rkn_checkpoint).exists():
        net = RKNGainCovNetwork(
            fc1_mult=args.rkn_fc1_mult, hidden_mult=args.rkn_hidden_mult, fc2_mult=args.rkn_fc2_mult,
        ).double()
        net.load_state_dict(torch.load(args.rkn_checkpoint, map_location="cpu", weights_only=True))
        net.eval()
        _bench_rkn(net, "RKN-CPU", x, P, z)

    if args.bench_rkn_reference_default:
        # Untrained is fine here: forward-pass compute cost depends only on
        # the architecture's shape, not the learned weights.
        ref_net = RKNGainCovNetwork().double()
        ref_net.eval()
        _bench_rkn(ref_net, "RKN-reference-default-untrained", x, P, z)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "methodology": "batch_size=1, single-threaded (torch.set_num_threads(1)), "
                        f"{args.n_warmup} warmup steps, {args.n_measure} measured steps, median/p99 reported",
        "results_ms": results,
    }, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
