#!/usr/bin/env python3
"""Evaluate the trained Recursive KalmanNet checkpoint on the same test.npz
(all mc realizations), test_highmc.npz, and robustness-sweep sets the
classical filters and vanilla KalmanNet are judged on -- with the SAME
metric code (filters/metrics.py), since unlike vanilla KalmanNet, RKN
produces an actual covariance and can be judged on NEES/NIS/coverage too.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from filters.knet.rkn_filter import RKNFilter
from filters.knet.rkn_network import RKNGainCovNetwork
from filters.metrics import anees_with_bounds, empirical_coverage, fraction_inside_bounds
from filters.state import POS, STATE_DIM, VEL

_INIT_OFFSET = torch.tensor([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
_BURN_IN = 20


def evaluate_dataset(net, dt, states, measurements):
    """states: [N,T,9], measurements: [N,mc,T,3]. Returns dict of arrays:
    err [N,mc,T,9], nees [N,mc,T]. (No NIS here -- RKN's per-step innovation
    covariance S isn't separately exposed by the reference's A/B split the
    way the classical filters' S is; NEES/coverage is the calibration
    signal this evaluation targets, matching the brief's headline metric.)"""
    N, mc, T, _ = measurements.shape
    err = np.zeros((N, mc, T, STATE_DIM))
    nees = np.zeros((N, mc, T))

    with torch.no_grad():
        for i in range(N):
            x_true = torch.tensor(states[i], dtype=torch.float64).unsqueeze(0)
            for m in range(mc):
                z = torch.tensor(measurements[i, m], dtype=torch.float64).unsqueeze(0)
                flt = RKNFilter(net, dt)
                x0 = x_true[:, 0, :] + _INIT_OFFSET
                P0 = torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0) * 100.0
                flt.init_sequence(x0, P0)
                for t in range(T):
                    x_upd, P_upd = flt.step(z[:, t, :])
                    e = (x_upd[0] - x_true[0, t]).numpy()
                    e = np.nan_to_num(e, nan=1e5, posinf=1e5, neginf=-1e5)
                    err[i, m, t] = e
                    P_np = P_upd[0].numpy()
                    try:
                        nees[i, m, t] = e @ np.linalg.solve(P_np, e)
                    except np.linalg.LinAlgError:
                        nees[i, m, t] = 1e10
    return {"err": err, "nees": nees}


def summarize(result, burn_in=_BURN_IN):
    err = result["err"][:, :, burn_in:, :]
    pos_rmse = float(np.sqrt(np.mean(np.sum(err[..., POS] ** 2, axis=-1))))
    vel_rmse = float(np.sqrt(np.mean(np.sum(err[..., VEL] ** 2, axis=-1))))

    nees = result["nees"][:, :, burn_in:]
    mc = nees.shape[1]
    anees, bounds = anees_with_bounds(nees, dim=STATE_DIM, mc_axis=1)
    frac_in_95 = fraction_inside_bounds(anees, bounds)
    coverage = empirical_coverage(nees.reshape(-1), dim=STATE_DIM, levels=(0.68, 0.95))

    return {
        "pos_rmse": pos_rmse, "vel_rmse": vel_rmse,
        "anees_mean": float(np.mean(anees)), "anees_bounds": bounds,
        "anees_fraction_inside_95": frac_in_95,
        "coverage_68": coverage[0.68], "coverage_95": coverage[0.95],
        "n_mc": mc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/splits")
    ap.add_argument("--checkpoint", default="results/rkn/best.pt")
    ap.add_argument("--fc1-mult", type=int, default=3)
    ap.add_argument("--hidden-mult", type=int, default=3)
    ap.add_argument("--fc2-mult", type=int, default=4)
    ap.add_argument("--out", default="results/rkn_eval.json")
    ap.add_argument("--out-csv", default="results/rkn_test_rmse.csv")
    args = ap.parse_args()

    torch.set_num_threads(1)
    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())

    net = RKNGainCovNetwork(fc1_mult=args.fc1_mult, hidden_mult=args.hidden_mult, fc2_mult=args.fc2_mult).double()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    net.eval()

    results = {}
    csv_rows = []
    for regime_name in manifest["regimes"]:
        path = data_dir / ("test.npz" if regime_name == "nominal" else f"test_{regime_name}.npz")
        if not path.exists():
            continue
        d = np.load(path)
        dt = manifest["regimes"][regime_name]["dt"]
        result = evaluate_dataset(net, dt, d["states"], d["measurements"])
        summary = summarize(result)
        results[regime_name] = summary
        csv_rows.append({"filter": "RKN", "regime": regime_name, "pos_rmse": summary["pos_rmse"], "vel_rmse": summary["vel_rmse"]})
        print(f"{regime_name}: pos_rmse={summary['pos_rmse']:.2f}m vel_rmse={summary['vel_rmse']:.2f}m/s "
              f"ANEES={summary['anees_mean']:.2f} (bounds {summary['anees_bounds'][0]:.2f}-{summary['anees_bounds'][1]:.2f}) "
              f"frac_in_95={summary['anees_fraction_inside_95']:.3f} coverage68={summary['coverage_68']:.3f} coverage95={summary['coverage_95']:.3f}",
              flush=True)

    # High-MC subset: the tight-bound headline calibration numbers.
    highmc = np.load(data_dir / "test_highmc.npz")
    dt = manifest["regimes"]["nominal"]["dt"]
    highmc_result = evaluate_dataset(net, dt, highmc["states"], highmc["measurements"])
    highmc_summary = summarize(highmc_result)
    results["highmc"] = highmc_summary
    print(f"highmc: ANEES={highmc_summary['anees_mean']:.2f} (bounds {highmc_summary['anees_bounds'][0]:.2f}-{highmc_summary['anees_bounds'][1]:.2f}) "
          f"frac_in_95={highmc_summary['anees_fraction_inside_95']:.3f} coverage68={highmc_summary['coverage_68']:.3f} coverage95={highmc_summary['coverage_95']:.3f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filter", "regime", "pos_rmse", "vel_rmse"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"wrote {args.out} and {args.out_csv}")


if __name__ == "__main__":
    main()
