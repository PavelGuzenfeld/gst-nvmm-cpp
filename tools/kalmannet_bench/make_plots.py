#!/usr/bin/env python3
"""Generate the plots required by the brief's Deliverables section: trajectory
overlays, RMSE-vs-time, NEES/NIS with chi-square bounds, coverage curves,
accuracy-vs-maneuver-intensity, and latency-vs-parameters. Reads the already-
computed results (results_table.csv, anees_highmc.json, latency.json,
knet_vanilla/training_log.json) plus a handful of fresh per-step runs (needed
for the time-series plots, which the aggregate-metric eval scripts don't
retain).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from filters.ekf import EKF
from filters.imm import IMM
from filters.state import STATE_DIM
from filters.ukf import UKF

_INIT_OFFSET = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_P0 = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])


def _run_ekf_like(flt, x_true, z):
    T = x_true.shape[0]
    xs = np.zeros((T, STATE_DIM))
    nees = np.zeros(T)
    x, P = x_true[0] + _INIT_OFFSET, _P0.copy()
    for t in range(T):
        if t == 0:
            if isinstance(flt, UKF):
                pts_pred = flt._sigma_points(x, P)
                x, P, _, _ = flt.update(x, P, pts_pred, z[t])
            else:
                x, P, _, _ = flt.update(x, P, z[t])
        else:
            x, P, _, _ = flt.step(x, P, z[t])
        xs[t] = x
        e = x - x_true[t]
        nees[t] = e @ np.linalg.solve(P, e)
    return xs, nees


def _run_imm(imm, x_true, z):
    T = x_true.shape[0]
    xs = np.zeros((T, STATE_DIM))
    nees = np.zeros(T)
    state = imm.init_state(x_true[0] + _INIT_OFFSET, _P0.copy())
    for t in range(T):
        x, P, state, _, _ = imm.step(state, z[t])
        xs[t] = x
        e = x - x_true[t]
        nees[t] = e @ np.linalg.solve(P, e)
    return xs, nees


def plot_trajectory_overlay(out_dir, dt, R, states, measurements):
    x_true = states[0]
    z = measurements[0, 0]

    ekf_xs, _ = _run_ekf_like(EKF(dt, R), x_true, z)
    ukf_xs, _ = _run_ekf_like(UKF(dt, R), x_true, z)
    imm_best = json.loads(Path("results/imm_tuning.json").read_text())["best"]
    imm = IMM(dt, R, omega_ct=np.deg2rad(imm_best["omega_ct_deg"]), p_stay=imm_best["p_stay"],
              ca_sigma_a=imm_best["ca_sigma_a"], ca_tau=imm_best["ca_tau"])
    imm_xs, _ = _run_imm(imm, x_true, z)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x_true[:, 0], x_true[:, 1], "k-", lw=2, label="ground truth")
    ax.plot(ekf_xs[:, 0], ekf_xs[:, 1], "--", label="EKF (CV-only)", alpha=0.8)
    ax.plot(ukf_xs[:, 0], ukf_xs[:, 1], "--", label="UKF (CV-only)", alpha=0.8)
    ax.plot(imm_xs[:, 0], imm_xs[:, 1], "-", label="IMM (tuned)", alpha=0.9)
    ax.set_xlabel("px (m)")
    ax.set_ylabel("py (m)")
    ax.set_title("Trajectory overlay: ground truth vs. filter estimates (test traj. 0)")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_overlay.png", dpi=150)
    plt.close(fig)


def plot_rmse_vs_time(out_dir, dt, R, states, measurements, n_traj=30):
    T = states.shape[1]
    filters = {
        "EKF": (lambda: EKF(dt, R), _run_ekf_like),
        "UKF": (lambda: UKF(dt, R), _run_ekf_like),
    }
    imm_best = json.loads(Path("results/imm_tuning.json").read_text())["best"]
    filters["IMM"] = (
        lambda: IMM(dt, R, omega_ct=np.deg2rad(imm_best["omega_ct_deg"]), p_stay=imm_best["p_stay"],
                    ca_sigma_a=imm_best["ca_sigma_a"], ca_tau=imm_best["ca_tau"]),
        _run_imm,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    t_axis = np.arange(T) * dt
    for name, (factory, run_fn) in filters.items():
        sq_err = np.zeros(T)
        for i in range(n_traj):
            flt = factory()
            xs, _ = run_fn(flt, states[i], measurements[i, 0])
            sq_err += np.sum((xs[:, :3] - states[i, :, :3]) ** 2, axis=-1)
        rmse_t = np.sqrt(sq_err / n_traj)
        ax.plot(t_axis, rmse_t, label=name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position RMSE (m)")
    ax.set_yscale("log")
    ax.set_title(f"Position RMSE vs. time ({n_traj} test trajectories, log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_vs_time.png", dpi=150)
    plt.close(fig)


def plot_nees_with_bounds(out_dir, dt, R, states, measurements, n_traj=None):
    T = states.shape[1]
    n_traj = n_traj or states.shape[0]
    mc = measurements.shape[1]
    dim = STATE_DIM

    imm_best = json.loads(Path("results/imm_tuning.json").read_text())["best"]

    fig, ax = plt.subplots(figsize=(9, 5))
    t_axis = np.arange(T) * dt

    for name, factory, run_fn in [
        ("EKF", lambda: EKF(dt, R), _run_ekf_like),
        ("IMM", lambda: IMM(dt, R, omega_ct=np.deg2rad(imm_best["omega_ct_deg"]), p_stay=imm_best["p_stay"],
                             ca_sigma_a=imm_best["ca_sigma_a"], ca_tau=imm_best["ca_tau"]), _run_imm),
    ]:
        nees_all = np.zeros((n_traj, mc, T))
        for i in range(n_traj):
            for m in range(mc):
                flt = factory()
                _, nees = run_fn(flt, states[i], measurements[i, m])
                nees_all[i, m] = nees
        anees_t = nees_all.mean(axis=(0, 1))
        ax.plot(t_axis, anees_t, label=f"{name} ANEES(t)")

    N = n_traj * mc
    low = chi2.ppf(0.025, df=N * dim) / N
    high = chi2.ppf(0.975, df=N * dim) / N
    ax.axhline(low, color="gray", ls=":", label="95% chi-square bound")
    ax.axhline(high, color="gray", ls=":")
    ax.axhline(dim, color="black", ls="--", alpha=0.5, label=f"expected (dim={dim})")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ANEES")
    ax.set_yscale("log")
    ax.set_title(f"ANEES(t) vs. chi-square 95% bounds ({n_traj} traj x {mc} mc)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "nees_vs_time.png", dpi=150)
    plt.close(fig)


def plot_coverage(out_dir, highmc_json_path, rkn_eval_json_path=None):
    data = json.loads(Path(highmc_json_path).read_text())["results"]
    filters = [r["filter"] for r in data]
    cov68 = [r["coverage_68"] for r in data]
    cov95 = [r["coverage_95"] for r in data]

    if rkn_eval_json_path and Path(rkn_eval_json_path).exists():
        rkn_highmc = json.loads(Path(rkn_eval_json_path).read_text())["highmc"]
        filters.append("RKN")
        cov68.append(rkn_highmc["coverage_68"])
        cov95.append(rkn_highmc["coverage_95"])

    x = np.arange(len(filters))
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    ax.bar(x - width / 2, cov68, width, label="empirical coverage @ 68%")
    ax.bar(x + width / 2, cov95, width, label="empirical coverage @ 95%")
    ax.axhline(0.68, color="C0", ls="--", alpha=0.6)
    ax.axhline(0.95, color="C1", ls="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(filters)
    ax.set_ylabel("empirical coverage")
    ax.set_title("Calibration coverage vs. nominal (dashed), high-MC subset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "coverage.png", dpi=150)
    plt.close(fig)


def plot_accuracy_vs_maneuver_intensity(out_dir, results_csv_path, knet_csv_path, rkn_csv_path=None):
    rows = list(csv.DictReader(open(results_csv_path)))
    knet_rows = list(csv.DictReader(open(knet_csv_path))) if Path(knet_csv_path).exists() else []
    rkn_rows = list(csv.DictReader(open(rkn_csv_path))) if rkn_csv_path and Path(rkn_csv_path).exists() else []

    regime_order = ["low_noise", "nominal", "high_noise", "heavy_tailed", "high_maneuver"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["EKF", "UKF", "IMM"]:
        ys = []
        for regime in regime_order:
            match = [r for r in rows if r["filter"] == name and r["regime"] == regime]
            ys.append(float(match[0]["pos_rmse"]) if match else np.nan)
        ax.plot(regime_order, ys, "o-", label=name)
    if knet_rows:
        ys = []
        for regime in regime_order:
            match = [r for r in knet_rows if r["regime"] == regime]
            ys.append(float(match[0]["pos_rmse"]) if match else np.nan)
        ax.plot(regime_order, ys, "o-", label="KalmanNet-vanilla")
    if rkn_rows:
        ys = []
        for regime in regime_order:
            match = [r for r in rkn_rows if r["regime"] == regime]
            ys.append(float(match[0]["pos_rmse"]) if match else np.nan)
        ax.plot(regime_order, ys, "o-", label="RKN")
    ax.set_ylabel("position RMSE (m)")
    ax.set_yscale("log")
    ax.set_xlabel("regime (roughly increasing maneuver/noise intensity ->)")
    ax.set_title("Accuracy vs. maneuver/noise intensity (robustness sweep)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_intensity.png", dpi=150)
    plt.close(fig)


def plot_latency_vs_parameters(out_dir, latency_json_path):
    data = json.loads(Path(latency_json_path).read_text())["results_ms"]
    names = list(data.keys())
    params = [data[n].get("param_count", 0) for n in names]
    latency = [data[n]["median_ms"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(params, latency, s=80)
    for n, p, l in zip(names, params, latency):
        ax.annotate(n, (p, l), textcoords="offset points", xytext=(6, 4))
    ax.axhline(2.0, color="red", ls="--", alpha=0.6, label="2 ms target")
    ax.set_xlabel("parameter count (0 for classical filters)")
    ax.set_ylabel("median per-step latency (ms)")
    ax.set_title("Latency vs. parameter count (single-threaded CPU)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_parameters.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/splits")
    ap.add_argument("--out", default="results/plots")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((data_dir / "manifest.json").read_text())
    dt = manifest["regimes"]["nominal"]["dt"]
    r = manifest["regimes"]["nominal"]
    R = np.diag([r["sigma_az_rad"] ** 2, r["sigma_el_rad"] ** 2, r["sigma_range_m"] ** 2])

    test_states, test_meas = np.load(data_dir / "test.npz")["states"], np.load(data_dir / "test.npz")["measurements"]
    highmc_states = np.load(data_dir / "test_highmc.npz")["states"]
    highmc_meas = np.load(data_dir / "test_highmc.npz")["measurements"]

    print("plotting trajectory overlay...", flush=True)
    plot_trajectory_overlay(out_dir, dt, R, test_states, test_meas)
    print("plotting RMSE vs time...", flush=True)
    plot_rmse_vs_time(out_dir, dt, R, test_states, test_meas, n_traj=30)
    print("plotting ANEES vs time (high-MC subset)...", flush=True)
    plot_nees_with_bounds(out_dir, dt, R, highmc_states, highmc_meas)
    print("plotting coverage...", flush=True)
    plot_coverage(out_dir, "results/anees_highmc.json", rkn_eval_json_path="results/rkn_eval.json")
    print("plotting accuracy vs maneuver intensity...", flush=True)
    plot_accuracy_vs_maneuver_intensity(out_dir, "results/results_table.csv", "results/knet_test_rmse.csv", rkn_csv_path="results/rkn_test_rmse.csv")
    print("plotting latency vs parameters...", flush=True)
    plot_latency_vs_parameters(out_dir, "results/latency.json")

    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
