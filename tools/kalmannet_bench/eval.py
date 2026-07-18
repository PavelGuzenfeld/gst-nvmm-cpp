#!/usr/bin/env python3
"""Evaluate every filter over the full test set (RMSE, ANEES/coverage on the
high-MC subset, NIS) and the robustness sweep regimes, with identical metric
code for every filter -- per the brief's fairness requirement. Writes
results/results_table.csv (one row per filter x regime) and
results/anees_highmc.json (the headline calibration numbers).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from filters.ekf import EKF
from filters.imm import IMM
from filters.metrics import anees_with_bounds, empirical_coverage, fraction_inside_bounds
from filters.state import POS, STATE_DIM, VEL
from filters.ukf import UKF

_INIT_OFFSET = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_P0 = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
_BURN_IN = 20  # steps skipped for RMSE/consistency stats, letting initial-condition error settle


def _load(path):
    d = np.load(path)
    return d["states"], d["measurements"]


def run_ekf_like(flt, x_true, z):
    """flt: EKF or UKF instance ((x,P,z)->x,P,y,S signature). x_true: [T,9],
    z: [T,3]. Returns (err[T,9], nees[T], nis[T])."""
    T = x_true.shape[0]
    err = np.zeros((T, STATE_DIM))
    nees = np.zeros(T)
    nis = np.zeros(T)
    x, P = x_true[0] + _INIT_OFFSET, _P0.copy()
    for t in range(T):
        if t == 0:
            if isinstance(flt, UKF):
                pts_pred = flt._sigma_points(x, P)
                x, P, y, S = flt.update(x, P, pts_pred, z[t])
            else:
                x, P, y, S = flt.update(x, P, z[t])
        else:
            x, P, y, S = flt.step(x, P, z[t])
        e = x - x_true[t]
        err[t] = e
        nees[t] = e @ np.linalg.solve(P, e)
        nis[t] = y @ np.linalg.solve(S, y)
    return err, nees, nis


def run_imm(imm, x_true, z):
    T = x_true.shape[0]
    err = np.zeros((T, STATE_DIM))
    nees = np.zeros(T)
    nis = np.zeros(T)
    state = imm.init_state(x_true[0] + _INIT_OFFSET, _P0.copy())
    for t in range(T):
        x, P, state, y, S = imm.step(state, z[t])
        e = x - x_true[t]
        err[t] = e
        nees[t] = e @ np.linalg.solve(P, e)
        nis[t] = y @ np.linalg.solve(S, y)
    return err, nees, nis


def evaluate_dataset(filter_factory, run_fn, states, measurements):
    """states: [N,T,9], measurements: [N,mc,T,3]. Returns dict of aggregated
    arrays: err [N,mc,T,9], nees [N,mc,T], nis [N,mc,T]."""
    N, mc, T, _ = measurements.shape
    err = np.zeros((N, mc, T, STATE_DIM))
    nees = np.zeros((N, mc, T))
    nis = np.zeros((N, mc, T))
    for i in range(N):
        for m in range(mc):
            flt = filter_factory()
            e, ne, ni = run_fn(flt, states[i], measurements[i, m])
            err[i, m] = e
            nees[i, m] = ne
            nis[i, m] = ni
    return {"err": err, "nees": nees, "nis": nis}


def summarize_rmse(result):
    err = result["err"][:, :, _BURN_IN:, :]
    pos_rmse = float(np.sqrt(np.mean(np.sum(err[..., POS] ** 2, axis=-1))))
    vel_rmse = float(np.sqrt(np.mean(np.sum(err[..., VEL] ** 2, axis=-1))))
    return pos_rmse, vel_rmse


def summarize_consistency(result, dim=STATE_DIM, meas_dim=3):
    nees = result["nees"][:, :, _BURN_IN:]  # [N, mc, T']
    nis = result["nis"][:, :, _BURN_IN:]
    mc = nees.shape[1]

    # ANEES: average over the MC axis per (trajectory, step), then test each
    # against the chi-square interval for N=mc, dim=dim -- the brief's
    # headline consistency number.
    anees, bounds = anees_with_bounds(nees, dim=dim, mc_axis=1)  # [N, T']
    anees_pass_frac = fraction_inside_bounds(anees, bounds)

    anis, nis_bounds = anees_with_bounds(nis, dim=meas_dim, mc_axis=1)
    anis_pass_frac = fraction_inside_bounds(anis, nis_bounds)

    coverage = empirical_coverage(nees.reshape(-1), dim=dim, levels=(0.68, 0.95))

    return {
        "anees_mean": float(np.mean(anees)),
        "anees_bounds": bounds,
        "anees_fraction_inside_95": anees_pass_frac,
        "anis_mean": float(np.mean(anis)),
        "anis_bounds": nis_bounds,
        "anis_fraction_inside_95": anis_pass_frac,
        "coverage_68": coverage[0.68],
        "coverage_95": coverage[0.95],
        "n_mc": mc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/splits")
    ap.add_argument("--imm-tuning", default="results/imm_tuning.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    dt = manifest["regimes"]["nominal"]["dt"]

    def R_for(regime_name):
        r = manifest["regimes"][regime_name]
        return np.diag([r["sigma_az_rad"] ** 2, r["sigma_el_rad"] ** 2, r["sigma_range_m"] ** 2])

    imm_tuning = json.loads(Path(args.imm_tuning).read_text())
    best = imm_tuning["best"]
    print(f"using tuned IMM: {best}")

    def make_filters(regime_name):
        R = R_for(regime_name)
        return {
            "EKF": (lambda: EKF(dt, R), run_ekf_like),
            "UKF": (lambda: UKF(dt, R), run_ekf_like),
            "IMM": (
                lambda: IMM(
                    dt, R, omega_ct=np.deg2rad(best["omega_ct_deg"]), p_stay=best["p_stay"],
                    ca_sigma_a=best["ca_sigma_a"], ca_tau=best["ca_tau"],
                ),
                run_imm,
            ),
        }

    rows = []

    # --- Main test set: RMSE with the full 200-trajectory / mc_test set ---
    test_states, test_meas = _load(data_dir / "test.npz")
    filters = make_filters("nominal")
    main_results = {}
    for name, (factory, run_fn) in filters.items():
        print(f"evaluating {name} on test.npz ({test_states.shape[0]} traj x {test_meas.shape[1]} mc)...", flush=True)
        result = evaluate_dataset(factory, run_fn, test_states, test_meas)
        main_results[name] = result
        pos_rmse, vel_rmse = summarize_rmse(result)
        cons = summarize_consistency(result)
        rows.append({"filter": name, "regime": "nominal", "pos_rmse": pos_rmse, "vel_rmse": vel_rmse, **cons})
        print(f"  {name}: pos_rmse={pos_rmse:.2f}m vel_rmse={vel_rmse:.2f}m/s "
              f"ANEES={cons['anees_mean']:.2f} (bounds {cons['anees_bounds'][0]:.2f}-{cons['anees_bounds'][1]:.2f}) "
              f"frac_in_95={cons['anees_fraction_inside_95']:.3f}")

    # --- High-MC subset: the tight-bound ANEES/NIS headline numbers ---
    highmc_states, highmc_meas = _load(data_dir / "test_highmc.npz")
    highmc_rows = []
    for name, (factory, run_fn) in filters.items():
        print(f"evaluating {name} on test_highmc.npz ({highmc_states.shape[0]} traj x {highmc_meas.shape[1]} mc)...", flush=True)
        result = evaluate_dataset(factory, run_fn, highmc_states, highmc_meas)
        cons = summarize_consistency(result)
        highmc_rows.append({"filter": name, **cons})
        print(f"  {name}: ANEES={cons['anees_mean']:.2f} (bounds {cons['anees_bounds'][0]:.2f}-{cons['anees_bounds'][1]:.2f}) "
              f"frac_in_95={cons['anees_fraction_inside_95']:.3f} coverage68={cons['coverage_68']:.3f} coverage95={cons['coverage_95']:.3f}")

    # --- Robustness sweep: RMSE per shifted regime ---
    for regime_name in manifest["regimes"]:
        if regime_name == "nominal":
            continue
        sweep_path = data_dir / f"test_{regime_name}.npz"
        if not sweep_path.exists():
            continue
        sweep_states, sweep_meas = _load(sweep_path)
        filters_r = make_filters(regime_name)
        print(f"evaluating robustness sweep regime '{regime_name}' ({sweep_states.shape[0]} traj)...", flush=True)
        for name, (factory, run_fn) in filters_r.items():
            result = evaluate_dataset(factory, run_fn, sweep_states, sweep_meas)
            pos_rmse, vel_rmse = summarize_rmse(result)
            cons = summarize_consistency(result)
            rows.append({"filter": name, "regime": regime_name, "pos_rmse": pos_rmse, "vel_rmse": vel_rmse, **cons})
            print(f"  {name}: pos_rmse={pos_rmse:.2f}m vel_rmse={vel_rmse:.2f}m/s frac_in_95={cons['anees_fraction_inside_95']:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import csv
    fieldnames = ["filter", "regime", "pos_rmse", "vel_rmse", "anees_mean", "anees_bounds",
                  "anees_fraction_inside_95", "anis_mean", "anis_bounds", "anis_fraction_inside_95",
                  "coverage_68", "coverage_95", "n_mc"]
    with open(out / "results_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    (out / "anees_highmc.json").write_text(json.dumps({
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "n_highmc_trajectories": int(highmc_states.shape[0]),
        "mc_per_trajectory": int(highmc_meas.shape[1]),
        "results": highmc_rows,
    }, indent=2))

    print(f"\nwrote {out}/results_table.csv and {out}/anees_highmc.json")


if __name__ == "__main__":
    main()
