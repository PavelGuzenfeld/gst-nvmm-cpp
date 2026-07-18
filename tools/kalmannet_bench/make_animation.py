#!/usr/bin/env python3
"""Animated 3D + HUD visualization of every filter's behavior on one
maneuvering trajectory (see motion-hud-video-plan.md). Left panel: an orbiting
3D view of the ground-truth target and all five filters' estimates, each drawn
as an alpha-fading trailing tail with a head marker. Right panel (HUD): live
per-filter position-error bars (log), per-filter NEES bars (log, with the
chi-square consistency band shaded; vanilla KalmanNet shows N/A because it has
no covariance), IMM mode-probability bars, a true-motion-mode banner, and a
clock.

Each filter is re-run per-step on the chosen trajectory with the SAME init
conventions its eval_*.py uses, so on-screen behavior matches the reported
numbers. Regime is high_maneuver (drama, per the plan); trajectory is auto-
selected as the one with the most CV/CT/CA mode switches.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.stats import chi2

from filters.ekf import EKF
from filters.imm import IMM
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.knet_filter import KalmanNetFilter
from filters.knet.rkn_filter import RKNFilter
from filters.knet.rkn_network import RKNGainCovNetwork
from filters.motion_models import LABEL_NAMES, cv_model
from filters.state import STATE_DIM
from filters.ukf import UKF

_INIT_OFFSET = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_P0_CLASSICAL = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])

# Filter display order, labels, colors. Ground truth is drawn separately.
FILTERS = ["EKF", "UKF", "IMM", "KalmanNet", "RKN"]
COLORS = {
    "truth": "black",
    "EKF": "tab:blue",
    "UKF": "tab:orange",
    "IMM": "tab:green",
    "KalmanNet": "tab:red",
    "RKN": "tab:purple",
}
# NEES is only meaningful for filters that produce a covariance.
HAS_NEES = {"EKF": True, "UKF": True, "IMM": True, "KalmanNet": False, "RKN": True}
IMM_MODE_LABELS = ["CV", "CT+", "CT-", "CA"]

_NEES_DIM = STATE_DIM  # full-state NEES, matching eval.py / eval_rkn.py
_CHI2_LO = float(chi2.ppf(0.025, _NEES_DIM))   # ~2.70 for df=9
_CHI2_HI = float(chi2.ppf(0.975, _NEES_DIM))   # ~19.02 for df=9


def _nees(e, P):
    try:
        return float(e @ np.linalg.solve(P, e))
    except np.linalg.LinAlgError:
        return float("nan")


def select_trajectory(labels):
    """Index of the trajectory with the most motion-mode switches."""
    switches = ((labels[:, 1:] != labels[:, :-1]).sum(axis=1))
    return int(np.argmax(switches)), int(switches.max())


def run_all_filters(states, z, dt, R, knet_ckpt, rkn_ckpt, rkn_mults):
    """Re-run every filter on one trajectory. Returns per-step dicts:
    est[name][T,3] positions, err[name][T] position error, nees[name][T]
    (nan where unavailable), and mus[T,4] IMM mode probabilities."""
    T = states.shape[0]
    est = {n: np.zeros((T, 3)) for n in FILTERS}
    err = {n: np.zeros(T) for n in FILTERS}
    nees = {n: np.full(T, np.nan) for n in FILTERS}
    mus = np.zeros((T, 4))

    def record(name, t, x_est, P=None):
        est[name][t] = x_est[:3]
        e = x_est - states[t]
        e = np.nan_to_num(e, nan=1e5, posinf=1e5, neginf=-1e5)
        err[name][t] = np.linalg.norm(e[:3])
        if P is not None:
            nees[name][t] = _nees(e, P)

    # --- EKF ---
    ekf = EKF(dt, R)
    x, P = states[0] + _INIT_OFFSET, _P0_CLASSICAL.copy()
    for t in range(T):
        x, P, _, _ = ekf.update(x, P, z[t]) if t == 0 else ekf.step(x, P, z[t])
        record("EKF", t, x, P)

    # --- UKF ---
    ukf = UKF(dt, R)
    x, P = states[0] + _INIT_OFFSET, _P0_CLASSICAL.copy()
    for t in range(T):
        if t == 0:
            pts = ukf._sigma_points(x, P)
            x, P, _, _ = ukf.update(x, P, pts, z[t])
        else:
            x, P, _, _ = ukf.step(x, P, z[t])
        record("UKF", t, x, P)

    # --- IMM (tuned) ---
    imm_best = json.loads(Path("results/imm_tuning.json").read_text())["best"]
    imm = IMM(dt, R, omega_ct=np.deg2rad(imm_best["omega_ct_deg"]), p_stay=imm_best["p_stay"],
              ca_sigma_a=imm_best["ca_sigma_a"], ca_tau=imm_best["ca_tau"])
    state = imm.init_state(states[0] + _INIT_OFFSET, _P0_CLASSICAL.copy())
    for t in range(T):
        x, P, state, _, _ = imm.step(state, z[t])
        record("IMM", t, x, P)
        mus[t] = state["mu"]

    # --- vanilla KalmanNet (no covariance -> no NEES) ---
    _, Q = cv_model(dt)
    prior_Q = torch.tensor(Q, dtype=torch.float64)
    prior_Sigma = torch.eye(STATE_DIM, dtype=torch.float64) * 100.0
    prior_S = torch.tensor(R, dtype=torch.float64)
    knet = KalmanGainGRU().double()
    knet.load_state_dict(torch.load(knet_ckpt, map_location="cpu", weights_only=True))
    knet.eval()
    kflt = KalmanNetFilter(knet, dt)
    x0 = torch.tensor(states[0] + _INIT_OFFSET, dtype=torch.float64).unsqueeze(0)
    kflt.init_sequence(x0, prior_Q, prior_Sigma, prior_S)
    z_t = torch.tensor(z, dtype=torch.float64)
    with torch.no_grad():
        for t in range(T):
            x_est = kflt.step(z_t[t].unsqueeze(0))[0].numpy()
            record("KalmanNet", t, x_est)

    # --- Recursive KalmanNet ---
    rnet = RKNGainCovNetwork(fc1_mult=rkn_mults[0], hidden_mult=rkn_mults[1], fc2_mult=rkn_mults[2]).double()
    rnet.load_state_dict(torch.load(rkn_ckpt, map_location="cpu", weights_only=True))
    rnet.eval()
    rflt = RKNFilter(rnet, dt)
    x0 = torch.tensor(states[0] + _INIT_OFFSET, dtype=torch.float64).unsqueeze(0)
    P0 = torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0) * 100.0
    rflt.init_sequence(x0, P0)
    with torch.no_grad():
        for t in range(T):
            x_est, P_est = rflt.step(z_t[t].unsqueeze(0))
            record("RKN", t, x_est[0].numpy(), P_est[0].numpy())

    return est, err, nees, mus


def scene_limits(states, est, max_side_m):
    """Cubic axis limits containing ground truth + all filter estimates over
    the whole clip, clamped so a wildly-diverged filter can't shrink the good
    ones to a dot (it clips instead -- an honest 'lost it' signal)."""
    pts = [states[:, :3]] + [est[n] for n in FILTERS]
    allp = np.concatenate(pts, axis=0)
    ctr = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
    span = (allp.max(axis=0) - allp.min(axis=0)).max() * 1.1
    side = min(max(span, 1000.0), max_side_m)
    half = side / 2.0
    return ctr, half


def build_animation(states, est, err, nees, mus, labels, dt, args):
    T = states.shape[0]
    ctr, half = scene_limits(states, est, args.max_extent_m)

    fig = plt.figure(figsize=(16, 9), dpi=args.dpi)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.55, 1.0], hspace=0.55, wspace=0.18,
                          left=0.02, right=0.97, top=0.94, bottom=0.06)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_err = fig.add_subplot(gs[0, 1])
    ax_nees = fig.add_subplot(gs[1, 1])
    ax_mode = fig.add_subplot(gs[2, 1])
    ax_banner = fig.add_subplot(gs[3, 1]); ax_banner.axis("off")

    ax3d.set_xlim(ctr[0] - half, ctr[0] + half)
    ax3d.set_ylim(ctr[1] - half, ctr[1] + half)
    ax3d.set_zlim(ctr[2] - half, ctr[2] + half)
    ax3d.set_xlabel("px (m)"); ax3d.set_ylabel("py (m)"); ax3d.set_zlabel("pz (m)")
    ax3d.set_title("3D target motion + filter estimates (high_maneuver regime)")
    # sensor at origin (may sit at/near the frame edge at long range)
    if (abs(-ctr) <= half).all():
        ax3d.scatter([0], [0], [0], marker="^", s=60, color="dimgray", label="sensor")

    # Tail collections + head markers per line (truth + 5 filters). Seed each
    # collection with a degenerate first-point segment -- add_collection3d
    # autoscales from the segments and errors on an empty one.
    tail, head = {}, {}
    first_pt = {"truth": states[0, :3], **{n: est[n][0] for n in FILTERS}}
    for name in ["truth"] + FILTERS:
        p0 = first_pt[name]
        lc = Line3DCollection([np.array([p0, p0])], linewidths=2.2 if name == "truth" else 1.8)
        ax3d.add_collection3d(lc)
        tail[name] = lc
        (h,) = ax3d.plot([], [], [], marker="o", ms=7 if name == "truth" else 5,
                         color=COLORS[name], label=("ground truth" if name == "truth" else name))
        head[name] = h
    ax3d.legend(loc="upper left", fontsize=8, ncol=2)

    W = args.tail_len

    def set_tail(name, traj, t):
        lo = max(0, t - W)
        pts = traj[lo:t + 1]
        if len(pts) < 2:
            tail[name].set_segments([])
            return
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        n = len(segs)
        rgba = np.zeros((n, 4))
        rgba[:, :3] = matplotlib.colors.to_rgb(COLORS[name])
        rgba[:, 3] = np.linspace(0.05, 1.0, n)  # fade tail -> head
        tail[name].set_segments(segs)
        tail[name].set_color(rgba)

    def update(frame):
        t = frame
        # 3D: orbit + tails + heads
        ax3d.view_init(elev=22, azim=args.azim0 + args.orbit_deg * frame)
        set_tail("truth", states[:, :3], t)
        head["truth"].set_data_3d([states[t, 0]], [states[t, 1]], [states[t, 2]])
        for name in FILTERS:
            set_tail(name, est[name], t)
            head[name].set_data_3d([est[name][t, 0]], [est[name][t, 1]], [est[name][t, 2]])

        # HUD: position-error bars (log)
        ax_err.clear()
        vals = [max(err[n][t], 1e-2) for n in FILTERS]
        ax_err.barh(range(len(FILTERS)), vals, color=[COLORS[n] for n in FILTERS])
        ax_err.set_xscale("log"); ax_err.set_xlim(0.1, 1e4)
        ax_err.set_yticks(range(len(FILTERS))); ax_err.set_yticklabels(FILTERS, fontsize=8)
        ax_err.invert_yaxis()
        ax_err.set_title("position error (m, log)", fontsize=9)
        for i, v in enumerate(vals):
            ax_err.text(v, i, f" {v:.1f}", va="center", fontsize=7)

        # HUD: NEES bars (log) with chi-square band; KalmanNet -> N/A
        ax_nees.clear()
        ax_nees.axvspan(_CHI2_LO, _CHI2_HI, color="gray", alpha=0.25)
        for i, n in enumerate(FILTERS):
            if HAS_NEES[n]:
                v = max(nees[n][t], 1e-2) if np.isfinite(nees[n][t]) else 1e-2
                ax_nees.barh(i, v, color=COLORS[n])
                ax_nees.text(v, i, f" {v:.1f}", va="center", fontsize=7)
            else:
                ax_nees.text(0.15, i, "N/A (no covariance)", va="center", fontsize=7, color=COLORS[n])
        ax_nees.set_xscale("log"); ax_nees.set_xlim(0.1, 1e7)
        ax_nees.set_yticks(range(len(FILTERS))); ax_nees.set_yticklabels(FILTERS, fontsize=8)
        ax_nees.invert_yaxis()
        ax_nees.set_title(f"NEES (log); gray band = 95% chi-square [{_CHI2_LO:.1f}, {_CHI2_HI:.1f}]", fontsize=9)

        # HUD: IMM mode probabilities
        ax_mode.clear()
        ax_mode.barh(range(4), mus[t], color="tab:green", alpha=0.8)
        ax_mode.set_xlim(0, 1)
        ax_mode.set_yticks(range(4)); ax_mode.set_yticklabels(IMM_MODE_LABELS, fontsize=8)
        ax_mode.invert_yaxis()
        ax_mode.set_title("IMM mode probability", fontsize=9)

        # HUD: true-mode banner + clock
        ax_banner.clear(); ax_banner.axis("off")
        true_mode = LABEL_NAMES[int(labels[t])].upper()
        ax_banner.text(0.5, 0.62, f"true mode: {true_mode}", ha="center", fontsize=16, weight="bold")
        ax_banner.text(0.5, 0.20, f"t = {t * dt:5.2f} s     step {t}/{T}", ha="center", fontsize=11)
        return []

    return fig, update, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/splits")
    ap.add_argument("--regime", default="high_maneuver")
    ap.add_argument("--knet-ckpt", default="results/knet_vanilla/best.pt")
    ap.add_argument("--rkn-ckpt", default="results/rkn/best.pt")
    ap.add_argument("--rkn-mults", default="3,3,4")
    ap.add_argument("--out-dir", default="results/video")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = full trajectory; >0 for a quick test render")
    ap.add_argument("--tail-len", type=int, default=75)
    ap.add_argument("--orbit-deg", type=float, default=0.2)
    ap.add_argument("--azim0", type=float, default=-60.0)
    ap.add_argument("--max-extent-m", type=float, default=8000.0)
    ap.add_argument("--mp4-fps", type=int, default=30)
    ap.add_argument("--gif-stride", type=int, default=5)
    ap.add_argument("--gif-fps", type=int, default=12)
    ap.add_argument("--gif-width", type=int, default=900)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    reg = manifest["regimes"][args.regime]
    dt = reg["dt"]
    R = np.diag([reg["sigma_az_rad"] ** 2, reg["sigma_el_rad"] ** 2, reg["sigma_range_m"] ** 2])

    d = np.load(data_dir / f"test_{args.regime}.npz")
    states_all, meas_all, labels_all = d["states"], d["measurements"], d["labels"]
    idx, nsw = select_trajectory(labels_all)
    print(f"selected trajectory {idx} ({nsw} mode switches) from {args.regime}", flush=True)

    states = states_all[idx]
    z = meas_all[idx, 0]
    labels = labels_all[idx]
    rkn_mults = tuple(int(x) for x in args.rkn_mults.split(","))

    print("running all filters over the trajectory...", flush=True)
    est, err, nees, mus = run_all_filters(states, z, dt, R, args.knet_ckpt, args.rkn_ckpt, rkn_mults)

    fig, update, T = build_animation(states, est, err, nees, mus, labels, dt, args)
    n_frames = T if args.max_frames <= 0 else min(args.max_frames, T)
    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = out_dir / "motion_hud.mp4"
    print(f"rendering {mp4_path} ({n_frames} frames @ {args.mp4_fps} fps)...", flush=True)
    anim.save(str(mp4_path), writer=FFMpegWriter(fps=args.mp4_fps, bitrate=4000))
    print(f"wrote {mp4_path}", flush=True)

    if not args.no_gif:
        gif_path = out_dir / "motion_hud.gif"
        gif_dpi = args.dpi * args.gif_width / (16 * args.dpi)  # scale so width ~= gif_width px
        print(f"rendering {gif_path} (every {args.gif_stride}th frame @ {args.gif_fps} fps)...", flush=True)
        gif_anim = FuncAnimation(fig, update, frames=range(0, n_frames, args.gif_stride), blit=False)
        gif_anim.save(str(gif_path), writer=PillowWriter(fps=args.gif_fps), dpi=gif_dpi)
        print(f"wrote {gif_path}", flush=True)


if __name__ == "__main__":
    main()
