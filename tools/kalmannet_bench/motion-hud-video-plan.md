# Motion + HUD animation: visualizing filter behavior differences

Plan for an animated 3D visualization comparing every filter in the benchmark
(EKF, UKF, IMM, vanilla KalmanNet, Recursive KalmanNet) on a single maneuvering
trajectory, with a live HUD, so the behavioral differences between the models
are directly watchable.

## Goal

Produce a shareable video (mp4 + GIF) showing a maneuvering target moving in 3D,
with all five filters' estimates tracking it live, plus a HUD that quantifies
the per-filter differences (position error, calibration/NEES, IMM mode
probabilities) frame by frame — so a viewer can *see* IMM lock on while EKF cuts
corners and the learned filters wander, and *see why* (IMM switching modes as
the maneuver starts).

## Non-goals

- Not a new metric or result — it is an illustrative visualization of the
  already-computed benchmark. The numbers in `REPORT.md` remain the headline;
  the video is explicitly the "dramatic illustrative case," not a table entry.
- Not real-time / on-device rendering. Offline render in the existing CPU Docker
  image only.
- Not a 2D top-down or gimbal (az/el) view. Not a picture-in-picture measurement
  inset (possible later enhancement, out of scope here).
- No committing of the rendered artifacts (they live under gitignored `results/`).

## Decisions

1. The primary panel is a **3D animated Cartesian view** (px, py, pz), sensor
   marked at the origin — not 2D top-down, not the az/el measurement plane.
2. **All five filters plus ground truth** are drawn in the 3D view: ground
   truth, EKF, UKF, IMM, vanilla KalmanNet, Recursive KalmanNet.
3. The 3D scene is scaled to roughly **8 km** extent (enlarged from an initial
   5 km specifically to contain the high_maneuver regime), sized so the learned
   filters' error — up to ~1700 m RMSE for RKN, its worst regime — and EKF's
   ~188 m corner-cutting are both in-frame while IMM (~2.9 m error) reads as
   locked onto ground truth. Extent follows the moving target; the sensor-origin
   marker may sit near the frame edge when the target is at long range.
4. The **camera slowly and continuously orbits** the scene at ~0.2°/frame.
5. The trajectory is drawn with **fading tails** (~75 steps ≈ 1.5 s, alpha-fading
   to zero) per line — not full persistent trails (which would be unreadable
   spaghetti in a 6-line orbiting 3D view).
6. **Raw measurements are NOT scattered** in the 3D view — at ~8 km scale the
   az/el/range noise (σ_az ≈ 0.001 rad, σ_range ≈ 2 m) is sub-pixel and would sit
   invisibly on the ground-truth line.
7. The **HUD** occupies a right-hand panel and contains, stacked:
   a. **Per-filter position-error bars** (log scale) — one per filter, updating
      each frame; the headline "who is tracking, who is lost" readout.
   b. **Per-filter NEES bars** (log scale) with the chi-square consistency band
      drawn as a shaded zone.
   c. **IMM mode-probability bars** (CV / CT+ / CT− / CA) — showing IMM deciding
      it is in a turn.
   d. **True motion-mode banner** (CV / CT / CA) for the current ground-truth
      segment.
   e. **Clock**: elapsed t (s) and step index.
8. **Vanilla KalmanNet has no covariance, so its NEES bar renders as "N/A"** (an
   explicit N/A label, never a zero bar — a zero would falsely read as perfectly
   calibrated). Its absence from the NEES panel is part of the story.
9. NEES and position-error panels are **log scale**, because values span ~10
   (IMM) to ~10⁶ (EKF); EKF's NEES bar visibly pins far above the chi-square band
   while IMM's sits inside it.
10. The animated trajectory is from the **high_maneuver regime** test set (not
    nominal), chosen because sharper turns produce more dramatic, more legible
    behavioral differences and more IMM mode-switching.
11. The specific trajectory is **auto-selected deterministically as the one with
    the most motion-mode switches** (CV↔CT↔CA transitions), so all three modes
    appear and the IMM mode bars visibly move.
12. The **full 15 s (750 steps)** of the chosen trajectory is animated.
13. Two output files are written: **mp4** (all 750 frames at 30 fps → ~25 s clip,
    ~1.7× slower than real-time) and **GIF** (every 5th frame, downscaled to
    900×506, ~12 fps → compact loop for embedding). Render frames once, write
    both.
14. **Resolution is 1600×900** (16:9): 3D scene on the left ~60%, HUD stack on the
    right ~40%.
15. mp4 output requires **ffmpeg**, which is added to
    `docker/Dockerfile.kalmannet-bench` (one `apt-get install -y ffmpeg` line) and
    the image rebuilt. GIF uses the pillow writer (already present).
16. The new script is **`tools/kalmannet_bench/make_animation.py`**, self-contained
    (its own per-frame runner recording position + NEES + IMM mode probabilities
    per step), following the same convention as the separate `eval_*.py` scripts
    rather than coupling to `make_plots.py`'s helpers.
17. Artifacts are written to **`results/video/motion_hud.mp4`** and
    **`results/video/motion_hud.gif`** (gitignored `results/`, so artifacts, not
    committed).
18. The script consumes existing on-disk dependencies: `data/splits/` (regenerate
    if absent), `results/imm_tuning.json`, `results/knet_vanilla/best.pt`,
    `results/rkn/best.pt`. It is invoked the same containerized way as everything
    else (`docker run … kalmannet-bench:cpu python make_animation.py`).

## Open questions

- None blocking. Two things to verify at build time, not decisions:
  - The trained checkpoints (`results/knet_vanilla/best.pt`, `results/rkn/best.pt`)
    are still on disk (they were generated earlier this session; confirm before
    rendering).
  - Whether 1600×900 at 750 frames renders in acceptable wall-time single-threaded
    in the container (matplotlib 3D + per-frame filter step). If too slow, the
    fallback is to decimate the mp4 too (e.g. every-2nd-frame at 30 fps → still a
    smooth ~12 s clip) — a quality/time knob, not a design change.
  - Whether ~8 km extent actually contains RKN across the whole clip. 1712 m is
    the RMSE; instantaneous peaks in high_maneuver (RKN's worst regime) may run
    2–3× that, so RKN could still momentarily clip an 8 km frame. Extent is a
    build-time tunable — enlarge if it clips, or accept the excursion (a diverged
    filter leaving the box is itself a legible "lost it" signal, per the rejected
    Q2-c alternative below).

## Steps

1. Add `ffmpeg` to `docker/Dockerfile.kalmannet-bench` and rebuild the
   `kalmannet-bench:cpu` image; verify `matplotlib.animation.writers.list()`
   now includes `ffmpeg`.
2. Confirm on-disk dependencies exist (`data/splits/`, `imm_tuning.json`, both
   `best.pt` checkpoints); regenerate `data/splits/` via `make_dataset.py` if
   absent.
3. Write `make_animation.py`:
   a. Load the high_maneuver test set; deterministically select the trajectory
      with the most mode switches from its label sequence.
   b. Per-frame runner: step EKF, UKF, IMM (with mode probs), vanilla KalmanNet
      (torch), and RKN (torch) over the trajectory, recording per step: each
      filter's position estimate, per-filter position error, per-filter NEES
      (IMM/EKF/UKF/RKN; N/A for vanilla KalmanNet), IMM mode probabilities, and
      the ground-truth mode label.
   c. Build the figure: left 3D axes (~60%, ~8 km cube following the target,
      sensor-origin marker, fading-tail lines for truth + 5 filters), right HUD
      panel with the four stacked sub-panels + banner + clock.
   d. `FuncAnimation` over 750 frames: advance orbit azimuth ~0.2°/frame, update
      each line's fading tail, update HUD bars/banner/clock from the recorded
      per-frame arrays.
4. Save `results/video/motion_hud.mp4` (ffmpeg, 30 fps, all frames) and
   `results/video/motion_hud.gif` (pillow, every-5th-frame, 900×506, ~12 fps).
5. Eyeball both outputs (Read the GIF/first frame) for legibility: verify all
   five filters + truth are in-frame at ~8 km (including RKN's worst-case
   excursions), HUD bars update sensibly, IMM mode
   bars visibly react at maneuver onset, EKF NEES pinned above the band, vanilla
   KalmanNet NEES shows "N/A".
6. If wall-time or legibility needs it, apply the fallback knobs from Open
   Questions (decimate mp4, tweak orbit rate / tail length / scene extent).

## Risks / rejected alternatives

- **Rejected: 2D top-down and az/el views** (Q1). User chose full 3D; the orbit
  is what justifies 3D over the flat top-down by exposing the pz component.
- **Rejected: showing only the classical spread (truth + IMM + EKF)** (Q2 option
  a). User wants all five filters in-frame; the ~8 km scale is the deliberate
  compromise that makes that legible, at the cost of IMM-vs-truth fine detail
  (IMM will read as "glued to truth" — acceptable, the HUD carries the
  2.9 m-vs-188 m quantitative difference for high_maneuver).
- **Rejected: nominal regime** (Q5). Chose high_maneuver for drama/legibility;
  the plan and any caption must state the video is the illustrative dramatic case
  and that the results-table numbers are nominal-regime, so the video is never
  mistaken for a cherry-picked headline metric.
- **Rejected: GIF-only to avoid touching the Docker image** (Q6). GIF's 256-color
  banding degrades the log-scale color-coded HUD; mp4 via a one-line ffmpeg add is
  cheap and preserves true color. Both are produced.
- **Rejected: full persistent trails and raw measurement scatter** (Q7). Both
  add clutter that a 6-line orbiting 3D view can't absorb; measurements are
  sub-pixel at ~8 km regardless.
- **Risk: NEES sign/zero confusion for vanilla KalmanNet.** Mitigated by decision
  8 — explicit "N/A", never a zero bar.
- **Risk: render wall-time** (matplotlib 3D, 750 frames, single-threaded).
  Mitigated by the mp4-decimation fallback; not a design change.
- **Risk: 3D depth ambiguity from a single projection.** Mitigated by decision 4
  (the orbit), which is the whole reason the camera moves.
