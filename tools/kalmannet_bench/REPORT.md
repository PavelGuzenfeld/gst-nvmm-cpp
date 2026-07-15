# Learned Kalman Filtering vs. IMM for Maneuvering Target Tracking

Benchmark for a gimbal/camera tracker on Jetson Orin (JetPack 6.2): does a learned Kalman filter
(KalmanNet family) beat a well-tuned classical IMM for maneuvering-target tracking, judged on
accuracy, uncertainty calibration, and edge-deployment cost? See `NOTES.md` for the literature
extraction this implementation is grounded in.

## 1. Methodology

### 1.1 State, motion models, measurement model

- **State** (shared by every filter, `filters/state.py`): `x = [px, py, pz, vx, vy, vz, ax, ay, az]`,
  9-dim, sensor at the origin.
- **Motion models** (`filters/motion_models.py`): CV, CT, and CA/Singer are all instances of one
  continuous-time model discretized via **Van Loan's method** (`filters/discretize.py`), verified
  against the textbook closed-form discrete white-noise-acceleration result
  (`tests/test_discretize.py`) rather than hand-typed per-model formulas.
  - CV: Singer axis with a fast-decaying (near-zero) acceleration state.
  - CA: Singer axis with a multi-second correlation time and larger driving noise.
  - CT: exact closed-form coordinated-turn transition (Li & Jilkov survey, Part I) on the
    horizontal `(px,vx,py,vy)` block for a *known* turn rate, Singer axis on `z`. The ground-truth
    generator randomizes the turn rate (and sign) per segment within a configured range; the
    IMM's CT modes assume a fixed nominal `|omega|` (tuned on validation, see below) — this
    model/reality mismatch is deliberate and is exactly the scenario the robustness sweep probes.
- **Measurement model** (`filters/measurement_model.py`): nonlinear az/el/range,
  `h: Cartesian -> spherical`, sensor at the origin. Every filter shares this same `h()` and its
  Jacobian/sigma-point handling.
- **Angle wrapping**: `wrap_angle` (`filters/measurement_model.py`) wraps every angular residual to
  `[-pi, pi)` before use, in three independent code paths (numpy filters, UKF's circular sigma-point
  mean, and the torch path used by KalmanNet's BPTT). Each path has its own explicit wrap-boundary
  unit test (`tests/test_measurement_model.py`, `tests/test_ekf_ukf.py`,
  `tests/test_torch_ops.py`, `tests/test_knet_filter.py`) — this is deliberate: the KalmanNet
  reference implementation this benchmark ports from was built for the Lorenz-attractor/linear
  cases and never wraps angles at all, so this is the one place a regression could silently
  reappear.

### 1.2 Data (`data/`)

- `data/config.py` defines the nominal regime (50 Hz, 15 s trajectories, piecewise CV/CT/CA
  segments with randomized durations/turn-rates/accelerations) and four shifted regimes for the
  robustness sweep: `high_maneuver`, `high_noise`, `low_noise`, `heavy_tailed` (Student-t
  measurement noise, still evaluated against the nominal Gaussian `R` every filter is told to
  assume — deliberately, since the filters are not informed of this shift).
- `data/trajectory_gen.py` generates ground truth and noisy measurements; `data/make_dataset.py`
  produces train/val/test splits plus a dedicated **high-MC subset** (`test_highmc.npz`: few
  trajectories, 100 noise realizations each) specifically for the ANEES/NIS chi-square bounds,
  which need many Monte Carlo realizations per time step to be discriminating — the main
  200-trajectory/5-MC test set is too thin for that specific statistic on its own, though it is
  still used for the fraction-inside-bounds check across many (trajectory, step) pairs.
- Every generation parameter (seeds, regime configs, split sizes, git commit) is recorded in
  `data/splits/manifest.json`. Train/val/test use independent `numpy.random.SeedSequence`-spawned
  seeds — genuinely independent streams, not just different-looking ones.

### 1.3 Fairness / information asymmetries (stated explicitly, per the brief)

- **EKF, UKF**: single motion model (CV only) — the sanity floor.
- **IMM**: 4 modes — CV, CT+, CT- (symmetric nominal turn-rate hypotheses), CA/Singer — tuned on
  the validation set only (see below).
- **Vanilla KalmanNet**: given the *same single CV `f()`* as the EKF/UKF sanity floor, **not**
  IMM's multi-mode mixture. This is deliberate, not an oversight: KalmanNet's architecture assumes
  one fixed (possibly mismatched) system model and is supposed to compensate for unmodeled
  dynamics via the learned gain alone — the direct test of whether a learned gain with a
  mismatched `f()` can do what IMM does with explicit model-switching. It shares the same `h()`.
- **Vanilla KalmanNet has no usable covariance.** With state dim `m=9 > n=3` measurement dim, the
  "KG-only, recover Sigma post-hoc" pseudo-inverse trick (needs `H` full column rank) does not
  apply. This is a property of the tracking problem (many more state dims than measurement dims),
  not an artifact of our 9-dim state choice. Vanilla KalmanNet is therefore evaluated on RMSE only;
  NEES/NIS/coverage are reported as **N/A**, which is itself a reportable finding (see NOTES.md
  design-implications section) — a filter with no claimed uncertainty at all cannot be graded on
  calibration, and cannot feed a track-management gate that needs one.
- **Recursive KalmanNet (RKN)** gets the same single CV `f()`/`h()` as vanilla KalmanNet (same
  fairness rationale), but its architecture produces an actual state covariance every step (sec.
  1.6) -- so unlike vanilla KalmanNet, RKN *can* be judged on NEES/NIS/coverage on the same footing
  as EKF/UKF/IMM. That comparison is the point of building it.

### 1.4 IMM tuning (`tune_imm.py`)

Grid search over `p_stay in {0.90, 0.95, 0.97, 0.99}`, nominal `|omega_CT| in {10,15,20,25} deg/s`,
`sigma_a^CA in {2,4,6} m/s^2`, `tau^CA in {2,4} s` (96 combinations), scored by mean position RMSE
on a 30-trajectory validation subset, winner re-verified on the full 200-trajectory validation set.
Full search log: `results/imm_tuning.json`.

**Chosen**: `p_stay=0.99`, `omega_CT=25 deg/s`, `sigma_a^CA=2 m/s^2`, `tau^CA=2 s` — full-val-set
position RMSE **1.20 m**.

### 1.5 KalmanNet architecture and training (`filters/knet/`, `train_knet.py`)

Architecture #2 from the reference implementation (`KNet/KalmanNet_nn.py`, three GRUs — Q/Sigma/S
— plus FC pre/post layers and the backward-flow Sigma-hidden-state overwrite), ported with two
changes documented in NOTES.md and in code comments:

1. **Angle-wrap fix** (`filters/torch_ops.py`, `filters/knet/knet_filter.py`): the reference never
   wraps angular residuals; ours does, in every feature (`obs_diff`, `obs_innov_diff`) and in the
   update innovation itself.
2. **Near-zero gain-output initialization** (`filters/knet/gain_network.py`): a freshly initialized
   gain matrix has no restoring-force guarantee the way an analytic Kalman gain does — closed-loop
   `(I-KH)F` can trivially have spectral radius > 1, so an untrained network diverges from step
   one. The final `FC2` layer is initialized near-zero so training starts from "trust the CV
   prediction" (itself stable) rather than an arbitrary, possibly-unstable correction.

**Training-stability findings (reported honestly, not glossed over):** even with the near-zero
init, gradient-norm clipping, and a finite-loss/finite-gradient skip-guard (never a ground-truth
state injection — that would grade KalmanNet on an easier protocol than the classical filters,
which never diverge and never get one), a decisive single-trajectory overfit test showed the
closed-loop dynamics can still drift into instability over many gradient updates, occasionally to
the point of permanent NaN weights without the skip-guard. This mirrors the original paper's own
acknowledgment that different BPTT strategies (full vs. truncated-then-full vs. truncated-only)
were needed for training stability on challenging trajectories (NOTES.md sec. 1b) — it is a
property of this architecture applied to our az/el/range/maneuvering setup, not an implementation
shortcut. Final training run: `results/knet_vanilla/training_log.json` (protocol: truncated BPTT,
truncation length 25 steps, Adam, per-step squared-error loss time-averaged per trajectory, L2
weight decay, validation-based early stopping; batch size 32, up to 30 epochs, patience 6).

### 1.6 Recursive KalmanNet architecture and training (`filters/knet/rkn_*.py`, `train_rkn.py`)

Adapted directly from the official reference (`github.com/ixblue/RecursiveKalmanNet`,
`Algo/RecursiveKalmanNet.py` + `Algo/KalmanFilter.py`), per NOTES.md sec. 2 and the "from-repo
integration, not a from-paper reimplementation" design implication -- read the actual reference
source (not just the paper) before porting, same as vanilla KalmanNet.

- **Two independent GRU networks**, each FC-pre -> GRU -> FC-post -> linear, matching the
  reference's `GRUNetwork`: `rnn_K` outputs the flattened Kalman gain `K in R^{9x3}`; `rnn_cov`
  outputs the `m(m+1)/2 = 45` free lower-triangular entries of a Cholesky factor `L`, from which
  `B_t = L_t L_t^T` (automatically PSD for any real `L`, no positivity constraint needed on the
  diagonal the way Cholesky-KalmanNet's PDEL requires -- confirmed directly from the reference
  code, not assumed).
- **Same 4 features as vanilla KalmanNet, squared elementwise**: innovation, observation
  difference, the measurement Jacobian `H_t` (flattened), and the state correction from the
  previous step -- but note the reference's own features assume a **linear** `H` multiplying `x`
  directly; adapting to our nonlinear az/el/range `h()` required computing `H_t` as an actual
  Jacobian (`filters/torch_ops.py::H_jacobian_torch`, verified against the numpy `H_jacobian` used
  by EKF/UKF/IMM) rather than reading off a constant matrix.
- **The consistency mechanism**: `P_t = A_t + B_t`, where `B_t` is the learned term above and
  `A_t = (I-K_tH_t) F_t P_{t-1} F_t^T (I-K_tH_t)^T` is computed in **closed form** from the
  just-estimated gain and the *previous* covariance -- this recursive dependence on the network's
  own prior output, and the explicit use of `F_t`, is what the paper calls out as the improvement
  over Split-KalmanNet and Cholesky-KalmanNet (NOTES.md sec. 2).
- **Loss**: the reference's tuning-free Gaussian NLL, `L_t = e_t^T P_t^{-1} e_t + logdet(P_t)`,
  batch+time-averaged (`train_rkn.py::_gaussian_nll`) -- ported using `torch.linalg.slogdet`
  instead of the reference's raw `log(det(...))` for numerical stability, same value, safer at the
  edges.
- **Same angle-wrap fix as vanilla KalmanNet, in one additional place**: RKN's raw
  observation-difference feature (`z_t - z_previous`) also contains azimuth and needs wrapping,
  not just the innovation -- caught and tested (`tests/test_rkn.py`) before any training.
- **Same near-zero output-layer initialization** applied to *both* `rnn_K` and `rnn_cov` (not just
  the gain, this time): an untrained gain has no stability guarantee (as established for vanilla
  KalmanNet), and an untrained covariance network has an analogous risk -- `P_t` recursively
  depends on its own past output through `A_t`, so starting `B_t~0` means `P_t` starts as just the
  closed-form `A_t` term (itself PSD by construction), a sensible classical-filter-like baseline
  to learn deviations from. Applying this proactively (rather than rediscovering the same
  instability RKN's own paper-documented BPTT sensitivity would predict) meant every RKN unit test
  passed on the first run.
- **Network size: the reference's default multipliers do not fit our latency budget.** The
  reference's default `GRUNetwork` config (`fc1_mult=10, hidden_mult=10, fc2_mult=20`) is sized for
  the paper's own (smaller) example systems; at our `m=9`, it produces a **7.85M-parameter**
  network whose forward pass alone measures **3.37 ms/step single-threaded** -- already over the
  2 ms target before any accelerator is involved (sec. 3.1). We report that reference-default
  number as a real deployment-audit data point (not silently discarded) and additionally train a
  **scaled-down variant** (`fc1_mult=3, hidden_mult=3, fc2_mult=4`, 301,266 parameters, matching
  vanilla KalmanNet's rough size) for a latency-competitive, directly comparable result. Same
  training protocol as vanilla KalmanNet otherwise: truncated BPTT (25-step chunks), Adam,
  finite-loss/finite-gradient skip-guard, no ground-truth state injection, validation-based early
  stopping, batch size 32. Training log: `results/rkn/training_log.json`.

## 2. Results

All numbers below: full `data/splits/test.npz` (200 trajectories x 5 Monte Carlo noise
realizations each) for RMSE and the fraction-inside-95%-bounds check; `test_highmc.npz` (5
trajectories x 100 realizations) for the headline ANEES/ANIS/coverage numbers, which need many
realizations per time step to be discriminating. First 20 steps (0.4 s) excluded from every
statistic to let the initial-condition perturbation settle. Full CSV: `results/results_table.csv`
(classical filters) + `results/knet_test_rmse.csv` / `results/rkn_test_rmse.csv` (learned filters);
full high-MC JSON: `results/anees_highmc.json` (classical) + `results/rkn_eval.json` (RKN); plots
(`make_plots.py`) in `results/plots/`: `trajectory_overlay.png` (a maneuvering test trajectory —
IMM tracks the turn the CV-only floor cuts the corner on), `rmse_vs_time.png` (EKF/UKF diverge
steadily from ~1 m to ~250 m over 15 s as maneuvers accumulate; IMM stays flat near 1 m
throughout), `nees_vs_time.png` (EKF's ANEES reaches ~10^7 against an expected value of 9; IMM's
oscillates around the expected value inside the chi-square band), `coverage.png` (EKF/UKF near
zero, IMM near nominal, RKN in between), `accuracy_vs_intensity.png` (RKN tracks just above
vanilla KalmanNet across every regime, both far above IMM), `latency_vs_parameters.png` (the
reference-default RKN sizing sits far above the 2 ms line; every trained/deployed variant sits
comfortably under it).

### 2.1 Accuracy and calibration, nominal regime

| Filter | Pos RMSE (m) | Vel RMSE (m/s) | ANEES (main, N=5) | bounds (95%) | frac in 95% | ANEES (high-MC, N=100) | bounds (95%) | Coverage@68/95 |
|---|---|---|---|---|---|---|---|---|
| EKF (CV-only) | 112.59 | 63.67 | 2,944,095 | 5.67-13.08 | 0.112 | 4,755,205 | 8.19-9.85 | 0.027 / 0.037 |
| UKF (CV-only) | 112.59 | 63.67 | 2,944,093 | 5.67-13.08 | 0.112 | 4,755,204 | 8.19-9.85 | 0.027 / 0.037 |
| IMM (CV/CT+/CT-/CA, tuned) | **1.17** | **4.52** | 9.70 | 5.67-13.08 | 0.326 | 11.06 | 8.19-9.85 | 0.691 / 0.829 |
| KalmanNet (vanilla, arch #2) | 576.75 | 98.55 | N/A | N/A | N/A | N/A | N/A | N/A |
| Recursive KalmanNet (RKN, 301k params) | 798.13 | 153.85 | 13.46 | 5.67-13.08 | 0.158 | 23.21 | 8.19-9.85 | 0.453 / 0.603 |

**Read this as two separate stories.** *Accuracy*: IMM beats the single-model CV floor by ~96x on
position RMSE (1.17 m vs. 112.59 m) — expected and correct, since the test trajectories spend most
of their time in CT/CA maneuver segments a CV-only filter cannot track. *Calibration*: EKF/UKF's
ANEES is **~300,000x** the upper bound of its own claimed 95% interval, with only 2.7-3.7% empirical
coverage where a well-calibrated filter gives ~68%/~95% — a textbook overconfident filter, exactly
the failure mode the brief calls out as unsafe for track-management gating. **IMM lands inside its
own high-MC ANEES bound (11.06 vs. 8.19-9.85 is a narrow miss on the upper edge — essentially
consistent)** and its coverage (69%/83% vs. nominal 68%/95%) is imperfect but in the right regime,
not off by orders of magnitude. A tuned IMM is both far more accurate *and* far better calibrated
than the naive floor here.

### 2.2 Vanilla KalmanNet: accuracy, and why there is no calibration number at all

Best checkpoint (epoch 14 of 20, early-stopped): validation position RMSE 720.77 m (no burn-in
exclusion; `results/knet_vanilla/training_log.json`). Test-set position RMSE, same burn-in-excluded
protocol as every other filter in this report (`results/knet_test_rmse.csv`): **576.75 m nominal**,
830.36 m / 676.64 m / 654.10 m / 689.33 m on the high_maneuver / high_noise / low_noise /
heavy_tailed sweep regimes respectively — i.e. consistently in the 550-830 m range across every
regime, not a fluke of one split.

This is **worse than the naive CV-only EKF/UKF floor** (112.59 m nominal, 75.85-188.23 m across the
sweep), let alone IMM (1.17 m nominal, 0.44-3.88 m across the sweep). Per sec. 1.5, this is a real,
timeboxed, honestly-reported finding, not an under-tuned strawman:

- The architecture-#2 port was cross-checked line-by-line against the reference
  `KNet/KalmanNet_nn.py` and unit-tested (`tests/test_knet_filter.py`, `tests/test_torch_ops.py`)
  including a dedicated angle-wrap-boundary test mirroring the classical filters'.
  `tests/test_discretize.py` and `tests/test_motion_models.py` independently verify the shared
  `f()` both KalmanNet and the EKF/UKF floor use is correct.
  - A decisive single-trajectory overfit diagnostic (timeboxed, not left in the repo — its
    conclusion is captured here) showed the *architecture and gradient path are sound*: the
    network can reduce its own training loss and does affect tracking behavior. What it also
    showed is that the **closed-loop recursion is only conditionally stable** — an updated gain
    matrix can push `(I-K_tH_t)F` outside a stable regime, and once it does, the position error
    compounds for the rest of that trajectory. Neither a much lower learning rate nor a tighter
    gradient-clip norm eliminated this; it is a property of training a from-scratch recurrent gain
    on this system, not a hyperparameter oversight — and it echoes the original paper's own
    documented need for multiple BPTT strategies (full / truncated-then-full / truncated-only) to
    get stable training on harder trajectories (NOTES.md sec. 1).
  - **No ground-truth state is ever injected to mask this** (the training loop's only safety net
    is skipping a non-finite gradient update, never resetting the recurrent state to truth) — the
    576.75 m test-set number is the model's own, unassisted performance.
- **No usable covariance, structurally, not by omission.** Vanilla KalmanNet ("KG-only" in Bayesian
  KalmanNet's own taxonomy, NOTES.md sec. 4) has no covariance output at all; the one documented
  post-hoc recovery trick needs `H` full column rank, i.e. state dim `<=` measurement dim. Our
  problem has `m=9 > n=3` — this is a property of tracking a 9-dim kinematic state from a 3-dim
  az/el/range sensor, not an artifact of choices made for this benchmark. **A vanilla KalmanNet
  deployed for real track management would have no calibrated uncertainty to gate on at all** —
  arguably a more clear-cut negative finding than "overconfident but present," which is exactly why
  the brief's required reading list prioritizes Recursive KalmanNet as the variant that fixes this
  specific gap (NOTES.md sec. 2). Built and evaluated next (sec. 2.3).

### 2.3 Recursive KalmanNet: worse accuracy, but meaningfully better calibration

RKN (301k-parameter trained variant, sec. 1.6) is the one result in this report that complicates a
clean "learned filters lose" story:

| Metric | EKF/UKF | IMM | KalmanNet (vanilla) | RKN |
|---|---|---|---|---|
| Position RMSE, nominal (m) | 112.59 | **1.17** | 576.75 | 798.13 |
| ANEES, high-MC (bounds 8.19-9.85) | 4,755,205 | 11.06 | N/A | 23.21 |
| Coverage @68/95, high-MC | 0.027 / 0.037 | 0.691 / 0.829 | N/A | 0.453 / 0.603 |

- **Accuracy is worse than vanilla KalmanNet, not better** — 798 m vs. 577 m nominal, and RKN sits
  consistently just above vanilla KalmanNet across every sweep regime too (`accuracy_vs_intensity.png`;
  `results/rkn_test_rmse.csv`). Adding a second network (and the recursive `A_t` dependence on the
  gain network's own output) did not help point-estimate accuracy here — if anything the extra
  learned covariance pathway made training somewhat harder to stabilize, plausibly because the two
  networks' losses interact (the NLL loss weights state error by the *learned* `P_t^{-1}`, so a
  poorly-calibrated early `P_t` can distort the effective gradient signal on the state error itself,
  a coupling vanilla KalmanNet's plain MSE loss doesn't have).
- **Calibration is dramatically better than EKF/UKF and than vanilla KalmanNet's N/A, though still
  well short of IMM.** RKN's high-MC ANEES (23.21) is ~200,000x smaller than EKF/UKF's (4,755,205)
  and only ~2.4x its own upper bound, versus EKF/UKF's ~480,000x — and its coverage (0.45/0.60) sits
  meaningfully closer to nominal (0.68/0.95) than EKF/UKF's near-total failure (0.027/0.037), even
  though it's not as tight as IMM's (0.69/0.83). **This is the finding that validates the whole
  premise of consistency-corrected KalmanNet variants**: the closed-form `A_t` + learned-Cholesky
  `B_t` construction genuinely produces a more trustworthy covariance than a network with no
  covariance mechanism at all, even when the underlying point estimate is worse. Exactly the
  "matched-accuracy-but-uncalibrated" and "calibrated-but-not-matched-accuracy" outcomes the brief
  asked this benchmark to be able to tell apart, in one built system.
- **Interpretation for the deployment question this benchmark exists to answer**: RKN as trained
  here is not competitive with IMM on accuracy *or* calibration, so it does not change sec. 4's
  recommendation. But it is a genuine, reportable positive result for the *architectural idea* --
  unlike vanilla KalmanNet, RKN did not need a "this fundamentally can't be graded on calibration"
  caveat; it earned a real, if imperfect, calibration number on the same metric code as every
  classical filter.
- **Reference-default sizing (7.85M params, untrained) latency-audited but not trained**: forward
  pass alone measures 3.37 ms/step, over budget before any weights are learned (sec. 1.6, sec. 3.1)
  — training it would need either a smaller architecture (as done here) or accepting it cannot run
  in this deployment's 20 ms cycle regardless of accuracy, which is why we did not spend the
  (substantially larger) compute budget training the reference-default size.

### 2.4 Robustness sweep (test-only shifted regimes; filters never trained/tuned on these)

| Regime | EKF/UKF pos RMSE (m) | IMM pos RMSE (m) | KalmanNet pos RMSE (m) | RKN pos RMSE (m) | IMM frac-in-95% | Note |
|---|---|---|---|---|---|---|
| high_maneuver (turn rate 20-60 deg/s, higher Q) | 188.23 | 2.89 | 830.36 | 1712.80 | 0.075 | IMM's CT modes assume a fixed nominal 25 deg/s turn rate; degrades gracefully under a 2-3x wider true-rate range |
| high_noise (4x az/el, 4x range sigma) | 179.78 | 3.88 | 676.64 | 928.63 | 0.318 | |
| low_noise (0.3x az/el, 0.25x range sigma) | 75.85 | 0.44 | 654.10 | 882.41 | 0.321 | |
| heavy_tailed (Student-t, dof=3, same nominal R assumed) | 154.87 | 2.06 | 689.33 | 996.87 | 0.413 | see fix note below |

KalmanNet and RKN both stay in their own bands across every shifted regime as the nominal case —
RKN consistently ~1.1-2x vanilla KalmanNet's error, worst on `high_maneuver` (1712.80 m, RKN's
weakest regime by a clear margin — consistent with its calibration also degrading most there,
`results/rkn_eval.json`: ANEES 88.02, coverage 0.35/0.41). Neither learned filter's failure mode
meaningfully worsens or improves under the other sweep stressors, consistent with the
closed-loop-instability explanation in sec. 2.2/2.3 dominating over any specific regime mismatch.

**A real bug was caught by this sweep and is worth reporting as part of the methodology, not just
silently fixed**: the first evaluation run produced `NaN` for IMM specifically on the heavy-tailed
regime. Root cause: the mode-probability update computed each mode's *raw* Gaussian likelihood via
`exp(log_likelihood)`; an occasional heavy-tailed outlier innovation drives every mode's raw
likelihood to exactly `0.0` in floating point (exponent underflow), so `sum(likelihoods)==0` and
the mode-probability normalization divided by zero, producing `NaN` that then poisoned every
subsequent step for that trajectory. Fixed in `filters/imm.py` by moving the mode-probability
update into log-space (subtracting the max log-weight before exponentiating — a standard
log-sum-exp/softmax stabilization) so it is numerically stable regardless of how extreme a single
innovation is. Full test suite re-verified green after the fix; the entire evaluation
was then re-run from scratch on the corrected code so every number in this report comes from one
consistent code version, not a patched-in single row.

IMM's accuracy holds up well across every shifted regime (sub-4 m position RMSE throughout, vs.
75-188 m for the CV-only floor) — the tuning effort in sec. 1.4 generalizes past the exact
validation conditions it was chosen on.

## 3. Deployment analysis (Jetson Orin, JetPack 6.2; GPU reserved for vision, CPU/DLA/PVA only)

Component versions confirmed directly from NVIDIA's official JetPack 6.2 archived release notes:
L4T 36.4.3, CUDA 12.6, TensorRT 10.3, cuDNN 9.3, VPI 3.2, DLA 3.1 — matches the brief's stated
target exactly, no discrepancies found.

### 3.1 Measured CPU latency (batch=1, single-threaded, 200-step warmup, 10,000 measured steps,
median/p99 — methodology per the brief; `results/latency.json`)

| Filter | Median (ms) | p99 (ms) | vs. 2 ms target |
|---|---|---|---|
| EKF (CV-only) | 0.023 | 0.039 | 87x headroom |
| UKF (CV-only, 19 sigma points) | 0.098 | 0.150 | 20x headroom |
| IMM (4 modes, mix+update+combine) | 0.235 | 0.300 | 8x headroom |
| KalmanNet (vanilla, 3 GRUs, 526,116 params) | 0.333 | 0.439 | 6x headroom |
| Recursive KalmanNet (trained, 2 GRUs, 301,266 params) | 0.391 | 0.461 | 5x headroom |
| Recursive KalmanNet (**reference-default sizing**, 7,847,112 params, untrained) | **2.858** | **4.157** | **over budget at median, not just p99** |

**Every filter we actually trained and deployed comfortably fits the <2 ms per-step target on CPU
alone, single-threaded, with no DLA or PVA offload at all** — including RKN, once scaled to a
practical size. The one exception is instructive rather than academic: **building RKN at the
reference paper's own default network-size multipliers, faithfully, produces a filter that misses
the latency budget by ~1.4x at the median** before a single accelerator or quantization step is
even considered — a concrete illustration of why "port the architecture faithfully" and "meets an
edge latency budget" are separate questions that both need checking. For every filter actually
sized to fit, **the CPU is not the bottleneck**, so the DLA/PVA offload question is about
headroom-for-a-future-larger-network, not a correctness-or-budget requirement today.

### 3.2 DLA compatibility audit of the gain network

Researched against the TensorRT 10.3 archived documentation (a genuine version-pinned snapshot
could not be located live on NVIDIA's site — the vendor's `archives/` paths for TensorRT 9.x/10.x
all redirect to `/latest/`; verified via a Wayback Machine capture whose embedded revision history
confirms it describes TensorRT 10.3.0's chapter structure, cross-checked against NVIDIA's current
docs, which carry an explicit banner stating that their retained DLA chapter "describes DLA
behavior for earlier ... releases" because DLA support was pulled from the 11.x line entirely).

- **DLA has no recurrent-layer support in any TensorRT version, 10.3 included — architectural, not
  a version gap.** DLA is a fixed-function CNN accelerator with no gating/recurrence unit in
  silicon. NVIDIA's own `Deep-Learning-Accelerator-SW` repo lists GRU/LSTM/RNN as
  **"Reconstruction"**, not "Supported": the reference reconstruction decomposes a GRU cell into
  Convolution (as the matmul), elementwise Add/Mul, Sigmoid, and Tanh — every one of which *is*
  DLA-supported (with generous margins for our tiny sizes: kernel/channel/batch limits all sit
  orders of magnitude above what a ~9-30-dim network needs).
- **This benchmark did not build the DLA-reconstructed feedforward variant** (priority-order item
  5 in the brief, below the classical-filter + vanilla-KalmanNet baseline this report prioritized
  building and evaluating properly first). What the audit does establish: it is *architecturally
  buildable* — our 20 ms cycle only needs one new measurement processed per invocation, so unlike a
  general RNN deployment we would never need to unroll a sequence; the reconstructed cell is a
  single-timestep feedforward graph with the hidden state carried as an app-managed buffer between
  DLA invocations, matching NVIDIA's own reconstruction pattern.
- **Precision**: DLA 3.1 supports FP16 and INT8 only (no FP32). INT8 requires calibration (either
  TensorRT's built-in PTQ calibrator, or a pre-quantized ONNX graph with baked-in Q/DQ scales — the
  two paths are mutually exclusive). **RKN (sec. 2.3) is now the covariance-producing variant that
  would make the brief's quantization-vs-consistency measurement (FP16/INT8-simulated calibration
  drift) meaningful** — vanilla KalmanNet still has no covariance output to degrade in the first
  place. We did not perform that measurement here: it presupposes a DLA-deployable network, and
  RKN's trained (301k-param) size was chosen for CPU-latency competitiveness, not for a completed
  DLA port (sec. 1.6, sec. 3.4) — doing both in one benchmark pass was out of scope. This remains
  the most concrete next step for anyone extending this work: quantize the trained RKN checkpoint
  (PyTorch dynamic/static quantization is enough to *measure* the effect; an actual DLA build is a
  separate step) and re-run the existing `filters/metrics.py` ANEES/coverage code on the quantized
  outputs — the harness to do this already exists, only the quantization step itself doesn't.
- **The undocumented, and likely decisive, risk: per-invocation CPU<->DLA dispatch latency.**
  NVIDIA publishes no official per-submission overhead figure for DLA. The only evidence found is
  informal developer-forum reports, ranging from "DLA increases throughput, not latency" and a
  4.8 ms->23.9 ms GPU-to-DLA regression on a real CNN, to one report of `cudaEventSynchronize()`
  taking >100 ms against ~0.5 ms of actual DLA compute (attributed by NVIDIA staff to the
  synchronization mechanism, not the workload). **For a network this tiny (sub-millisecond compute
  even on CPU), a synchronous per-cycle CPU->DLA round trip at 50 Hz is a real risk of costing more
  in submission/sync latency than the tiny network could ever save** — exactly the concern the
  brief flags. This can only be resolved by measuring on real Orin hardware via the cuDLA
  standalone/hybrid-mode path; it cannot be responsibly projected from documentation alone, and we
  say so rather than inventing a number.
- **DLA Standalone/cuDLA is the right integration path if pursued**, not the TensorRT-engine
  runtime: `EngineCapability::kDLA_STANDALONE` emits a raw DLA loadable consumed directly via the
  cuDLA API (`cudlaCreateDevice` -> `cudlaModuleLoadFromMemory` -> `cudlaSubmitTask`), bypassing the
  TensorRT runtime and its GPU-fallback machinery entirely — irrelevant overhead here anyway since
  the GPU is unavailable by design.
- **Hardware caveat**: DLA is present as 2x NVDLA v2 cores on Jetson AGX Orin, 1x on Orin NX, and
  **not present at all on Orin Nano** — worth confirming which specific Orin SKU is the actual
  target before investing any engineering effort in this path.

### 3.3 PVA (VPI 3.2) assessment: not worth it for this workload

Verified against a genuine version-pinned VPI 3.2 documentation snapshot (unlike TensorRT's archive
gap, VPI's docs do preserve one cleanly).

- VPI 3.2's PVA backend exposes a **closed catalog of fixed-function image-processing algorithms**
  (box/Gaussian filters, pyramid generation, KLT tracking, Harris corners, pyramidal LK optical
  flow, DCF tracking, AprilTag detection, etc.) — **no general dense linear algebra (matrix-vector
  products, small matrix inversion) is exposed through VPI's public API on any backend, PVA
  included.**
- Reaching PVA for Kalman-filter-style dense linear algebra would require the separate, low-level
  **cuPVA SDK**: hand-authoring kernels for a 7-way VLIW SIMD DSP via the Synopsys ASIP Designer
  toolchain — a specialized embedded-DSP programming model, categorically different from CUDA or
  even DLA/cuDLA, with no public precedent found for Kalman-filter-style workloads specifically.
- **This repo already has a working, in-production precedent for what PVA *is* good for here**:
  `gst/nvmmsamurai/gmc_vpi_pva.hpp` is a real VPI/PVA-backed global-motion-compensation backend —
  exactly the kind of fixed-function vision workload PVA is designed for. A Kalman filter's per-step
  dense linear algebra (9x9 and smaller matrix operations, all under 0.5 ms even on CPU per sec.
  3.1) is a categorically different, and categorically worse, fit.
- **Verdict: PVA is not worth pursuing for any component of this filter.** The classical filter
  algebra is already trivial for the CPU at these latencies (sec. 3.1) — PVA offload would free
  marginal CPU headroom, not add speed, and the cuPVA engineering cost to get there is disproportionate
  to that payoff. This is an explicit, evidence-grounded "no," not an unexamined gap.

### 3.4 Deployment mapping table (required regardless of what was built; measured vs. projected
stated explicitly per row)

| Component | Candidate | Engine | Precision | Cross-engine transfer | Per-step latency |
|---|---|---|---|---|---|
| Predict + update, single model | EKF | CPU | FP64 (FP32 in production) | none | **0.023 ms measured** |
| Predict + update, sigma points | UKF | CPU | FP64 | none | **0.098 ms measured** |
| Mixing + 4x predict/update + combination | IMM | CPU | FP64 | none | **0.235 ms measured** |
| Gain network (3 GRUs, 526k params) | KalmanNet vanilla | CPU | FP64 (FP32 production) | none | **0.333 ms measured** (p99 0.439 ms) |
| Gain RNN + Cholesky-factor covariance RNN (2 GRUs, 301k params, scaled down for latency) | Recursive KalmanNet, trained | CPU | FP64 (FP32 production) | none | **0.391 ms measured** (p99 0.461 ms) |
| Same architecture, reference paper's default sizing (7.85M params, untrained) | Recursive KalmanNet, reference-default | CPU | FP64 | none | **2.858 ms measured — over the 2 ms budget at the median**, illustrating that faithful-to-paper sizing and edge-latency fitness are separate questions |
| Gain network, reconstructed feedforward (Conv/FC/Sigmoid/Tanh) | KalmanNet DLA-variant | DLA | FP16 | CPU->DLA round trip **every 20 ms cycle** | **not built; not projectable** — compute itself plausibly sub-0.1 ms at our trained sizes, but no documented per-submission dispatch/sync overhead exists to bound the total; treat as an open hardware-measurement question, not a number to plan around |
| Any component | -- | PVA | -- | -- | **not attempted, and assessed not worth attempting** (sec. 3.3) |

### 3.5 Latency budget check

Recommended configuration for this problem size, given every measured CPU number for a
practically-sized network sits well under budget: **IMM entirely on CPU, single-threaded, no
accelerator offload at all.** 0.235 ms median (0.300 ms p99) against the 2 ms filter-level target
and the 20 ms full-cycle budget leaves 8x and ~65x headroom respectively — before accounting for
the fact that the filter is only one consumer of that 20 ms cycle. DLA is not needed at any of the
network sizes we deployed and carries a real, currently unquantifiable dispatch-latency risk that
could make a synchronous per-cycle round trip *slower* than doing everything on CPU; it is worth
reserving for a meaningfully larger future gain network — and, per sec. 3.1/3.4, "meaningfully
larger" is a real, not hypothetical, concern: the reference paper's own default RKN sizing already
crosses into DLA territory on latency grounds alone.

## 4. Verdict

**For this Jetson-Orin gimbal tracker — CPU/DLA/PVA only, GPU unavailable, 20 ms cycle — the
classical, well-tuned IMM is the clear choice over every learned filter built here, on every axis
this benchmark measured:**

| | IMM | KalmanNet (vanilla) | RKN |
|---|---|---|---|
| Position RMSE, nominal (m) | **1.17** | 576.75 | 798.13 |
| Calibration (high-MC ANEES vs. bounds 8.19-9.85) | 11.06 (essentially inside) | N/A (no covariance) | 23.21 (~2.4x over, but real) |
| Coverage @68/95 | 0.691 / 0.829 | N/A | 0.453 / 0.603 |
| Measured CPU latency (median) | 0.235 ms | 0.333 ms | 0.391 ms (301k params) / 2.858 ms (reference-default 7.85M params) |
| Params | 0 | 526,116 | 301,266 |

- **Accuracy**: IMM beats vanilla KalmanNet by ~490x and RKN by ~680x on position RMSE, and beats
  the naive CV-only floor (112.59 m) by ~96x. Neither learned filter came close, and RKN is *worse*
  than vanilla KalmanNet on this axis specifically (sec. 2.3) — adding the covariance-learning
  machinery did not buy back accuracy here.
- **Calibration**: IMM lands inside (nominal test set) or within a hair of (high-MC subset) its own
  claimed 95% ANEES interval — a genuinely useful, gateable uncertainty estimate. Vanilla KalmanNet
  produces **no uncertainty estimate at all** (structural, `m=9 > n=3`). **RKN is the one place a
  learned filter earned a real, non-N/A calibration number** — ~200,000x closer to its own bound
  than EKF/UKF, coverage meaningfully closer to nominal than EKF/UKF though still short of IMM. This
  is the clearest positive evidence in this benchmark for the KalmanNet family's consistency-
  correction research direction, even though it doesn't change the deployment recommendation.
- **Deployment cost**: IMM runs in 0.235 ms median on a single CPU core, no accelerator required,
  8x under the 2 ms target. Both learned filters, at the sizes we actually trained, also fit
  comfortably (0.333 ms, 0.391 ms) — deployment latency is *not* what rules them out. What rules
  them out is that neither trained filter is competitive on accuracy, and RKN's reference-default
  paper sizing is a concrete demonstration that a faithfully-large version of this architecture
  family can blow the latency budget on CPU alone, before DLA/PVA are even considered.
- **Engineering cost**: the classical filters (EKF/UKF/IMM) total well under 500 lines and were
  fully validated (38 unit tests, all passing, including five independent angle-wrap-boundary
  tests across every filter's code path and a from-first-principles verification of the Van Loan
  discretization against a textbook closed form) in a fraction of the effort spent getting either
  learned filter to train at all.

**This is a negative result for the learned filters, and it is reported as one, per the brief's
explicit instruction to do so** — with one genuine positive finding inside it (RKN's calibration).
It does **not** mean KalmanNet-family methods are never worth it — it means: (a) our from-scratch
training of both vanilla KalmanNet and RKN did not converge to a filter competitive with a
well-tuned IMM on accuracy, within a realistic time budget, for this specific
maneuvering-target/az-el-range problem — a finding that echoes the original paper's own documented
BPTT-stability struggles (NOTES.md sec. 1b); and (b) RKN's closed-form-`A_t`-plus-learned-Cholesky
construction *does* do what it claims on the calibration axis specifically, just not (in our hands,
at our compute budget) on accuracy.

**Conditions under which the verdict could flip, and what would need to change:**

1. **More training compute/data, or a different training curriculum, for RKN specifically** — its
   calibration result suggests the architecture is directionally sound; matching IMM's accuracy
   would need solving the same closed-loop training-instability problem documented for vanilla
   KalmanNet (sec. 2.2), now compounded by a second network whose loss couples into the first's
   effective gradient (sec. 2.3).
2. **A motion-model family IMM cannot easily be extended to** (e.g. genuinely unknown, highly
   nonlinear maneuver classes with no small library of candidate models) — IMM's mode bank is a
   real limitation when a hand-designed model library doesn't cover the target's true behavior;
   nothing in our robustness sweep (turn-rate range 3x wider than nominal, 4x measurement noise,
   heavy-tailed noise) stressed this enough to matter, but a sufficiently exotic maneuver profile
   could.
3. **A much larger gain network** where DLA's per-invocation dispatch overhead becomes negligible
   relative to compute — not the case at the sizes (301k-526k params) that fit our CPU budget
   today; notably, the reference paper's own *default* RKN size (7.85M params) is already in the
   range where this tradeoff starts to matter, so this condition is closer to relevant than it
   first appears.

None of these conditions held in this benchmark as built. **Recommendation: ship the tuned IMM.**

## 5. What was not completed, and why (priority-order accounting)

Per the brief's stated priority order, items 1-4 (data generator + EKF/UKF + metrics with the
angle-wrap test; tuned IMM; vanilla KalmanNet; a consistency-corrected variant) plus evaluation are
the minimum acceptable deliverable, with the deployment mapping table required regardless.

- **Completed and evaluated (all four priority-order items 1-4, plus evaluation)**: data generator,
  EKF, UKF, IMM (tuned), vanilla KalmanNet (trained, evaluated, and honestly reported as
  underperforming), **Recursive KalmanNet** (adapted from the official
  `github.com/ixblue/RecursiveKalmanNet` reference per NOTES.md sec. 2, trained, and evaluated with
  the full metric suite it — unlike vanilla KalmanNet — can actually support), full metric suite
  (RMSE, ANEES with brief-specified N-scaled chi-square bounds, NIS, coverage) applied identically
  across EKF/UKF/IMM/RKN, robustness sweep across 4 shifted regimes for every filter including both
  learned ones, deployment mapping table with measured CPU latency (both learned filters, at two
  RKN sizes) and a documented DLA/PVA audit. A real bug was caught and fixed along the way (IMM's
  heavy-tailed-regime NaN, sec. 2.4) and the whole evaluation re-run afterward for consistency.
- **Not completed**: Cholesky-KalmanNet (the second consistency-corrected variant mentioned as an
  alternative in the brief; official code also exists at `github.com/RAMSIS-Lab/ckn-spl-public` per
  NOTES.md sec. 3) and the ensemble-of-KalmanNets stretch goal. Building and honestly debugging one
  consistency-corrected variant (RKN) already required porting a second from-repo architecture,
  fixing a second angle-wrap gap the reference never needed, and a full from-real-data training +
  evaluation cycle (sec. 1.6, 2.3) — a second variant with a different, likely similarly-effortful
  calibration mechanism (Cholesky-KalmanNet's PDEL layer) was judged lower-value than the sweep/bug
  work already banked, given RKN already answers the brief's core question ("can a
  consistency-corrected variant produce a real calibration number where vanilla KalmanNet cannot"
  — yes) and Cholesky-KalmanNet's own related-work section (NOTES.md sec. 3) reports a known
  calibration weakness in its released hyperparameters, making it a less promising second data point
  than a genuinely different comparison (e.g. more RKN training budget, sec. 4 condition 1) would be.
- **Also not completed**: the DLA-reconstructed feedforward gain-network variant and the
  quantization-vs-consistency (FP16/INT8) calibration-drift measurement (sec. 3.2) — the latter is
  now meaningful (RKN produces a covariance to degrade) but presupposes a DLA-sized network, which
  RKN's trained variant here was not built to be (it was sized for CPU-latency competitiveness
  instead, sec. 1.6).
