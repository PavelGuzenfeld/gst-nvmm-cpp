# Source notes: KalmanNet-family vs. classical IMM for maneuvering-target tracking

Research notes for the benchmark comparing learned Kalman filtering (KalmanNet family) against a
classical tuned IMM filter, for a gimbal/camera tracker (az/el/range measurements, 50 Hz). Written
before any code, per repo convention ("read the source, don't work from memory"). Every section
states what was actually read and how confident that reading is; secondary-source reliance is
flagged explicitly, not glossed over.

Primary artifacts consulted directly (not from memory):
- Reference KalmanNet implementation (`KalmanNet_TSP`, github.com/KalmanNet/KalmanNet_TSP):
  `KNet/KalmanNet_nn.py`, `Pipelines/Pipeline_EKF.py`, `Simulations/config.py`,
  `Simulations/Extended_sysmdl.py`, `README.md`.
- Full PDFs read directly (via arXiv / NSF public-access / institutional open mirrors, cited per
  section below): KalmanNet (arXiv:2107.10043), Recursive KalmanNet (arXiv:2506.11639),
  Cholesky-KalmanNet (via `par.nsf.gov/servlets/purl/10656999`), Bayesian KalmanNet
  (arXiv:2309.03058), AI-Augmented Designs for Kalman-Type Algorithms (arXiv:2410.12289, also
  hosted at the Weizmann Institute's site for the paper).

---

## 1. KalmanNet — Revach, Shlezinger, Ni, Escoriza, van Sloun, Eldar, IEEE TSP 70 (2022) 1532–1547, arXiv:2107.10043

Foundational paper: a hybrid model-based/data-driven recursive filter that keeps the EKF's
predict/update flow and known (possibly approximate) f(·), h(·), but replaces only the
model-based Kalman-gain computation with a GRU-based RNN. This is our baseline learned-filter
comparator.

**(a) Architecture #2 (GRU-based Kalman gain network)** — grounded in both the paper (Sec. III,
Figs. 3–4, eqs. 6–14) and the reference implementation (`KNet/KalmanNet_nn.py`); the repo's
`README.md` states explicitly: "This branch simulates architecture #2 in our paper."

- The paper presents two architectures; **architecture #2** is the one actually shipped/trained in
  the maintained reference repo, and is lower-parameter than architecture #1 (see below).
- Inputs: four causal feature *differences*, all computable online from the filter's own running
  state (no ground truth, no lookahead):
  - **F1** observation difference, Δỹ_t = y_t − y_{t−1}
  - **F2** innovation difference, Δy_t = y_t − ŷ_{t|t−1}
  - **F3** forward evolution difference — a lagged difference between consecutive posterior state
    estimates (feeds the "evolution" branch; exact index bookkeeping is one-step-lagged so nothing
    from the future leaks in — see code for precise indices, `fw_evol_diff` in `KalmanNet_nn.py`)
  - **F4** forward update difference — a lagged difference between a prior and its corresponding
    posterior (feeds the "update" branch; `fw_update_diff` in code)
  - All four are L2-normalized before entering the network (`func.normalize(..., p=2, ...)` in code).
- Internal structure (three GRUs, each with dedicated FC pre/post layers; code names in
  parentheses):
  - **Q-GRU** (`GRU_Q`): input = FC5(F4); hidden dim = **m²** — tracks the (unknown) process-noise
    covariance Q.
  - **Sigma-GRU** (`GRU_Sigma`): input = concat(Q-GRU output, FC6(F3)); hidden dim = **m²** — tracks
    the predicted state covariance Σ_{t|t−1}.
  - **S-GRU** (`GRU_S`): input = concat(FC1(Sigma-GRU output), FC7(concat(F1,F2))); hidden dim =
    **n²** — tracks the innovation covariance S_{t|t−1}.
  - Output: FC2 maps concat(Sigma-GRU output, S-GRU output) → a flat vector of size n·m, reshaped
    to the full **m×n Kalman gain matrix** (not a further-factorized representation — a straight
    reshape).
  - A "backward flow" (FC3, FC4) feeds a function of the just-computed gain back to directly
    overwrite the Sigma-GRU's hidden state (`self.h_Sigma = out_FC4`) each step — a non-standard
    interconnection the paper calls out explicitly, distinct from a plain stacked-GRU RNN.
  - Hyperparameters `in_mult_KNet` (default 5, widens FC5/6/7 outputs) and `out_mult_KNet` (default
    40, widens FC2's hidden layer) control capacity (`Simulations/config.py`).
  - Hidden states are *initialized* from user-supplied `prior_Q`/`prior_Sigma`/`prior_S` matrices
    (flattened) — a one-time prior, not a per-step input; the network is free to diverge from it.
  - Contrast: architecture #1 uses a single GRU of hidden size 10·(m²+n²) directly outputting the
    gain. Per the paper's Lorenz-attractor case study (Sec. IV-D): architecture #1 ≈ 5×10⁵
    trainable parameters vs. architecture #2 ≈ 2.5×10⁴ — architecture #2 is ~20× smaller.

**(b) Training loss** (paper eqs. 10, 13, 14; matches `Pipelines/Pipeline_EKF.py`):
- Per-step: L = ‖x_t − x̂_{t|t}‖² (squared error), differentiated through the learned gain (eq. 11
  gives the explicit gradient w.r.t. K_t).
- Per-trajectory: **time-averaged** (not summed) over the T_i steps of a trajectory, plus an ℓ2
  weight-decay term γ‖Θ‖² on all trainable parameters (eq. 13).
- Per-minibatch: further averaged over the M trajectories in the batch (eq. 14).
- Trained via BPTT; the paper evaluates 3 BPTT variants (full BPTT; truncated-then-full; truncated
  only) for stability on long trajectories.
- Reference code implements this exactly as `nn.MSELoss(reduction='mean')` plus Adam's
  `weight_decay` (realizing the γ‖Θ‖² term). The repo also supports an optional, non-default
  **"composition loss"**, `α·MSE(x̂,x) + (1−α)·MSE(h(x̂),y)` (`config.py`'s `CompositionLoss` flag,
  default `False`) — this is a repo-level training option, not the paper's primary/default loss.

**(c) Partial model knowledge** — this determines how our comparison must be structured:
- The paper is explicit: f(·) and h(·) are **known or approximated** (possibly mismatched vs. the
  true generative model), while the noise covariances **Q, R are unavailable** — the inverse of
  classical KF assumptions.
- Two evaluated regimes: "full information" (f,h exactly match the generator) vs. "partial
  information" (f or h deliberately mismatched, e.g. a rotated state-transition matrix F_α with
  α ∈ {10°, 20°}, or a ~5%-misaligned observation matrix H_α) — used to show KalmanNet degrades
  gracefully where the EKF (whose gain is analytically tied to the wrong linearization) does not.
- Code confirms this precisely: `InitSystemDynamics(f, h, m, n)` takes literal Python closures for
  f and h, used directly in `step_prior()`. The *only* learned component is the gain-computing GRU
  stack; Q and R never appear as network inputs (only as one-time hidden-state priors, see above).
- **Design implication**: for our az/el/range tracker, fairness vs. EKF/UKF/IMM requires handing
  KalmanNet (and all its variants below) the *identical* f() (per-mode CV/CT/CA motion model) and
  h() (nonlinear az/el/range projection) used by the classical filters — only the
  gain/covariance-equivalent computation should be learned.

**Status: read primary source** (arXiv PDF, full text of pages 1–10 — architecture, loss, and
partial-information sections) **and cross-checked against the reference implementation code**
(`KNet/KalmanNet_nn.py`, `Pipelines/Pipeline_EKF.py`, `Simulations/config.py`,
`Simulations/Extended_sysmdl.py`). Highest-confidence source in this set — dual-verified.

---

## 2. Recursive KalmanNet — Mortada, Falcon, Kahil, Clavaud, Michel, EUSIPCO 2025, arXiv:2506.11639

Our primary consistency-corrected variant. **Correction to task premise: official code exists** —
`github.com/ixblue/RecursiveKalmanNet` (confirmed reachable; contains `Algo/` with SS-model, KF,
RKN, and loss classes, plus a demo notebook `main_bimodal_noise.ipynb`). This is *not* a
from-scratch reimplementation situation; start from the released repo.

- **Consistency mechanism**: keeps a KalmanNet-style gain-estimating RNN (RNN₁, params Θ₁), and
  adds a **second RNN** (RNN₂, params Θ₂) that estimates only the Cholesky factor of one term of a
  Joseph-form covariance recursion. Exact decomposition (paper eqs. 6–8):
  - P_{t|t} = (I−K_tH_t)P_{t|t−1}(I−K_tH_t)ᵀ + K_tR_tK_tᵀ = **A_t + B_t**
  - **A_t** = (I−K_tH_t)F_tP_{t−1|t−1}F_tᵀ(I−K_tH_t)ᵀ — computed **in closed form**, using the
    just-estimated gain K_t and the *previous* covariance estimate P̂_{t−1|t−1} (this recursive
    dependence on the network's own prior output is what gives the method its name).
  - **B_t** = (I−K_tH_t)Q_t(I−K_tH_t)ᵀ + K_tR_tK_tᵀ = **C_tC_tᵀ**, with C_t (lower-triangular,
    real entries) directly output by RNN₂ — the only quantity that is learned via a Cholesky
    parametrization.
  - Notably incorporates F_t (the state-transition matrix) explicitly into A_t — the paper calls
    this out as an improvement over Split-KalmanNet and Cholesky-KalmanNet, neither of which uses
    F_t in their covariance computation.
- **Architecture**: two separate RNNs (both GRU + FC pre/post layers), same 4 input features per
  step: F1 innovation ŷ_t; F2 previous state correction K̂_{t−1}ŷ_{t−1}; F3 Jacobian H_t of the
  observation equation; F4 measurement temporal difference z_t − z_{t−1} (F1/F2/F4 borrowed from
  KalmanNet; F4 also used by Split-KalmanNet). Squared (elementwise) features empirically improve
  performance, motivated by covariance being a second-order/quadratic quantity. Batch normalization
  was deliberately removed from these features — the paper found it suppresses the transient
  dynamics the network needs to track under time-varying state-space models.
- **Loss**: Gaussian negative log-likelihood, jointly over Θ₁ and Θ₂ (not alternating):
  `L_t = e_tᵀ P̂_{t|t}⁻¹ e_t + log det P̂_{t|t}` (eq. 9), batch+time-averaged, with ℓ2 weight
  regularization — a **tuning-free** balance between accuracy and calibration. At critical points,
  the gradient conditions (eqs. 10a/10b) reduce to P̂_{t|t} = e_te_tᵀ and e_tŷ_tᵀ = 0 (state-error/
  innovation orthogonality) — the loss's stationary points are exactly the calibration conditions
  for an optimal filter.
- Evaluated with **Mean Squared Mahalanobis Distance (MSMD)** = time/sample-averaged
  e_tᵀP_{t|t}⁻¹e_t, whose expectation should equal the state dimension m if the covariance is
  well-calibrated (a chi-squared/NEES-style consistency check) — directly usable for our benchmark.
- The paper **directly critiques Cholesky-KalmanNet's calibration**: CKN's own released
  implementation weights its hybrid loss at β=0.05 (covariance term) vs. 0.95 (MSE term),
  empirically producing MSMD values far from the theoretical mean m, i.e. poorly calibrated
  covariance, whereas RKN's tuning-free NLL loss tracks the theoretical MSMD closely.

**Status: read primary source in full** (arXiv PDF, complete 6-page paper — abstract, architecture,
feature list, all equations, and experiments).

---

## 3. Cholesky-KalmanNet — Ko & Shafieezadeh, IEEE SPL 32 (2025) 326–330

**Correction to task premise: this source was not paywalled-unavailable.** The full paper (not
just the abstract) is openly hosted as an author-accepted manuscript via the NSF Public Access
Repository: `https://par.nsf.gov/servlets/purl/10656999`. Read in full.

- **Built on top of Split-KalmanNet (SKN)**, not directly on KalmanNet arch #1/#2. SKN uses two
  DNNs: G¹_t(Θ₁) learns the prior state covariance Σ_{t|t−1} implicitly, G²_t(Θ₂) learns the
  inverse innovation covariance S⁻¹_{t|t−1} implicitly, combined as
  K_t = G¹_t(Θ₁)·H_tᵀ·G²_t(Θ₂). CKN adds a **Positive Definite Enforcing Layer (PDEL)** on top of
  this split-GRU architecture so these covariance-related outputs (and the derived posterior
  covariance Σ̂_{t|t}) are provably positive-definite.
- **PDEL mechanics** (exact, eq. 4): an RNN output vector **A ∈ ℝ^p** is reshaped into a
  lower-triangular matrix **L′ ∈ ℝ^{q×q}** where p = q(q+1)/2 (only the lower-triangular free
  parameters — half the entries of a full q×q matrix), with a strictly-positive function p(·)
  applied to the diagonal entries only. They empirically settled on **p(a) = |a| + ε** (over
  exp/square/ReLU alternatives — smoother training). Final covariance: **C = L′L′ᵀ**, positive
  definite by construction. This halves the DNN output size needed vs. plain SKN's full q²-entry
  estimate, reducing overparameterization and training-data requirements.
- **Loss** (exact, eqs. 5–7): `L_total = (1−β)·L_ℓ2 + β·L_cov`, where L_ℓ2 is the standard
  time/batch-averaged state MSE, and **L_cov is a mean-absolute-deviation (L1) term** comparing
  every entry (diagonal *and* off-diagonal) of the empirical outer-product e_te_tᵀ to the
  estimated Σ̂_{t|t}. β∈[0,1] is a hand-tuned hyperparameter — per Recursive KalmanNet's critique
  (source #2), the released implementation uses β=0.05, heavily favoring MSE over calibration.
- Evaluated against EKF/KN/SKN on: (i) synthetic linear-state/nonlinear-observation model
  (rotation state matrix + polar-like h(x)=[‖x‖, atan2(x)]) — structurally close to our own
  az/el/range problem; (ii) synthetic nonlinear sinusoidal-state/polynomial-observation model under
  3 tiers of increasing f/h mismatch; (iii) real-world Michigan NCLT dataset (nearly-constant-
  acceleration model tracking a Segway robot from noisy velocity/GPS/odometry) — CKN wins on MSE
  and on run-to-run stability (narrowest MSE variance across 10 retrainings) in all three.
- **Code**: `github.com/RAMSIS-Lab/ckn-spl-public` (stated in paper).

**Status: read primary source in full** (complete PDF via NSF public-access mirror of the IEEE SPL
accepted manuscript).

---

## 4. Bayesian KalmanNet — Dahan, Revach, Duník, Shlezinger, IEEE TSP 73 (2025) 2558–2573, arXiv:2309.03058

Framing/uncertainty-quantification source, lower priority than #1–#2 per task brief, but a full
primary read was obtained cheaply.

- Frames UQ as **Bayesian deep learning applied to KalmanNet's gain-estimating network**: instead
  of one deterministic weight set θ, learns/samples a distribution q(θ|φ) over weights, realized
  in their numerical study via **Monte Carlo dropout** — dropout layers (Bernoulli parameter p)
  inserted into the **fully-connected layers only** of a KalmanNet "Architecture #1" backbone
  (single GRU + FC in/out), sampled at *inference* time (not just training).
- Each of **J** i.i.d. dropout realizations {θ_j} produces its own gain K̂_t^(j) and state estimate
  x̂_t^(j); outputs are ensemble statistics (eqs. 22a/22b): state estimate = **sample mean** over J,
  error covariance Σ̂_t = **sample covariance** of the J estimates around their mean. This is a
  genuine MC-ensemble second-moment estimate, not a closed-form propagated covariance like
  RKN/CKN.
- The paper frames its own method as the 4th of 4 approaches for extracting covariance from
  DNN-aided KFs, the other 3 being: (i) black-box DNN + explicit covariance output neuron
  (needs a dedicated loss, no ground-truth covariance to supervise against); (ii) "KG + prior
  covariance" architectures (Split-KalmanNet-style, prior covariance as an addressable internal
  feature); (iii) "KG-only" architectures (plain KalmanNet) where prior covariance must be
  *recovered* post-hoc via a pseudo-inverse trick requiring H̃_t=(H_tH_tᵀ)⁻¹ to exist, i.e. H_t
  full column rank (same limitation independently flagged in Cholesky-KalmanNet's related-work
  section — both papers cite the same ICASSP'22 result, Klein et al.). Their own Bayesian
  KalmanNet is motivated specifically to avoid that full-column-rank requirement.
- Two loss variants proposed for making covariance meaningful under existing architectures: an
  empirical second-moment loss L^M2 (compares predicted diagonal covariance entries to
  instantaneous squared error) and a Gaussian-log-likelihood loss L^GP — but their actual "Bayesian
  KalmanNet" method is trained with **plain ℓ2 (state MSE)**; calibration comes purely from
  MC-dropout ensembling at inference, not from a covariance-aware loss term.
- **Naming-collision note**: this paper cites "RKN" (recurrent Kalman network) as a *different,
  older, unrelated* method (Becker et al. 2019) that directly outputs a diagonal error-variance
  vector — not to be confused with "Recursive KalmanNet" (source #2 above, Mortada et al. 2025).
  Same acronym, unrelated papers — see also source #6 below where the same collision recurs.

**Status: read primary source** (arXiv PDF, full text of pages 1–6 — system model, architecture,
algorithm box, and all equations through the training-loss discussion).

---

## 5. Ensemble of KalmanNets — Mari & Snidaro (Information Fusion 127 Pt. A (2026) 103777; and FUSION 2024 / Information Fusion 103224 (2025))

The learned analogue of IMM we may implement later — one KalmanNet per motion model, fused.
**Primary text could not be obtained**; every detail below is second-hand and should be treated as
unverified until we can get institutional/library access.

- One KalmanNet instance is instantiated per candidate motion model (CV/CT/CA-equivalent) — the
  learned analogue of IMM's mode-conditioned filter bank.
- Per repeated search-engine-synthesized summaries (consistent across three independent queries,
  suggesting they reflect indexed abstract/snippet text rather than one-off hallucination): a
  **two-branch fusion network** — a GRU-based branch consumes the per-model KalmanNet outputs and
  produces a *correction term* for each (compensating for that model's mismatch); a separate
  **LSTM-based branch** computes the most-likely target mode, explicitly described as **replacing
  the likelihood-based mode-probability update of classical IMM** with a learned RNN estimate.
- "Innovation-based attention" (per the 2026 paper's title) is apparently the mechanism used
  somewhere in this fusion, but the exact query/key/value construction could not be confirmed from
  any text I actually read.
- Reported (again, second-hand) to outperform both classical IMM and a plain single-motion-model
  (CV) KalmanNet on simulated air-traffic-control (ATC) domain trajectories.

**Status: secondary description, primary unread.** Attempted and failed to fetch: the OpenReview
PDF mirror (`openreview.net/pdf/23f672e79b768e66a036b96067822b531ce1a153.pdf`, returned a bot-check
page, no content), the ScienceDirect abstract page (`sciencedirect.com/.../S1566253525008395`,
403/paywalled), and a related Udine institutional-repository survey PDF that might have cited the
authors' own description (`air.uniud.it/retrieve/.../S1566253525008516-main.pdf`, 403 Forbidden).
IEEE Xplore hosts the FUSION-2024 conference version (`ieeexplore.org/document/10706253`), also
paywalled. Do not cite any architectural claim from this section as fact in downstream design docs
without the "secondary, unverified" caveat.

---

## 6. AI-Augmented Designs for Kalman-Type Algorithms — Shlezinger, Revach, Ghosh, Chatterjee, Tang, Imbiriba, Duník, Straka, Closas, Eldar, IEEE SPM 42(3) (2025) 52–76, arXiv:2410.12289

Tutorial-style survey/taxonomy, read for framing only, to sanity-check how the field categorizes
these methods.

- High-level taxonomy: **task-oriented** (discriminative — DNN trained directly to output the
  state estimate, e.g. KalmanNet and all its gain-replacement variants) vs. **SS-model-oriented**
  (generative — DNN learns a missing piece of the state-space model itself, e.g. system
  identification of f/h from data, then plugged back into a classical filter). The entire
  KalmanNet family (sources #1–#5 above) sits in the task-oriented/discriminative camp.
- Within generic (non-SS-aware) DNN building blocks for time series, the survey reviews RNNs,
  self-attention/transformers, and 1D-CNNs, then turns to SS/KF-*inspired* architectures:
  "SSM-inspired" (linear latent state-space parameterizations, related to selective-state-space-
  model / S4-style work) and "KF-inspired" — the latter's worked example is the **Recurrent Kalman
  Network (RKN, Becker et al. 2019)**, which imitates the KF's predict/update *structure* but
  replaces both stages with generic learned DNNs, unlike KalmanNet's approach of keeping the
  *analytic* predict/update equations and replacing only the gain.
- **Naming-collision warning (repeated from source #4)**: this survey's "RKN" = Recurrent Kalman
  Network (Becker et al.) is unrelated to "Recursive KalmanNet" (source #2, Mortada et al. 2025).
  Same acronym, different papers/authors/mechanisms.
- Provides a checklist of classical KF desirable properties (optimal-for-linear-Gaussian, adaptive
  to known model variations, interpretable, provides uncertainty, low complexity) vs. challenges
  (SS model typically approximated, noise model elusive, suboptimal for nonlinear models, latency
  for nonlinear filters) motivating the whole research program — a useful checklist for what our
  own benchmark should report, beyond plain RMSE.

**Status: read primary source** (arXiv PDF, pages 1–8 of 25 — introduction, KF fundamentals, and
the start of the "Combining AI with KFs" taxonomy section). Did **not** read the remainder (full
task-oriented/SS-model-oriented design catalog, or the paper's own quantitative Lorenz-attractor
comparison study) — treat anything beyond the taxonomy summary above as unread.

---

## 7. Classical IMM — Blom & Bar-Shalom (1988); Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and Navigation*

Standard classical baseline for maneuvering-target tracking with multiple motion-model hypotheses
(e.g., CV/CT/CA) — what every learned filter above must be benchmarked against.

Standard algorithm structure:
1. **Mode-conditioned filters**: r parallel Kalman-type filters (one per motion model/mode
   j=1..r — e.g., EKF/UKF instances for CV, CT, CA), each maintaining its own state/covariance.
2. **Interaction/mixing step**: at the start of each cycle, the r filters' previous-cycle outputs
   are mixed using a Markov mode-transition probability matrix Π (entries π_ij = P(mode j at k |
   mode i at k−1)) and the previous mode probabilities, producing r *mixed initial conditions*
   (weighted combinations of all filters' prior states/covariances, weighted by mixing
   probabilities μ_{i|j}).
3. **Mode-conditioned filtering**: each of the r filters runs one predict/update cycle from its own
   mixed initial condition and the current measurement, producing an updated state/covariance and,
   crucially, a measurement likelihood Λ_j (typically the Gaussian likelihood of the innovation
   under that mode's predicted innovation covariance).
4. **Mode probability update**: Bayesian update of the mode probabilities μ_j using the
   likelihoods Λ_j and the predicted mode probabilities from step 2, renormalized to sum to 1.
5. **Output combination**: the overall state estimate and covariance are the probability-weighted
   combination of all r mode-conditioned estimates/covariances (weighted by the updated mode
   probabilities from step 4), which also feeds the next cycle's step 2.

**Status: textbook knowledge, not fetched from primary source** (pre-print era; Blom & Bar-Shalom
1988, IEEE Trans. Automatic Control, and the Bar-Shalom/Li/Kirubarajan textbook are not freely
available online and were not fetched). This is a standard, uncontroversial description reproduced
from well-established textbook/survey knowledge, not verified against the 1988 paper's exact
notation.

---

## Design implications

- **Fairness constraint (all KalmanNet variants)**: give KalmanNet/RKN/CKN/BKN the *exact same*
  f()/h() as the classical EKF/UKF/IMM baselines — our az/el/range nonlinear h() and per-mode
  CV/CT/CA f(). Per source #1(c), only the gain (which stands in for Q/R-dependent computations)
  should be learned; anything else invalidates the comparison.
- **Baseline KalmanNet = architecture #2**, matching the reference repo exactly: three GRUs
  (Q-GRU on F4, Sigma-GRU on F3+Q-GRU-output, S-GRU on F1+F2, hidden dims m², m², n²
  respectively), trained with per-step squared-error state loss (time-averaged per trajectory +
  ℓ2 weight decay), not the optional composition loss — this matches the paper's reported numbers
  and is what we should reproduce first before adding complexity.
- **Recursive KalmanNet is the first calibration-corrected variant to build, and it is a
  from-repo integration, not a from-paper reimplementation**: official code exists at
  `github.com/ixblue/RecursiveKalmanNet`. Start there; adapt its two-RNN design (gain RNN + a
  second RNN outputting only the Cholesky factor of the Joseph-form noise term B_t, with the
  propagated term A_t computed in closed form from F_t and the previous covariance) and its
  tuning-free Gaussian-NLL loss (eq. 9) to our az/el/range Jacobian H_t. Evaluate calibration with
  their own MSMD metric (should converge to the state dimension m if well-calibrated) — this
  gives us an NEES-equivalent metric "for free" to compare directly against IMM's covariance
  output.
- **Cholesky-KalmanNet is likewise a from-repo integration** (`github.com/RAMSIS-Lab/ckn-spl-public`),
  useful as a second covariance-producing baseline, but budget for its known calibration weakness:
  its released β=0.05 hybrid loss favors MSE over covariance accuracy (per RKN's own critique) — if
  we reproduce it, either retune β toward covariance accuracy or report MSE and MSMD/NEES side by
  side so the tradeoff is visible rather than hidden behind a single headline number.
- **Ensemble-of-KalmanNets (source #5) is our best learned analogue of IMM but is unverified** —
  do not hard-code its "GRU-correction branch + LSTM-mode-probability branch" design into any
  downstream plan/report as established fact; treat it as a research direction to revisit once we
  can get the actual paper (institutional access, or emailing the Udine authors), and prioritize
  implementing the verified RKN/CKN/IMM comparison first.
- **Benchmark metrics beyond RMSE**: per the AI-Augmented survey's taxonomy (source #6) and RKN's
  MSMD metric (source #2), our benchmark should report calibration/consistency (an NEES/MSMD-style
  statistic, expected to equal the state dimension for a well-calibrated filter) as a first-class
  metric alongside tracking-accuracy RMSE — this is precisely the axis distinguishing plain
  KalmanNet (no covariance output at all) from RKN/CKN/BKN (all covariance-producing) and from
  classical IMM (which produces covariance "for free" as part of its Kalman-filter machinery).
