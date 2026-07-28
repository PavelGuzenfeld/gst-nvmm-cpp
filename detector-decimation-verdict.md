# Detector decimation — CLOSED, not worth landing

**Final outcome (measurements 8–9). Two operating points, neither a good trade:**

| | fps gain | GT success | tracks lost |
|---|---|---|---|
| N=3 ungated | +57.8% | **−22.9%** | 3 of 12 |
| N=3 acquisition-gated | **+14.0%** | −4.9% | 0 |

Gating fixes the quality but collapses the throughput, because the gate and the
interval are coupled: with a detection on only ~84% of frames, `gate=5` keeps the
gate mostly closed and the interval rarely reaches a skip (N=2 and N=3 measure
*identically*). There is no obvious setting in between.

`nvmminfer` gained `infer-interval` and `infer-gate-frames`; **both stay at their
defaults (1 and 0 = off)** and are kept as instruments, not features.

> **This doc is a chronological log and earlier sections are superseded. Read
> measurements 7, 8 and 9 for what is true.** Specifically: (a) measurement 2's
> +57.8% is *ungated* throughput and must not be paired with gated quality — an
> error made and corrected in measurement 9; (b) measurement 3's clip2/clip3 rows are
> invalid (seeded on detector false positives — see measurement 4); (c) measurements
> 3–4 blamed `nvmmfusekf` flush-BB toggling by misreading the `yolo_fused` CSV column
> as a flush-BB indicator when it is the KF-fusion flag — see measurement 5;
> (d) measurement 6's −7.8% came from a degraded pipeline and understated the cost
> ~3× — see measurement 7.
>
> The recurring cause of every one of those: **measuring on a proxy configuration
> and only later checking the deployed one.** Simplified pipeline, hand-picked seeds,
> degraded operating point. Each proxy gave a confident answer the real
> configuration contradicted.

This effort came out of a design interview about decomposing the SAMURAI ONNX
graphs and placing components per accelerator. The decomposition itself is
specified in [samurai-decomposition-plan.md](samurai-decomposition-plan.md) and
was **not started** — this lever surfaced mid-interview with apparently better
expected value and consumed the effort.

## Why it looked promising

`nvmminfer` ran YOLO unconditionally on every frame, while `nvmmsamurai max-kf=2`
does full inference only every 3rd frame and coasts on the Kalman filter between.
The detector had no interval property, so that asymmetry was unexploited.

The trap in estimating it: dividing a ~21 ms detector into an ~89 ms frame gives
~24%, which is what we first projected (~16% at N=3). The *deployed* frame at
`max-kf=2` is only ~49.6 ms, because the tracker coasts 2 of every 3 frames while
the detector runs on all 3. **The detector is ~50% of the deployed frame budget.**

## Measurement 1 — bound the headroom (removal A/B)

Arm A = deployed shape; arm B = `detector` removed. Both arms force the tracker
seed via `seed-roi`, so tracker work is identical and the only variable is the
detector. Without the forced seed, arm B never seeds, does no inference, and the
fps gain is an artifact rather than a measurement.

Orin NX 16 GB, MAXN, `clip1.mp4` 1442 frames, 2 iterations, spread <0.3%.

| max-kf | A: yolo present | B: yolo absent | gain | detector cost |
|---|---|---|---|---|
| **2 (deployed)** | **20.18 fps** (49.6 ms/f) | **40.31 fps** (24.8 ms/f) | **+99.8%** | 24.8 ms/f |
| 0 (full every frame) | 11.64 fps (85.9 ms/f) | 15.11 fps (66.2 ms/f) | +29.8% | 19.7 ms/f |

Cross-validation: arm B @`max-kf=0` = 66.2 ms/frame against the DLA pathfinder's
independently instrumented 67.8 ms frame total — two unrelated methods, 2.4%
apart.

Note the detector's marginal cost is *lower* at `max-kf=0` (19.7 ms) than at
`max-kf=2` (24.8 ms), implying ~5 ms of detector work does overlap tracker work
when the tracker is busy every frame. So SM headroom on this board is small but
not zero.

Repro: `tools/samurai/detector-decimation/ab_detector_cost.sh`.

## Measurement 2 — the implemented property (throughput)

`infer-interval` sweep, deployed `max-kf=2`, same clip and forced seed.

| infer-interval | fps | gain | projected |
|---|---|---|---|
| 1 (baseline) | 20.20 | — | — |
| 2 | 27.19 | +34.6% | +33% |
| **3** | **31.87** | **+57.8%** | +50% |
| 6 | 35.55 | +76.0% | +71% |
| 1000 | 40.17 | +98.9% | +100% |

Two validations:

- **Cross-build**: `interval=1` = 20.20 fps vs the removal A/B's detector-present
  arm 20.18 fps — 0.1% apart, despite different checkouts.
- **Functional**: `interval=1000` = 40.17 fps vs measured detector-absent
  40.31 fps — 0.35% apart, confirming frames are genuinely skipped rather than
  the fps moving for an unrelated reason.

`N=3` beat projection (+57.8% vs +50%), consistent with the overlap noted above:
decimation preferentially removes the *contended* frames. Mechanism inferred, the
number measured.

Repro: `tools/samurai/detector-decimation/interval_sweep.sh`.

## Measurement 3 — behavioural parity: FAILS on all three clips

New gate `tools/trajectory_compare.py` diffs two `nvmmfusekf` `$NVMMFUSEKF_CSV`
dumps frame by frame. It reads **fusekf's emitted box**, not `nvmmsamurai`'s,
because that is what downstream receives and where decimation's cost lands:
fusekf takes the best YOLO detection as a per-frame *gated measurement*, and its
flush-BB path publishes the tight detection box when the fused box is materially
larger.

Bar: per-frame IoU ≥ 0.99 median, no frame < 0.9, no valid-flag flips.

| clip | frames | N | median IoU | frames <0.9 | baseline yolo-fused rate |
|---|---|---|---|---|---|
| clip1 | 1442 | 2 | 0.9708 | 641 | **83.7%** |
| clip1 | 1442 | 3 | 0.4847 | 884 | 83.7% |
| clip2 | 2466 | 2 | 1.0000 | 239 | **6.2%** |
| clip2 | 2466 | 3 | 1.0000 | 305 | 6.2% |
| clip3 | 2264 | 2 | 1.0000 | 501 | **3.9%** |
| clip3 | 2264 | 3 | 1.0000 | 665 | 3.9% |

All six fail. **The failure mechanism inverts with detector density** — this is
the transferable finding:

| clip, N=2 | non-toggle frames <0.9 | of those, within 5 frames of a toggle | median centre distance | median area ratio |
|---|---|---|---|---|
| clip1 | 60 | **60/60** | 0.8 px | 1.17× |
| clip2 | 163 | **22/163** | **157.2 px** | **30.7×** |
| clip3 | 466 | **4/466** | **172.7 px** | **35.1×** |

- **clip1 (dense detections, 83.7%)**: no tracking damage at all. On the 848 frames
  where `yolo_fused` agrees between runs the emitted box is *bit-identical*
  (median IoU 1.0000). 100% of failures trace to flush-BB toggling — 581 direct,
  60 as knock-on fused-KF settling within 5 frames. The damage is cosmetic box
  flapping: at frame 667, N=1 emits 19.0×9.8 with `yolo_fused=1` and N=2 emits
  49.4×25.9 with `yolo_fused=0`; centres agree to ~1 px, area differs 6.9×.
- **clip2/clip3 (sparse detections, 4–6%)**: **genuine, persistent trajectory
  divergence.** Median centre error 157–173 px. clip3 has one divergence run
  spanning frames **[843, 1725]** — 883 consecutive frames, diverged and never
  reconverged.

**Why clip1 alone would have misled us.** With the detector firing on 5 of 6 frames,
decimating still leaves abundant detections and the fused box stays
detector-dominated. Where the detector fires on 4–6% of frames, every detection is
load-bearing — reseed authority, drift correction — and dropping any lets the
track walk away for good.

**Consequence for the obvious fix.** Making flush-BB detection-independent (carry
the last tight-box scale instead of reverting to the diffuse box) would fix clip1's
failure mode and do **nothing** for clip2/clip3's. There is no cheap repair here.

Repro: `tools/samurai/detector-decimation/quality_ab_clip.sh`,
attribution via `explain_nontoggle.py`.

## Measurement 4 — RE-RUN WITH VERIFIED SEEDS. Supersedes the clip2/clip3 rows above.

The clip2/clip3 seeds used in measurement 3 were taken from the first box the detector
emitted. **Both were false positives.** Visual inspection of the seeded regions
(frames extracted with `videocrop`/`videoscale`/`pngenc`) shows clip2's
`852,257` is hazy sky/ridgeline and clip3's `1410,822` is terrain texture — no
target in either. The tracker was tracking noise, so `nvmmfusekf` distance-gated
out the *real* detections: clip3 has conf ≥0.80 detections on ~74% of frames yet
only **3.9%** were fused. That gap is the tell.

Re-run with visually verified targets, `seed-delay` pinning both arms to the same
seed frame:

| clip | seed | frame | conf | target |
|---|---|---|---|---|
| clip3 | `772,537 18x10` | 117 | 0.83 | bright object in dark sky above ridge (thermal/IR) |
| clip2 | `1126,373 23x10` | 1167 | 0.76 | dark aircraft silhouette against pale terrain |

Baseline fused rate rises to **93.5%** (clip3) and **51.7%** (clip2), confirming the
seeds are on target.

| clip | N | median IoU | frames <0.9 | comparable |
|---|---|---|---|---|
| clip3 | 2 | 0.9206 | 901 | 2147 |
| clip3 | 3 | 0.8661 | 1345 | 2147 |
| clip2 | 2 | 0.9168 | 600 | 1299 |
| clip2 | 3 | 0.8461 | 827 | 1299 |

Still fails the 0.99 bar. But the attribution now matches clip1 on all three clips:

| clip, N=2 | <0.9 total | toggle frames | non-toggle | non-toggle near a toggle | median centre dist | median area ratio |
|---|---|---|---|---|---|---|
| clip3 | 901 | 862 | **39** | **39/39** | **0.4 px** (max 1.5) | 0.90 |
| clip2 | 600 | 584 | **16** | **16/16** | **1.4 px** (max 4.8) | 0.96 |

**The "genuine persistent divergence" reported in measurement 3 does not exist.**
It was an artifact of tracking noise (a track on featureless haze has no stable
appearance to re-find, so any perturbation sends it anywhere), compounded by an
attribution bug that scored the 1167 pre-seed invalid frames as IoU 0.0. Both are
fixed: `explain_nontoggle.py` now excludes frames invalid in both runs.

**Corrected conclusion.** Across all three clips, decimation does **not** perturb
the tracker — centres agree to 0.4–1.4 px median. ~100% of the quality failure is
`nvmmfusekf` flush-BB alternating between the tight detection box and the diffuse
fused box on frames without a detection. So **fixing flush-BB's detector gating is
the right fix and applies to every clip**, reversing measurement 3's claim that it
would help only clip1.

Residual uncertainty: even on non-toggle frames the area ratio is 0.90 (clip3) /
0.96 (clip2), i.e. the fused KF's *size* estimate carries some detection history.
Whether removing the toggle alone clears a 0.99 median at ~200 px² targets is
unmeasured — plausible given the centre agreement, not established.

## Measurement 5 — the flush-BB fix FAILS, and reveals the attribution was wrong

Implemented `nvmmfusekf flush-carry` (carry the last flushed box SIZE at the
current fused centre for N frames when a frame has no usable det), measured, and
**reverted**. Results vs the `(interval=1, carry=0)` baseline:

| test | result |
|---|---|
| clip3 `interval=1, carry=2` (neutrality) | PASS — median/mean/p05/**min** all 1.0000 |
| clip3 `interval=3`, carry=0 vs 2 vs 5 | **byte-identical CSVs** — carry never fired |
| clip1 `interval=1, carry=2` (neutrality) | **FAIL** — 175 frames <0.9, min 0.0774 |
| clip1 `interval=3` + carry=2 | median 0.4847→0.5789, but frames <0.9 rose 884→933 |

A no-op on the clip it was designed from, harmful on the clip where it does fire.

**Why it was a no-op on clip3**: flush-BB requires
`fused_area > flush_ratio(1.5) * det_area`. clip3's emitted box is ~255 px² against
det boxes of ~180 px² — a ratio near 1.2, so **flush-BB never triggers on clip3 at
all**, and the carry (which only arms after a real flush) never arms.

**The attribution error.** The `yolo_fused` column in the `NVMMFUSEKF_CSV` dump is
set at `gstnvmmfusekf.cpp:140` from `fused_yolo`, which flags whether the YOLO det
passed the **Kalman gate and updated the KF**. The flush-BB block is separate and
never writes that column. Measurements 3 and 4 read `yolo_fused` as a flush-BB
indicator and concluded "~100% of failures are flush-BB toggling". **That was a
misreading of the instrument.**

**The actual mechanism.** Decimation removes YOLO as the *second measurement to the
master KF* on skipped frames, so the fused box is computed from SAMURAI alone. The
box differs because **information is genuinely absent**, on every skipped frame and
on every clip. flush-BB toggling is a real but *secondary* effect, present only
where the mask is diffuse enough to trigger it (clip1 yes, clip3 no).

**Consequence: no publishing-layer fix can recover parity**, because there is
nothing to publish. And it follows that **IoU-parity-vs-baseline is the wrong bar
for decimation.** A tracker given fewer measurements produces a *different* track;
parity asks it to produce an *identical* one, which is unachievable by construction
for any change that removes input. The measured 0.85–0.92 medians say "different",
not "worse".

**The right next instrument is ground truth, not parity.** The box has labelled
sequences (`$ASSET_DIR/$EVALSET_ZIP`) and existing scoring
(`$DEPLOY_HARNESS`, `tools/score_pr.py`). Scoring N=1 vs N=3 against GT answers
whether the different track is *worse* — the question that actually decides whether
+57.8% is free. Until that is run, decimation is neither cleared nor killed.

Repro: `tools/samurai/detector-decimation/flush_carry_ab.sh` (the flush-carry
element change itself is reverted, so the script's `flush-carry=` args are inert
against current HEAD — kept as the experiment record).

## Measurement 6 — GROUND TRUTH. The cost is ~8%, and it is almost all acquisition.

the labelled evaluation set GT scoring (`$SCORER`, `$EVALSET_ZIP` labels), 5 sequences × 1500
frames, auto-seed (`seed-prefer-center=true`) so acquisition behaviour is exercised.

| metric | N=1 | N=3 | change |
|---|---|---|---|
| success (IoU>0.5) | 0.395 | 0.364 | **−7.8% rel** |
| state_acc (official) | 0.299 | 0.276 | −7.7% rel |
| mean_iou | 0.302 | 0.278 | −7.9% rel |
| miss | 0.019 | 0.044 | 2.3× worse |
| **seed_latency_f** | 22.4 | **59.4** | **2.65× worse** |
| yolo_rate | 0.254 | 0.083 | 0.33× (as designed) |

**The loss tracks acquisition delay, not steady-state tracking:**

| seq | seed_latency 1→3 | success 1→3 |
|---|---|---|
| seq-B | 0 → 0 | 0.987 → 0.973 (**−1.4%**) |
| seq-C | 0 → 0 | 0.325 → 0.327 (**no change**) |
| seq-G | **35 → 219** | 0.627 → 0.506 (**−19%**) |
| seq-K | 48 → 48 | 0.015 → 0.005 (both ~0) |
| seq-L | 29 → 30 | 0.022 → 0.007 (both ~0) |

Where acquisition is not delayed the cost is 0–1.4%. Where it is, 19%. A detector
running 1/3 as often takes ~3× longer to produce the first usable detection.

Per-attribute: TS (Tiny-Size, 3993 frames) 0.710→0.660, DBC (2392) 0.582→0.507.
FM (Fast-Motion) is 0.004 at N=1 — already broken, so this run cannot say whether
decimation hurts there.

**Indicated design: decimate only after the track is acquired.** Run the detector
every frame until first lock, then apply the interval. Acquisition is a small
fraction of a long sequence, so this should retain most of +57.8% while removing the
mechanism behind nearly all the measured loss. Note this is adaptive on *track
state*, not on detection density (the earlier idea, dropped for the wrong reason).

**Caveats that bound how far this reads:**

1. **Degraded operating point, not deployment.** `fp_absent_rate = 1.000` on 4 of 5
   sequences — a box is always reported even when the target is absent. That is the
   missing `nvmmdetgate` + teardown. Absolute success 0.395 is far below the
   deployed scorecard, so −7.8% is measured off-nominal and may differ at the real
   operating point.
2. **2 of 5 sequences carry no signal** (`seq_020`, `seq_034` score
   0.005–0.022 in *both* arms). The comparison effectively rests on 3 sequences and
   the headline seed-latency effect on 1.
3. Only 5 of the 12 `$SEQ_LIST` sequences were run (disk was at 94%).

Repro: `tools/samurai/detector-decimation/gt_score_ab.sh`. Two divergences between
`repo-checkout` and the deployed `deploy-checkout` harness each cost a debug
cycle: `nvmmdetgate` (element absent here) and `kf-vel-noise` (property absent
here) — gst-launch hard-rejects an unknown property, silently yielding empty CSVs.

## Open question that gates a final verdict (SUPERSEDED — see measurements 4, 5, 6)

Under the forced seeds used here, the **clip2/clip3 baselines themselves look
unhealthy**: clip2's median baseline box is 52947 px² (~230×230) from a 13×7 seed,
and clip3 emits degenerate boxes such as `(1653.4, 824.7, 1.7, 14.1)`. Both seeds
came from the first detector box, which on these clips was a low-confidence
`person` hit (0.44–0.65) and may be a false positive.

So measurement 3 is sound as *"decimation changes the output"* but not yet as
*"decimation degrades a healthy track"*. Resolving it needs verified targets —
hand-picked ROIs, or auto-seed pinned to a fixed frame so both arms seed
identically without a hand-chosen box. Until then `infer-interval` stays at its
default of 1.

## What is kept

- `nvmminfer infer-interval` — default 1, zero behaviour change unless set. Kept
  as the instrument needed to re-run the above, not as a shipped feature.
- `tools/trajectory_compare.py` — reusable behavioural gate for any
  math-changing change (decimation, speculative crop, precision).
- `tools/samurai/detector-decimation/` — every script above, for repro.

## Environment traps hit (both cost real time)

- **The dev Jetson's clock reads 1970** (RTC unset). Any file `scp`'d in gets a
  1970 mtime — older than existing build objects — so **ninja silently skips
  recompiling it** and links a stale `.so` while reporting success. Verify with
  `strings <so> | grep -c <new-symbol>` after every cross-copy.
- `repo-checkout`'s builddir was configured host-side and bakes host paths
  (`/usr/lib/aarch64-linux-gnu/tegra/...`) that do not exist in
  `gst-nvmm-infer:jp6` (which puts tegra libs under `.../nvidia/`). Build it on
  the host with `/usr/local/cuda/bin` on PATH. `$DEPLOY_SRC` is a
  **divergent** checkout — do not cross-copy between them.

## State left on the dev Jetson (2026-07-28) — one step staged, NOT run

The deployed-operating-point re-run (measurement 6's main caveat) is **staged but
not executed**:

- `deploy-checkout`'s `nvmminfer` is **patched with `infer-interval` and built**
  (`builddir-fix`), so the deployed pipeline shape — with `nvmmdetgate`,
  `kf-vel-noise`, and `detector_ir` on DLA core 0 — can now be A/B'd. That tree is
  **not a git repo**; a backup sits at
  `gst/nvmminfer/gstnvmminfer.cpp.pre-interval.bak`. Patch script:
  `tools/samurai/detector-decimation/port_interval_to_deploy.py` (idempotent).
- **The all-12-sequence GT run was never launched.** `gt_score_ab.sh` needs its
  pipeline switched to the deploy build + deployed element list, and should extract →
  run → delete per sequence rather than all at once (disk is at 94%).

Only the GPU `detector` should be decimated; `detector_ir` runs on DLA and costs no
GPU time, so leave it at interval 1.

### Disk on the dev board

Container images dominate the disk (~86% of used space); build-cache pruning
reclaimed effectively nothing because the cache is still referenced. Reclaiming space
means deleting whole image groups, which is an owner decision and was not done here.
Do not remove the build/inference image the harnesses depend on.

## Measurement 7 — DEPLOYED operating point, all 12 sequences. SUPERSEDES measurement 6.

Same GT scoring, but the deployed pipeline (`deploy-checkout` + `nvmmdetgate` +
`kf-vel-noise` + fusekf teardown), all 12 `$SEQ_LIST` sequences.
**The cost is ~3x what the degraded run reported.**

| metric | N=1 | N=3 | change |
|---|---|---|---|
| success | 0.490 | 0.378 | **−22.9% rel** |
| state_acc | 0.476 | 0.393 | −17.4% rel |
| mean_iou | 0.388 | 0.295 | −24.0% |
| miss | 0.281 | **0.461** | +64% |
| yolo_rate | 0.401 | 0.142 | 0.35x |

**Three of twelve sequences fail completely at N=3 — they never acquire:**

| seq | success 1→3 | miss 1→3 | seeded at N=3 |
|---|---|---|---|
| seq-G | 0.541 → **0.000** | 0.148 → **1.000** | never |
| seq-J | 0.004 → 0.000 | 0.976 → 1.000 | never |
| seq-L | 0.038 → 0.000 | 0.930 → 1.000 | never |
| seq-F | 0.213 → 0.014 | 0.015 → 0.691 | 108 |

**Do not read the seed_latency mean** (179.2 → 104.7, apparently better): the scorer
excludes never-seeded sequences, so losing three tracks entirely shortens the
average of the survivors. `miss` is the honest column.

**Why measurement 6 understated it.** That configuration barely worked —
`fp_absent_rate = 1.000` (a box always emitted) and success only 0.395, so there was
little to lose. At the deployed point the pipeline functions, so breaking acquisition
costs whole tracks.

**"Decimate after acquisition" survives only partially.** Sequences that still
acquire lose 1.4–13%:

| seq | seed 1→3 | success 1→3 | yolo_rate 1→3 |
|---|---|---|---|
| 01_2192 | 24→27 | 0.973→0.959 (−1.4%) | 0.961→0.303 |
| seq-H | 95→120 | 0.909→0.881 (−3.1%) | 0.609→0.130 |
| 01_1751 | 20→30 | 0.868→0.815 (−6.1%) | 0.250→0.060 |
| 02_1610 | 26→66 | 0.509→0.463 (−9.0%) | 0.322→0.096 |
| 04_8618 | 136→147 | 0.860→0.748 (−13%) | 0.723→0.207 |

`04_8618` delays acquisition by 11 frames yet loses 13% — **steady-state** loss, and
it is the most detector-dependent sequence (`yolo_rate` 0.723). So acquisition
gating would recover the three total failures and most seed-driven loss, but
sequences that lean on YOLO every frame still degrade.

Per-attribute: ALL present 0.531→0.412, DBC 0.631→**0.369** (−41%), TS 0.730→0.569,
FM 0.427→0.349 (FM is measurable here, unlike measurement 6).

### Verdict

**N=3 is not viable at the deployed operating point.** +57.8% fps costs ~23%
relative success and three lost tracks out of twelve. Unmeasured and worth one run:
**N=2** at this operating point (+34.6% fps) with acquisition gating — the only
configuration where the trade might close. `infer-interval` stays at its default
of 1.

Repro: `tools/samurai/detector-decimation/gt_score_ab_deployed.sh`; scorecards
`scorecard_gtd_n{1,3}.md`.

## PENDING — acquisition-gated run left in flight (2026-07-28)

`nvmminfer infer-gate-frames` is implemented (repo + ported to the deploy tree, built
there) and a GT A/B is **running unattended on the box**, launched with
`setsid nohup` so it survives disconnection. It was at 4/12 sequences when the
session ended.

- Script: `tools/samurai/detector-decimation/gt_score_gated.sh`
- Log: `$ASSET_DIR/gtg_ab.log` (finishes on `GATED-AB-DONE`)
- Arms: `n2g5` (interval 2, gate 5) and `n3g5` (interval 3, gate 5)
- Predictions: `$ASSET_DIR/results/gtg_n2g5/`, `.../gtg_n3g5/`
- Scorecards written at the end: `results/scorecard_gtg_n{2g5,3g5}.md`
- Baseline to compare against: `results/scorecard_gtd_n1.md` (success 0.490,
  miss 0.281, state_acc 0.476), and ungated N=3 in `scorecard_gtd_n3.md`
  (success 0.378, miss 0.461, 3 sequences never acquiring)

**What to look at first:** `miss` and the count of sequences with `seed_latency_f = -`
(never seeded). The gate exists to fix exactly those. Do **not** read the
`seed_latency_f` mean — `$SCORER` excludes never-seeded sequences from it,
so losing tracks makes it look better.

**Known limitation of this gate**: it keys on ANY detection, not target-class, because
`nvmminfer` cannot see the target class. Spurious detections (this dataset produces
plenty on cloud and terrain) can open the gate while the real target is still
unacquired. If the gated result is only partially recovered, that is the first
suspect — the fix would be plumbing track state upstream, as `nvmmfusekf` already
does for its `nvmm-reseed` event.

**Also unmeasured**: fps at the *deployed* pipeline shape. The 20.20 fps baseline
(and hence every +34.6%/+57.8% figure) comes from the simplified
`clip1.mp4 → nvmminfer → nvmmsamurai → nvmmfusekf` pipeline in the gmc build, not from
the deployed shape with `nvmmdetgate` + teardown. ~10 min with `pipeline_bench.py`
against `builddir-fix`.

**Cosmetic defect in the deploy tree only**: its `infer-interval` property blurb shows
a literal `~50%%` and still cites "~8% GT success", which measurement 7 superseded
with ~23%. The repo's own blurb does not carry that claim.

## Measurement 8 — acquisition gating: recovers the quality, gives back the throughput

`infer-gate-frames=5` holds decimation off until 5 consecutive inferred frames have
produced a detection, re-arming on the first that does not. GT, deployed operating
point, all 12 sequences.

| config | success | state_acc | mean_iou | miss | never acquired |
|---|---|---|---|---|---|
| N=1 baseline | 0.490 | 0.476 | 0.388 | 0.281 | 0 |
| **N=2 gate=5** | **0.478** (−2.4%) | 0.468 (−1.7%) | 0.380 | 0.288 | **0** |
| **N=3 gate=5** | **0.466** (−4.9%) | 0.461 (−3.2%) | 0.372 | 0.326 | **0** |
| N=3 ungated | 0.378 (−22.9%) | 0.393 | 0.295 | 0.461 | **3** |

Gating cuts N=3's quality cost **4.7×** (−22.9% → −4.9%) and fixes all three
never-acquiring sequences. Seven of twelve seed latencies become *identical* to
baseline (136, 222, 95, 175, 858, 72, 491), confirming the detector really does stay
at full rate through acquisition. The sequence that went to 0.000/never-seeded
ungated returns to 0.541 at its exact baseline seed latency of 222.

Residual: `seq-F` is still not protected — success 0.213 → 0.137,
miss 0.015 → **0.501**, seed latency 16 → 75. Consistent with the any-detection
proxy: spurious `person` hits on cloud/terrain can open the gate while the real
target is unacquired. A target-class- or track-state-aware gate would likely take
another bite out of the residual.

## Measurement 9 — deployed-shape fps. THE GATED GAIN IS +14%, NOT +57.8%.

`pipeline_bench.py`, deployed element list, clip1.mp4, 2 iterations (spread ≤0.4%):

| config | fps | ms/frame | gain |
|---|---|---|---|
| interval=1 | 19.50 | 51.3 | — |
| interval=2, gate=5 | **22.23** | 45.0 | **+14.0%** |
| interval=3, gate=5 | **22.23** | 45.0 | **+14.0%** |

Also: deployed shape at interval=1 is 19.50 fps vs 20.20 on the simplified chain, so
the extra deployed elements cost only ~3.5%.

**Correction to measurement 8's framing.** Gated N=3 was earlier described as
"+57.8% fps for −4.9% success". Wrong: +57.8% was measured **ungated**. The gated
throughput is **+14.0%**, so the real trade is **+14% fps for −4.9% GT success** —
much weaker. Gated quality was being paired with ungated throughput, a configuration
that does not exist.

**Why gating costs the throughput.** Detector-off is ~24.8 ms/frame; gated N=3 sits
at 45.0 ms/frame, so ~20 ms of detector still runs *every* frame — far more often
than 1-in-3. `clip1`'s detection rate is 83.7%, so ~1 frame in 6 has none, and
`gate=5` demands five consecutive hits to reopen; with gaps that frequent the gate is
mostly closed and the interval never reaches a skip. That also explains N=2 ≡ N=3
exactly: if a skip decision is rarely reached, its depth is irrelevant. The two knobs
are coupled.

### Final verdict on detector decimation

**Closed. Not worth landing.** Ungated: +57.8% fps for −22.9% success and 3 of 12
tracks lost. Gated: quality recovered to −4.9% but throughput collapses to +14%.
Neither point is a good trade, and the gate's coupling means there is no obvious
setting in between. `infer-interval` and `infer-gate-frames` both stay at their
defaults (1 and 0 = off); they are kept as instruments, not features.

Unmeasured, if anyone revisits: a skip-rate count (`GST_DEBUG=nvmminfer:6` emits one
`frame N: M detections` line per *inferred* frame) would confirm the re-arming
explanation directly rather than by arithmetic.

Overlay videos for visual inspection: `render_overlays.sh` →
`$ASSET_DIR/overlays/` (`clip1_n1`, `clip1_n3gate`, `seq-F_n1`,
`seq-F_n3gate`, plus `watch.html`).
