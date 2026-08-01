# `nvmmgeoanchor` — absolute position anchoring from georeferenced imagery (plan)

> Status: **PLAN ONLY — nothing built.** No phase gate cleared. Sibling of the
> B5 inference-graph family; reuses `TrtEngine`, the VIC/NPP preprocess, the
> meta pattern, and the `gmc_backend.hpp` swappable-backend precedent.
> Target: Orin first.

## What this is

One element that turns a live NVMM video stream plus a coarse pose prior into
an **absolute position fix with a covariance**, by matching each (decimated)
frame against **georeferenced reference imagery** — an orthophoto tile store on
disk.

It is a **passthrough** on the video path. The video buffer comes out unchanged;
the result is metadata, exactly like `nvmm_det_meta` / `nvmm_track_meta` /
`nvmm_optical_flow_meta`.

```
sink: live NVMM video ─────────────────────────────► src: same buffer, passthrough
                         │                     ▲
                         │ pose-prior meta     │ geofix meta
                         ▼                     │
             ┌────────────────────────────────────┐
             │ 1. tile select      (prior → tile) │
             │ 2. prewarp          (VPI Remap)    │
             │ 3. preprocess       (VIC + NPP)    │
             │ 4. match            (backend)      │
             │ 5. solve + gates    (host C++)     │
             └────────────────────────────────────┘
```

Intended graph position — a sibling branch of the existing tracker graph, not a
replacement:

```
src → tee ─┬─ nvmminfer → nvmmtracker ─────────┐
           ├─ nvmmofa → nvmmflowstats ─────────┼→ nvmmfusion → consumer
           └─ nvmmgeoanchor ───────────────────┘
```

## Why the accuracy does not come from the matcher

The dominant error terms are not matcher pixel error. Ground position error is
roughly:

```
σ_pos² ≈ (σ_px · GSD)²              matcher
       + (σ_attitude · slant_range)² attitude
       + (σ_DTM · tan θ)²            terrain, θ = off-nadir
       + σ_georef²                   reference imagery georeferencing
```

At a typical 0.3–0.5 m/px reference GSD, the difference between a good and a
very good matcher is a few tenths of a metre — under the DTM and attitude terms
on anything off-nadir. **The prewarp (step 2) is the biggest single lever**, not
the matcher: projecting the reference tile into the predicted camera view using
DTM + attitude collapses a wide-baseline, large-scale-ratio, extreme-viewpoint
match into a near-planar registration. Every published result showing matchers
struggle is measuring the problem the prewarp deletes.

Consequence for the design: **once prewarped, extreme-viewpoint robustness stops
being the discriminator.** A heavyweight dense matcher stops paying for itself
and throughput plus gating dominate. Hence the backend split below.

## Decisions (proposed — none validated yet)

| # | Decision | Choice |
|---|---|---|
| Shape | Aggregator (2 sink pads) vs single-pad | **Single sink pad, passthrough.** The second image is not a stream — it is a tile from a map store, selected internally by the prior. |
| Output | New buffer vs meta | **Meta.** `nvmm_geofix_meta`, same pattern as the existing metas. Video path untouched. |
| Matcher | Fixed vs swappable | **Swappable backend**, `matcher-backend = xfeat \| romav2`. Follows `gmc_backend.hpp` / `gmc_vpi_fft.hpp` / `gmc_vpi_pva.hpp` precedent in `nvmmsamurai`. |
| First backend | Which one ships first | **`xfeat`.** `gst/common/xfeat_matcher.{cpp,hpp}` + `xfeat_sparse.hpp` + `xfeat_register.hpp` are already in-tree, OpenCV-free, host C++. Descriptor + match half is done. |
| Second backend | Dense matcher | **`romav2`** — RoMa v2 ONNX → `trtexec` → `.engine`, bound through the existing `TrtEngine`. ~1 GB fp32 graph, ViT-L, GPU-only. Deferred behind a measured need. Upstream weight licence to be reviewed before vendoring. |
| Prewarp | CUDA kernel vs fixed function | **VPI `Remap` on VIC.** Verified backend support: Remap and Perspective Warp are CPU/CUDA/**PVA**/**VIC**. Keep the GPU free; keep PVA free for GMC (see contention). |
| Rate | Every frame vs decimated | **Decimated + async.** `interval` property (the `nvmminfer` idiom), worker thread, queue depth 1, drop-oldest. The matcher must never block the streaming thread. |
| Timestamps | Attach-time vs origin PTS | **Origin PTS, carried in the meta.** Non-negotiable — see below. |
| Inter-fix motion | New OF vs reuse | **Reuse `nvmmofa` output.** Single OFA on the SoC; share the result, do not submit twice. |
| Solve | Reuse `xfeat_register` projector | **No — new code.** `PointToPointProjector` is a 3-point affine from the 9 nearest keypoints: a local projector for tracking, not a global fit with residuals. Global solve + covariance is the real new work. |
| Fusion | Inside vs outside the element | **Outside.** Emit fix + covariance; `nvmmfusekf` consumes it. Element does no filtering. |
| CI | Host build | **Skip-on-host**, following the VPI / `nvmminfer` precedent. Host-CI-able: metas, gates, solve, tile-cache eviction. Jetson-only: prewarp, preprocess, TRT backend. |

## The PTS contract

The meta **must** carry the PTS of the frame the fix was computed from, not the
PTS of the frame it is attached to.

A decimated, asynchronous matcher attaches its result some hundreds of
milliseconds after the originating frame. A downstream filter that treats the
fix as a current measurement injects a latency-proportional position error —
potentially larger than the drift being corrected. Put `src_pts` in the struct
on the first commit; retrofitting it is painful and the failure is silent.

```c
typedef struct {
    GstMeta   meta;
    GstClockTime src_pts;        /* frame the fix was computed FROM */
    double    pos[3];            /* solved position, reference frame */
    double    cov[9];            /* 3x3 row-major */
    guint32   inliers;
    float     inlier_ratio;
    float     spread;            /* inlier spatial coverage / conditioning */
    float     margin;            /* best vs runner-up hypothesis */
    gboolean  accepted;          /* survived all gates */
    guint8    reject_reason;
} NvmmGeoFixMeta;
```

Input side: `nvmm_pose_meta` (position + attitude + altitude prior, with its own
covariance). **Verify first** whether `docs/metadata-ipc.md` /
`gst/common/shm_protocol.h` already carries a pose channel — if so, reuse it and
skip the new meta.

## False-positive rejection

A missed fix costs drift. A wrong fix corrupts the filter. Gates are ordered
cheapest-first and are all recorded in the meta so rejections are diagnosable:

1. **Certainty / confidence mask** from the matcher. The dense backend emits a
   per-pixel precision matrix — use it to weight the solve, not just threshold.
2. **Robust global fit** (RANSAC family) → homography or PnP against the DTM.
3. **Inlier count *and* inlier ratio**, thresholded separately. Ratio alone
   passes degenerate low-match cases.
4. **Spatial spread / conditioning** of inliers. Inliers clustered in one
   quadrant give a well-fitting, badly-conditioned solution.
5. **Hypothesis margin** — when several candidate tiles are matched (below), the
   winner must beat the runner-up by a margin or the fix is rejected.
6. **Covariance propagation** from the solve Jacobian. Published work on
   localization failure detection finds propagated covariance ranks failures
   better than inlier count alone, so do not stop at (3).
7. **Innovation gate** against the prior (chi-squared). The only gate using
   information independent of the image — highest value of the set.
8. **Temporal consistency** — N-of-M mutually consistent fixes required before a
   re-acquisition after a gap is trusted.

Throughput and false-positive rate are **one operating point**, not two goals:
tighter gates mean fewer accepted fixes. The target is stated as
**accepted-fix rate at a bounded false-fix rate**, and both numbers get reported.

## Accelerator budget

Decompose by rate, across engines, rather than trying to split one match:

| Rate | Work | Engine | Existing code |
|---|---|---|---|
| frame | dense optical flow → inter-fix motion | **OFA** (exclusive) | `nvmmofa`, `probes/vpi_ofa_probe.cpp` |
| frame | NV12→RGB, crop, rescale | **VIC** | `nvmm_transform.cpp`, `nvmminfer/preprocess.cpp` |
| frame | corner / texture pre-gate (optional) | **PVA** | `gmc_vpi_pva.hpp`, `probes/vpi_pva_probe.cpp` |
| fix | reference-tile prewarp | **VIC** (or PVA) | new — VPI Remap |
| fix | match | **GPU** (or DLA, xfeat only) | `nvmminfer/trt_engine.*` |
| fix | solve, gates, covariance | **CPU** | new |

### Contention — these engines are not spare capacity

- **PVA**: Orin NX has one PVA with two vector processors → at most **two
  concurrent PVA tasks**. `nvmmsamurai`'s GMC backend already claims it. This is
  why the prewarp is planned on **VIC**, not PVA.
- **OFA**: single engine. Share `nvmmofa`'s output; do not submit twice.
- **VIC**: Gen 4.2, shared with the whole video path including encode.
- **DLA**: Orin NX has **one** NVDLA v2.0. Detector on DLA *or* XFeat on DLA,
  not both without time-slicing.
- **DLA and transformers**: ViT / attention does not run on DLA as of JP6.2.
  `--allowGPUFallback` moves the whole graph back to GPU and adds transfer
  overhead. DLA offload is therefore viable for the **xfeat backend only**.

Measure current occupancy (`tegrastats` + VPI timestamps) **before** designing
around any of these.

## Parallelism inside the fix

A single dense-matcher forward is one engine and cannot be split across
accelerators. What can be parallelised:

1. **Batched multi-hypothesis.** When the prior is loose, stack N candidate
   tiles on the batch axis and match them in one launch instead of N sequential
   calls. Better occupancy, and it produces gate (5) for free.
2. **Coarse→fine cascade.** Screen candidates at low resolution, refine only the
   survivor at full resolution. Largest fix-rate gain for the least code.
3. **Stage pipelining.** Separate VPI and CUDA streams per stage, depth 3:
   VIC preprocesses frame N+1 while PVA/VIC prewarps tile N+2 while the GPU
   matches N. Turns sum-of-stages into slowest-stage.
4. **DLA offload** of the xfeat descriptor, freeing the GPU for the detector and
   tracker sharing the device. Only if measurement shows GPU contention is the
   binding constraint.

## Reference-tile residency

The prewarped-tile cache is a real memory budget line on a unified-memory SoC
shared with the video pipeline and (on the dense backend) a ~1 GB model:

```
cache_bytes = n_tiles × tile_bytes
```

Eviction keyed on prior motion — keep along-track tiles, drop tiles behind.
`gst/nvmmsecondaryinfer/secondary_cache.{cpp,hpp}` is the template. Size this
before building it; this is what separates working-on-the-bench from OOM in the
field.

## Verify before building

Assumptions in this plan that are **not** yet confirmed:

- [ ] Does `gst/nvmminfer/preprocess.cpp` already emit RGB float32 `[0,1]` NCHW?
      If so the dense backend's input path is nearly free.
- [ ] Do `analytics/detail/homography.hpp` / `analytics/dual_homography.hpp`
      provide a global fit with usable residuals, or is the solve fully new?
- [ ] Does the shm IPC path already carry a pose/attitude channel?
- [ ] VIC Remap throughput and accuracy for the DTM prewarp at target tile size.
- [ ] Actual PVA/VIC/OFA occupancy with the tracker graph running.

## Phases

### Phase 0 — gate

Offline, no element. Feed recorded frames + recorded prior + a reference tile
store through a host harness using the existing `xfeat_matcher`, a prewarp, and
a global solve. **Question: does the prewarp + sparse path meet the position
error budget on real reference imagery?**

Report accepted-fix rate at a bounded false-fix rate, and the error breakdown by
term. **If yes, the dense backend is never needed and Phase 4 is dropped.**
This gate exists to avoid building a 1 GB engine for a problem the prewarp
already solved.

Second, separate question at this gate: **reference staleness.** Match against
imagery with the worst realistic season/age delta. No published benchmark covers
this, and it is the axis most likely to decide real-world false-positive rate.

### Phase 1 — metas

`nvmm_geofix_meta` (+ `nvmm_pose_meta` if not already available over IPC).
Host-CI-able, unblocks everything else.

### Phase 2 — prewarp

VPI `Remap` on VIC, DTM + attitude → tile projected into predicted view. Bench
against the CUDA reference for accuracy and against VIC occupancy for cost.

### Phase 3 — element, xfeat backend

`nvmmgeoanchor` with `matcher-backend=xfeat`: tile select, prewarp, match,
global solve, gate cascade, meta emit. Async worker, `interval`, origin-PTS
contract. Wire to `nvmmfusekf`.

Then: cascade (parallelism item 2) and batched multi-hypothesis (item 1).

### Phase 4 — dense backend (conditional on Phase 0)

RoMa v2 ONNX → engine → `TrtEngine`, behind the same backend interface.

Build notes: ~1 GB fp32 ONNX, build in the jp6-infer container with
`--memPoolSize`, not on a loaded device. The export is static H/W with dynamic
batch only, so one fixed profile suffices and `set_input_shape` is unnecessary.
`TrtEngine::load_file` consumes a serialized engine, so precision is fixed at
`trtexec` time — the upstream model asserts on reduced float32 matmul precision,
so validate any `--fp16` engine against the reference before trusting it.

## Risks

| Risk | Mitigation |
|---|---|
| Prewarp quality gates everything | Phase 0 measures it before any element work |
| Stale reference imagery drives false positives | Explicit Phase 0 sub-gate on worst-case season delta |
| Origin-PTS omitted → filter corruption | In the struct from Phase 1, before any consumer exists |
| PVA/OFA/DLA already saturated by the tracker | Occupancy measured at Phase 0; prewarp planned on VIC |
| Tile cache OOM | Sized and bounded at design time, eviction keyed on motion |
| Dense backend built and then unused | Phase 4 is conditional on Phase 0 failing |
