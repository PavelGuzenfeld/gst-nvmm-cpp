# Plan: drop the OpenCV runtime dependency from analytics/

Branch `feat/analytics-drop-opencv` off `main` at d12ec54 (PR #57 merge).

## Goal

Eliminate OpenCV from the shipped `analytics/*.hpp` headers entirely. **Not** by
replacing each OpenCV call 1:1 — by designing **fused single-pass stages** that
collapse each component's op chain into as few sweeps over the image as the
dataflow allows, cutting latency and memory traffic. OpenCV survives only as a
test-time reference oracle in opt-in golden-comparison tests and benchmarks.

Per-op replacement (a `sobel()`, a `gaussian_blur()`, a `morph_close()` mirroring
cv::) is an explicit **anti-goal**: it reproduces OpenCV's pass structure and its
intermediate-buffer traffic. Shared code between components is fine where two
fused stages are genuinely identical (e.g. a separable-blur core), but the unit
of design is the fused stage, not the OpenCV op.

## Resolved design decisions

1. **Buffer type**: new analytics-local header (`analytics/image.hpp`): owning
   `img::Image<T>` + non-owning strided `img::View<T>`, `uint8_t`/`float`,
   sub-rect views, indexing only — no elementwise-op method surface (ops live
   inside the fused stages). `gst/common/nvmm_buffer.hpp` was evaluated and
   rejected: it is an RAII handle over NVIDIA's `NvBufSurface` (dma-buf, plane
   mapping), not a host algorithm buffer.
2. **Existing behavioral tests**: keep, but port their synthetic-scene
   generators off OpenCV (onto `Image<T>`), so every `-Danalytics=enabled` build
   runs them with zero OpenCV. The `analytics` meson option description changes
   accordingly (no longer "Requires OpenCV when enabled").
3. **Golden tests**: fused-stage granularity **and** end-to-end component
   granularity, gated behind a new default-disabled `analytics_golden` feature
   (the only place OpenCV remains). Metrics per stage class:
   - linear stages (blur, gradient, IIR, warp-sample): max-abs-diff / PSNR
     with documented tolerance;
   - thresholded stages (masks, morphology): mask-disagreement-rate — a hard
     nonlinearity flips pixels on ~1e-4 float drift near the threshold, so
     bit-parity is not the bar;
   - RANSAC-dependent output (dual_homography): OpenCV's RANSAC is randomized —
     compare homography *quality* (reprojection error on known-truth synthetic
     warps, residual suppression) rather than output equality.
4. **dual_homography features**: implement **both** pipelines behind one
   interface and let the results decide:
   - (a) small-motion pipeline: FAST-style corners + ZNCC patch matching within
     a small search radius + 4-point DLT/SVD homography in a fixed-seed RANSAC
     loop (deterministic, sized to the real use case: near-consecutive frames,
     small pan, negligible rotation);
   - (b) full from-scratch ORB: oriented FAST + rotated-BRIEF descriptors +
     Hamming KNN + Lowe ratio, mirroring the current OpenCV path.
   Golden tests + benchmarks run both; the PR reports quality and latency side
   by side and recommends a default.

## Fusion plan per component (easy → hard)

1. **active_region.hpp** — currently 4 full-image `cv::reduce` sweeps.
   → **one** sweep accumulating per-row min/max and per-column min/max
   simultaneously; the bar-trim scan is unchanged. CUDA: skip — a single sweep
   over a gray frame is memory-bound and trivial on CPU; not worth a kernel
   unless benchmarks say otherwise.
2. **motion_magnify.hpp** — currently ~6 whole-image temporaries per frame
   (convert, blur, 2×IIR update, band, output).
   → **one** fused per-pixel pass: convert + both IIR updates + band + magnified
   output, O(1) state per pixel as now. Optional blur stays a separable
   pre-pass sharing the blur core with #3.
3. **low_texture_motion.hpp** — currently ~10 passes.
   → ~3 fused stages:
   - (a) gradient pass: Sobel-x + Sobel-y + magnitude in one sweep;
   - (b) mask pipeline: separable Gaussian blur where the final column pass
     fuses the threshold; morphological close of a square SE is separable
     (row-max then col-max, then row-min/col-min), fused with the threshold
     output;
   - (c) output pass: `min(|cur−ref_a|, |cur−ref_b|) · mask` → float, plus
     border-zeroing, in one sweep. Optional output blur reuses the blur core.
   Primary CUDA candidate: every stage is embarrassingly parallel and fusion
   pays most on GPU (avoids global-memory round trips).
4. **detection_motion_gate.hpp** — cv::Mat plumbing + `minMaxLoc` on a sub-rect.
   → trivial max-over-window helper on `img::View`; falls out once #1–#3 land.
5. **dual_homography.hpp** — hardest, last. Fusions independent of the feature
   pipeline choice:
   - `residual()` currently does warp + absdiff + second ones-warp + compare +
     erode + masked copy (≈6 passes, 2 of them full perspective warps).
     → **one** inverse-warp pass: per output pixel compute the source
     coordinate, bilinear-sample, absdiff, and validity — with the erode folded
     in as a coordinate-margin test against the source rectangle (no second
     warp, no erode pass, no mask buffer).
   - the two-reference `min` combine + blur + border-zero fuse into the
     surrounding passes where profitable.
   - fix in passing: `orb_match` hard-codes `>= 20` where `p.min_matches` was
     intended; the rewrite honors the param.
   - feature pipelines (a)+(b) per decision 4 above.
6. **persistence_gate.hpp** — already pure C++. Untouched.

## CUDA scope

Per-component, evidence-driven: low_texture_motion (and possibly the
dual_homography warp/residual pass) are the candidates; active_region and the
gate plumbing are not. Wiring: new narrow `have_cuda`-style gate — analytics
kernels need neither TensorRT nor NvBufSurface, so do **not** reuse the
`have_tensorrt and have_nvbufsurface` Jetson gate; follow the existing
`add_languages('cuda', ...)`/`gpu_arch`/`gpu_cxx_std`/`cuda_args` conventions
and the `tests/samurai_kernel_probe` host-vs-CUDA parity-test pattern.

**Resolved**: only `low_texture_motion` got a CUDA kernel
(`analytics_kernels.hpp`/`.cu`). The benchmark evidence closed the
dual_homography question — its fused CPU `small_motion` pipeline already beat
OpenCV's own ORB path (1.4x, see the PR #58 benchmarks), so there was no
latency gap left for a CUDA port to close.

**Added after the initial design** (Jetson/plugin-zoo interop requirement):
`run_device(DevicePlane<...>, ..., cudaStream_t)` takes pitched DEVICE
pointers directly — e.g. the CUDA mapping of an NvBufSurface luma plane — and
enqueues all work on the caller's stream with no host round-trip. `run()`
(upload/execute/download host buffers) is a convenience wrapper over it for
tests and benchmarks; `DevicePlane<T>` is the pitched-pointer type shared by
both.

## Testing & benchmarks

1. **Behavioral tests** (existing, ported): scene generators rewritten on
   `Image<T>`; assertions unchanged in intent. Run in every
   `-Danalytics=enabled` build, OpenCV-free.
2. **Golden tests** (new, `-Danalytics_golden=enabled`, default disabled,
   requires OpenCV): fused-stage + component level per decision 3; tolerance
   and metric documented per stage in the test file.
3. **Benchmarks** (new, in `benchmarks/`, `bench_nvmm.cpp` CSV convention,
   meson `benchmark()`), as two separate tools rather than one — the golden
   lane doesn't need CUDA and the CUDA lane doesn't need OpenCV, so a single
   binary would force an unwanted dependency on one lane or the other:
   - `bench_analytics` (`-Danalytics_golden`, needs OpenCV): OpenCV vs
     fused-CPU at representative frame sizes; both dual_homography pipelines
     measured against OpenCV's ORB path.
   - `bench_analytics_cuda` (`-Danalytics_cuda`, needs the CUDA toolkit):
     fused-CPU vs CUDA for the one component with a GPU path
     (low_texture_motion).
   A three-way OpenCV/fused-CPU/CUDA comparison is assembled by hand from the
   two CSVs when both lanes are available (see the PR description). Losing to
   an OpenCV SIMD kernel on some op is a legitimate finding to report, but the
   headline metric is per-component end-to-end latency, where fusion is
   expected to win.

## Conventions

C++14, `-Wall -Wextra -Werror` clean; header-only `#pragma once`,
`namespace nvmm { namespace ... {`; comments only for non-obvious *why*;
local `TEST(name)` harness, meson `test(..., protocol : 'exitcode')`;
persistence_gate.hpp and pipeline docs untouched unless a public API changes.

## Exit criteria

- `analytics/*.hpp` (except persistence_gate.hpp): no OpenCV includes or `cv::`
  types.
- Default build unaffected; `-Danalytics=enabled` builds **and passes tests**
  without OpenCV installed.
- `-Danalytics_golden=enabled` builds golden tests + benchmarks (needs OpenCV),
  all passing within documented tolerances.
- Benchmark CSVs checked in or reported in the PR description, including the
  dual_homography (a)-vs-(b) comparison and a recommended default.
