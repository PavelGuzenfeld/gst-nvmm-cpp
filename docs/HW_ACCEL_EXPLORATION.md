# Hardware Acceleration Exploration Plan

Status: **exploration / proposal** (not yet implemented)

## Motivation

gst-nvmm-cpp already owns the hard part: a zero-copy `NvBufSurface` backbone
(custom allocator, GPU-copy IPC pool, fd passing) and a clean C-ABI GStreamer
element pattern (`nvmmconvert`, `nvmmsink`, `nvmmappsrc`). Today only **one**
Tegra fixed-function engine is used — the **VIC** (2D scale/crop/convert/flip via
`NvBufSurfTransform`).

The same SoC carries several other offload engines that consume the *same*
`NvBufSurface`/DMA-buf memory, so we can extend the suite to drive them with no
extra copies. This doc captures (A) VIC capabilities we leave on the table today,
and (B) new engines worth wrapping, with a phased plan.

Target hardware (verified box: Jetson Xavier NX, JP5.1.2; also Orin NX):

| Engine | Xavier NX | Orin NX | Reached via |
|---|---|---|---|
| VIC (2D) | Gen 4 | Gen 4.2 | `NvBufSurfTransform` (we use) |
| NVDEC (decode) | yes | yes (+AV1) | V4L2 `nvv4l2decoder`, NvMedia |
| NVENC (encode) | **yes** | **NO** | V4L2 `nvv4l2*enc` |
| NVJPG (JPEG) | yes | yes | `nvjpegenc/dec`, NvJPEG |
| OFA (optical flow) | — | yes (dedicated) | VPI |
| PVA (vision DSP) | 2× | 1× (v2) | VPI |
| DLA (inference) | 2× | 1× (v2) | TensorRT / cuDLA |
| ISP (camera) | yes | yes | libargus |
| GPU (CUDA) | Volta | Ampere | CUDA / VPI |

> **Caveat that shapes priorities:** Orin NX has **no NVENC**. Any encode element
> must degrade gracefully (or be documented Xavier-only / GPU-fallback).

---

## Part A — VIC capabilities we do NOT use yet

`nvmmconvert` exposes only: src-crop, scale, color-convert, and flip/rotate
(90/180/270, H/V). `NvBufSurfTransform` (+ its Composite/Blend siblings) offers
considerably more, all on the VIC we already drive:

1. **Interpolation/filter selection** — `NvBufSurfTransformInter_{Nearest,
   Bilinear, Algo1(5-tap), Algo2(10-tap), Algo3(Smart), Algo4(Nicest)}`. We
   always use Default. Exposing a `interpolation` property is a quality/perf knob
   (e.g. 5-tap for downscale quality, Nearest for speed).
2. **Destination crop / placement** (`NVBUFSURF_TRANSFORM_CROP_DST`) — place the
   scaled image into a sub-rectangle of the destination. Enables
   **aspect-ratio-preserving fit with letterbox/pillarbox padding** — currently
   impossible (we only src-crop).
3. **Transpose / inverse-transpose** flips (`FlipMethod` 5 and 7) — already in our
   `nvmm::FlipMethod` enum but **not exposed** in the `flip-method` GEnum. Trivial
   to add.
4. **Normalization** (`NVBUFSURF_TRANSFORM_NORMALIZE`) — per-channel mean-subtract
   + scale in the same VIC pass. This is **AI inference preprocessing for free**
   (the classic normalize step before TensorRT/DLA).
5. **Composite / tiling** (`NvBufSurfTransformComposite`) — stitch multiple NVMM
   inputs into one surface (multi-camera mosaic / NxM grid). No element today.
6. **Blend / overlay** (`NvBufSurfTransformCompositeBlend`,
   `MultiInputBufCompositeBlend`) — alpha-blend an overlay (OSD, watermark,
   segmentation mask) onto video. No element today.
7. **Batch transforms** (`batchSize > 1`) — transform an array of surfaces in one
   VIC call. We always run batch=1; batching raises throughput for multi-stream.
8. **Async + explicit compute mode / CUDA stream** (`NvBufSurfTransformConfigParams`
   + `*Async`) — we run synchronous, `Compute_Default`. We could (a) pin to
   `Compute_VIC` to keep the GPU free, or `Compute_GPU` when VIC is saturated, and
   (b) pipeline via async + sync objects.

These are the cheapest wins — same engine, same buffers, mostly new properties on
`nvmmconvert` or one new `nvmmcompositor` element.

---

## Part B — New engine plugins

Each reuses the existing `NvBufSurface` allocator + buffer-pool plumbing, so
input/output stay zero-copy on the GPU.

### B1. `nvmmjpegenc` / `nvmmjpegdec` — NVJPG (both chips)
Hardware JPEG straight from/to NVMM, no CPU bounce. `nvjpegenc` exists upstream
but goes through system memory; an NVMM-native variant keeps the surface on-GPU
end-to-end (camera → ISP → NVMM → NVJPG → file/socket). Low risk, both SoCs.

### B2. `nvmmcompositor` — VIC composite/blend (both chips)
Multi-pad mixer using `NvBufSurfTransformMultiInputBufCompositeBlend`: N NVMM
inputs → one tiled/blended NVMM output. Headline use case: multi-camera fan-in to
a single mosaic, or OSD/mask overlay. Pairs naturally with the existing
multi-camera `nvmmsink` fan-out.

### B3. `nvmmofa` — Optical Flow Accelerator via VPI (**Orin only**)
Dense/semi-dense optical flow between consecutive NVMM frames, output as a motion-
vector surface. VPI exposes OFA with zero-copy NvBufSurface wrapping. Gated to
Orin (Xavier has no dedicated OFA) — build-time / runtime capability check.

### B4. `nvmmcv` — PVA vision ops via VPI (both chips)
Wrap selected VPI algorithms that run on PVA (e.g. stereo disparity, KLT feature
tracking, box/Gaussian filtering, remap/undistort) as GStreamer transform
elements. PVA keeps GPU/CPU free for other work. PVA v2 on Orin, 2× PVA on Xavier.

### B5. `nvmminfer` — DLA/GPU inference via TensorRT (both chips)
Run a TensorRT engine on a DLA core (fallback GPU) directly on NVMM input, with
the VIC `NORMALIZE` preprocessing (Part A.4) folded in. Output tensors/meta on the
GStreamer bus. The most ambitious; depends on Part A.4 + a tensor-meta design.

### B6. `nvmmenc` — NVENC (**Xavier only; document Orin gap**)
NVMM-native H.264/H.265 encode. Must detect NVENC absence on Orin NX and either
fail with a clear message or fall back to a GPU/CPU encoder. Lower priority given
the portability footgun.

---

## Phasing

**Phase 0 — Exploration (this branch).** For each candidate: confirm the API
(NvMedia/VPI/TensorRT) accepts our `NvBufSurface` zero-copy, write a tiny
standalone reproducer on the Jetson (same method as the IPC root-cause work:
build in the `gst-nvmm-cpp:jp5` image, verify on hardware), and record findings
here. Output: a go/no-go + effort estimate per element.

**Phase 1 — VIC quick wins (Part A).** Add to `nvmmconvert`: `interpolation`
property, dst-crop/letterbox, transpose/inv-transpose enum values, and (optional)
`compute-mode`. Add `nvmmcompositor` (composite/blend). All on the engine we
already trust; unit-testable in mock + pipeline-testable on hardware.

**Phase 2 — NVJPG + Compositor productionized (B1, B2).** Highest value/lowest
risk, both SoCs.

**Phase 3 — VPI engines (B3 OFA Orin, B4 PVA).** New dependency (VPI); gate by
SoC capability.

**Phase 4 — Inference (B5) and NVENC (B6).** Largest scope; B6 only where NVENC
exists.

Each element follows the established pattern: C-ABI GStreamer element, mock build
for CI (extend `nvbufsurface_mock.h` with the new API stubs), real verification on
the Jetson, README + tests, then a version bump + tag.

## Open questions for Phase 0
- Does VPI wrap an externally-allocated `NvBufSurface` zero-copy, or require its
  own `VPIImage` allocation (a copy)? Determines B3/B4 viability.
- TensorRT DLA: input binding from an `NvBufSurface` device pointer without a copy?
- Mock strategy for VPI/TensorRT in CI (stub vs. skip-on-host).
