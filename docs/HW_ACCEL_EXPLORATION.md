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

## Reality check — what NVIDIA already ships (build vs. reuse)

Before building anything, map each idea against what NVIDIA already provides.
There are three off-the-shelf sources:

- **L4T "Accelerated GStreamer"** — free, ships with JetPack
  (`apt install nvidia-l4t-gstreamer`), upstream-ish GStreamer elements:
  `nvvidconv`, `nvv4l2decoder`, `nvv4l2h264enc`/`h265enc`, `nvjpegenc`/`nvjpegdec`,
  `nvarguscamerasrc`.
- **DeepStream SDK** — free to use but **heavyweight**: large install, its own
  `NvDsBatchMeta` metadata model, redistribution licensing, and version lock-step
  with specific JetPacks. Elements: `nvstreammux` (batch), `nvmultistreamtiler`
  (tiling/mosaic), `nvdsosd` (overlay/OSD), `nvinfer` (TensorRT incl. DLA),
  `nvof` (optical flow), `nvvideoconvert`, `nvdspreprocess` (ROI + normalize),
  `nvtracker`.
- **VPI** — a **library** (not GStreamer): OFA optical flow, PVA stereo/KLT, etc.
  No stock GStreamer element — you wrap it yourself.

### Mapping

| Capability | NVIDIA off-the-shelf | Verdict for this project |
|---|---|---|
| 2D scale/crop/convert/flip (VIC) | `nvvidconv` (L4T, free) | **Parity** — `nvmmconvert` overlaps it; our edge is open-source + integration with our NVMM pool/IPC, not new 2D features |
| Scaling interpolation filter | `nvvidconv interpolation-method` | **Parity** — our Phase-1 `interpolation` matches it; no new ground |
| JPEG encode/decode (NVJPG) | `nvjpegenc` / `nvjpegdec` (L4T, free) | **PASS** — reuse; only build NVMM-native variant if a zero-copy gap is *measured* |
| Video decode (NVDEC) | `nvv4l2decoder` (L4T, free) | **PASS** — reuse |
| Video encode (NVENC) | `nvv4l2h264enc` (L4T, free; Xavier only) | **PASS** — reuse |
| Camera / ISP | `nvarguscamerasrc` (L4T, free) | **PASS** — reuse |
| Tiling / mosaic | `nvmultistreamtiler` (DeepStream) | **BUILD only if avoiding DeepStream** — else pass |
| Overlay / blend (OSD) | `nvdsosd` (DeepStream) | **BUILD only if avoiding DeepStream** — else pass |
| Inference (DLA/GPU) | `nvinfer` (DeepStream/TensorRT) | **PASS** for DeepStream users; niche = lightweight no-DS element |
| Optical flow (OFA) | `nvof` (DeepStream) + VPI (lib) | **THIN-WRAP VPI** if open-source/no-DS wanted; else pass |
| Normalize for inference | `nvdspreprocess` (DeepStream) | VIC `NORMALIZE` (Part A.4) is the no-DS equivalent |
| **Cross-process zero-copy NVMM IPC** | **nothing** (DeepStream batches *within one process*; `nvstreammux` is not cross-process) | **BUILD — this is our unique value** |

### Conclusion that re-centers the plan

Single-engine wrapping is **mostly a solved problem** off-the-shelf. The honest
differentiated value of gst-nvmm-cpp is the combination NVIDIA does *not* ship:

1. **Cross-process, zero-copy NVMM sharing** (`nvmmsink`/`nvmmappsrc`) — share GPU
   frames between *independent* processes without DeepStream and without a CPU copy
   on the consumer. DeepStream is single-process; there is no stock element pair
   for this.
2. **Lightweight, open-source (LGPL), upstream-able** elements with a real
   `GstAllocator` for `NvBufSurface` — no SDK install, no metadata-model lock-in,
   no licensing friction.
3. **No DeepStream dependency** — for teams (ROS2 nodes, custom inference servers)
   that want NVMM interop but can't or won't pull in DeepStream.

**So prioritise the niche, not the wrappers:**
- **Do build:** anything that extends the cross-process IPC story (multi-consumer,
  metadata/timestamp sidechannel, ROS2-native consumer), and *small* open-source
  elements only where the no-DeepStream angle is the point.
- **Thin-wrap (only if no-DS matters):** OFA/PVA via VPI, a minimal DLA inference
  element.
- **Pass (reuse NVIDIA's):** JPEG, encode, decode, camera/ISP — document the
  recommended stock element instead of reimplementing.
- **Re-scope Part B accordingly:** `nvmmjpegenc`/`nvmmenc` drop to "only if a
  measured zero-copy gap" (likely PASS); `nvmmcompositor`/`nvmminfer`/`nvmmofa`
  become "no-DeepStream alternatives," explicitly positioned against their
  DeepStream equivalents.

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
8. **Explicit compute mode** (`NvBufSurfTransformConfigParams`) — **DONE** (v1.1.x):
   `nvmmconvert` has a `compute-mode` property (`default`/`gpu`/`vic`) — pin to VIC
   to keep the GPU free, or GPU when the VIC is saturated. Validated on Xavier +
   Orin. **Remaining:** async pipelining via `*Async` + sync objects.

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

### B2. `nvmmcompositor` — VIC composite/blend (both chips) — ✅ DONE (v1.2.0)
Multi-pad mixer: N NVMM inputs → one tiled NVMM output. Headline use case:
multi-camera fan-in to a single mosaic, or OSD/mask overlay. Pairs naturally with
the existing multi-camera `nvmmsink` fan-out.

**Shipped:** `GstAggregator`-based element with request `sink_%u` pads, each
carrying `xpos`/`ypos`/`width`/`height` placement; element `width`/`height` set
the output size. Each pad is blitted into its rectangle via `NvBufSurfTransform`
with `CROP_DST` (one VIC transform per pad) — simpler than the batched
`NvBufSurfTransformMultiInputBufCompositeBlend` and sufficient for opaque
mosaic/PiP. Built on `GstAggregator` because `GstVideoAggregator`'s
`GstVideoFrame` mapping does not understand NVMM memory.

**Validated dual-host:** 4 mock unit tests (create/props/request-pads/placement,
ASan+UBSan clean with `libasan` preload); 2-input composite run on Xavier
(JP5, Docker) and Orin (JP6, native), side-by-side placement visually confirmed
on Orin; sustained 600-frame 2×1080p throughput **~68 fps Xavier / ~164 fps
Orin**, rc=0. Docs: [`nvmmcompositor`](elements/nvmmcompositor.md) element page +
[compositing pipeline example](pipelines.md#multi-input-compositing-mosaic-pip).
See [Validation](validation.md).

**Future enhancement (not blocking):** the output buffer is not cleared, so
layouts must tile the full frame (or use a background pad); alpha blend / true
`CompositeBlend` and a clear/background option are deferred.

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
risk, both SoCs. **B2 `nvmmcompositor` done (v1.2.0)**; B1 NVJPG next.

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
