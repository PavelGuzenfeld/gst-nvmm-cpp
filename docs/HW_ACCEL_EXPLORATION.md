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

### B1. `nvmmjpegenc` / `nvmmjpegdec` — NVJPG (both chips) — ✅ PASS (reuse stock), measured 2026-06-07
Hardware JPEG straight from/to NVMM. **Measured, dual-host:** the assumption that
stock `nvjpegenc` "goes through system memory" is **wrong** — on both Xavier (JP5)
and Orin (JP6), `gst-inspect-1.0 nvjpegenc`/`nvjpegdec` advertise
`video/x-raw(memory:NVMM)` on the relevant pad, and
`videotestsrc ! nvvidconv ! 'video/x-raw(memory:NVMM),NV12' ! nvjpegenc ! filesink`
runs rc=0 with valid output (~670 KB) and **no** system-memory conversion in the
path. There is no zero-copy gap, so per the re-scope this stays **PASS** — document
the stock `nvjpegenc`/`nvjpegdec` recommendation instead of building a variant.

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

### B3. `nvmmofa` — Optical Flow Accelerator via VPI (**Orin only**) — ✅ DONE (v1.3.0)
Dense optical flow between consecutive NVMM frames on the OFA engine, with the
motion-vector field carried as per-frame metadata. Orin-only hardware (Xavier has
no OFA → documented N/A, like B6's NVENC).

**Shipped:** [`nvmmofa`](elements/nvmmofa.md) (`GstBaseTransform`, in-place):
NV12 frame passes through **zero-copy**; for each consecutive pair it wraps both
surfaces zero-copy (`VPI_IMAGE_BUFFER_NVBUFFER`), runs `vpiSubmitOpticalFlowDense`
on `VPI_BACKEND_OFA` (queries the wrapped input format at runtime — NvBuffer NV12
maps to an ER/BL variant, not plain `NV12_BL`), then locks the `2S16_BL` result
to host and attaches it as `NvmmOpticalFlowMeta`. Ships with `nvmmflowstats`, an
example sink that consumes the meta. `grid-size` (1/2/4/8) + `quality` props.

**Layout — corrected from the earlier note:** OFA *requires* **block-linear**
input (`NV12_BL`/`Y8_BL`/…), which is exactly what `nvvidconv` emits — so the
natural NVMM pipeline feeds OFA zero-copy. (The earlier "pitch-linear required"
was a wrong carry-over from B4's *PVA* morphology, which is the opposite case.)

**Honest zero-copy scope:** the *frame* is zero-copy throughout; the *flow field*
is not — OFA only writes a VPI-native `2S16_BL` image (a wrapped `SIGNED_R16G16`
NVMM surface is rejected), so the small field (e.g. 160×120 → ~75 KB) is
copied out to host metadata. The meta is also **in-process only** (does not cross
the nvmmsink→nvmmappsrc IPC boundary).

**Validated on Orin** (probe `probes/vpi_ofa_probe.cpp` + end-to-end): `nvmmofa`
→ `nvmmflowstats` produced and consumed a 160×120 grid-4 field; frame 1 correctly
empty (19/20 with flow); `grid-size=1` dense 640×480 works; 720p grid=4 sustains
**~46 fps**. Mean magnitude rises monotonically with scene motion (static smpte
4.59 px < ball 5.12 < noise 6.93), confirming the field tracks real content —
though the non-zero static baseline is the usual dense-flow artifact in
textureless regions, so the field is approximate, not per-cell exact. No unit
test in CI (VPI-gated, OFA Orin-only) — tested on-device, the documented
exception. See [Validation](validation.md).

### B4. `nvmmcv` — PVA vision ops via VPI (both chips)
Wrap selected VPI algorithms that run on PVA (e.g. stereo disparity, KLT feature
tracking, box/Gaussian filtering, remap/undistort) as GStreamer transform
elements. PVA keeps GPU/CPU free for other work. PVA v2 on Orin, 2× PVA on Xavier.

**Phase-0 verdict (2026-06-07): measured — zero-copy wrap PASSES, but PVA compute
is NOT a clean dual-host zero-copy win. Recommend NO-GO as specified; needs
re-scope.** A standalone reproducer (`vpi_pva_probe.cpp`: create a GRAY8
`NvBufSurface`, wrap it via `VPI_IMAGE_BUFFER_NVBUFFER`, run `vpiSubmitErode`) was
built and run on **both** hosts. Results:

| Host (VPI / PVA) | layout | zero-copy wrap | CUDA erode | **PVA erode** |
|------------------|--------|----------------|------------|---------------|
| Orin (VPI 3, PVA v2)   | pitch-linear | ✅ | ✅ | ✅ |
| Orin                   | block-linear | ✅ | ❌ `Y8_BL16` | ❌ `Y8_BL16` |
| Xavier (VPI 2, PVA v1) | pitch-linear | ✅ | ✅ | ❌ `NOT_IMPLEMENTED` |
| Xavier                 | block-linear | ✅ | ❌ | ❌ |

Three findings: (1) **VPI wraps an external `NvBufSurface` zero-copy on both
chips** — the Phase-0 open question is answered YES. (2) But the PVA *algorithm*
support is **version/chip-uneven**: `vpiSubmitErode` on PVA runs on Orin (VPI 3)
yet returns `VPI_ERROR_NOT_IMPLEMENTED` on Xavier (VPI 2) — so this op fails the
"both Xavier and Orin" bar. (3) **Block-linear** GRAY8 (the layout `nvvidconv`
emits for NVMM) maps to `VPI_IMAGE_FORMAT_Y8_BL16`, which the algo rejects on
*every* backend — only **pitch-linear** works, so a VPI element fed by a normal
NVMM pipeline would need a VIC **de-tile copy** first, defeating the zero-copy
value proposition.

**Implication:** B4 as a *both-chips, zero-copy* element is **NO-GO** with the
current VPI/PVA stack. The options considered were: (a) an **Orin-only** PVA/VPI
element (like B3's gating), accepting **pitch-linear** input and documenting the
de-tile for block-linear sources; (b) drop PVA and expose these ops on the
already-working **CUDA/VIC** backends (no new differentiator); (c) park B4 until
a use case justifies the Orin-only + pitch-linear constraints.

**Decision (2026-06-09): PARK (option c).** There is **no puller** — no consumer
pipeline demands a PVA vision op today, and a single-host vision op does not
advance the project's differentiator (cross-process zero-copy NVMM IPC). Option
(a) is strictly worse than B3: Orin-only **and** pitch-linear, versus B3's
Orin-only but block-linear-native. Option (b) carries no differentiator for the
trivially-available ops.

**Re-entry trigger (narrow):** re-open B4-PVA only if (1) a *named* Orin-only
consumer needs a specific PVA op, **and** (2) a measurement shows PVA beats the
CUDA/VIC backend for that op. Default expectation: stays parked — CUDA/VIC
usually wins on availability (both chips, no version-uneven `NOT_IMPLEMENTED`).

**Carve-out — `nvmmremap` (dormant candidate, not a roadmap item):** of the
candidate op list, **remap / lens-undistort** is the one with no stock no-DS
NVMM element (`nvvidconv` does scale/crop/convert/flip, not arbitrary remap). If
lens-undistort demand ever appears, the clean build is a **CUDA-backed
`nvmmremap`** — both chips, NV12 zero-copy, **no PVA dependency** — which fits
the no-DeepStream thesis far better than Orin-only PVA morphology. Recorded
separately from B4 because its justification differs; no puller yet, so dormant.

### B5. `nvmminfer` — DLA/GPU inference via TensorRT (both chips)
Run a TensorRT engine on a DLA core (fallback GPU) directly on NVMM input, with
the VIC `NORMALIZE` preprocessing (Part A.4) folded in. Output tensors/meta on the
GStreamer bus. The most ambitious; depends on Part A.4 + a tensor-meta design.

**Phase-0 verdict (2026-06-07): VIABLE, deferred (largest scope).** Orin ships
**TensorRT 10.3** (`NvInfer.h` + `libnvinfer-dev`). TensorRT binds raw CUDA device
pointers, and an NvBufSurface exposes a device pointer per surface, so input
binding without a copy is achievable. But this is the largest item (tensor-meta
design + DLA engine build + Part A.4 NORMALIZE) — **explicitly surface to the user
before committing the multi-day build.**

### B6. `nvmmenc` — NVENC (**Xavier only; document Orin gap**) — ✅ PASS (reuse stock), measured 2026-06-07
NVMM-native H.264/H.265 encode. **Measured, dual-host:** stock `nvv4l2h264enc`
advertises `video/x-raw(memory:NVMM)` sink caps and encodes straight from an NVMM
buffer (`… ! 'video/x-raw(memory:NVMM),NV12' ! nvv4l2h264enc ! h264parse ! filesink`,
rc=0, valid bitstream) on both Xavier and Orin — no system-memory bounce. No
zero-copy gap → **PASS**; recommend the stock `nvv4l2{h264,h265}enc`. (V4L2 encode
is also present on Orin NX here, so the original "Xavier-only" caveat is moot for
the reuse recommendation.)

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
SoC capability. **B3 `nvmmofa` done (v1.3.0, Orin-only)**; B4 PVA is NO-GO as
specified (see B4 verdict).

**Phase 4 — Inference (B5) and NVENC (B6).** Largest scope; B6 only where NVENC
exists.

Each element follows the established pattern: C-ABI GStreamer element, mock build
for CI (extend `nvbufsurface_mock.h` with the new API stubs), real verification on
the Jetson, README + tests, then a version bump + tag.

## Open questions for Phase 0
- ~~Does VPI wrap an externally-allocated `NvBufSurface` zero-copy, or require its
  own `VPIImage` allocation (a copy)? Determines B3/B4 viability.~~ **Answered
  2026-06-07:** yes — VPI 3 (installed on Orin) exposes `VPI_IMAGE_BUFFER_NVBUFFER`
  ("on Tegra platforms only"), the zero-copy NvBuffer/NvBufSurface wrap path.
  B3/B4 viable.
- ~~TensorRT DLA: input binding from an `NvBufSurface` device pointer without a
  copy?~~ **Likely yes** — TensorRT 10.3 (installed on Orin) binds raw CUDA device
  pointers and NvBufSurface exposes a per-surface device pointer; full reproducer
  deferred with the B5 build.
- ~~Mock strategy for VPI in CI (stub vs. skip-on-host).~~ **Settled by B3
  precedent:** VPI engines ship with **no CI unit test** — VPI/SoC-gated, tested
  on-device as the documented exception. Remaining open only for **B5**:
  TensorRT CI-mock strategy (stub vs. skip-on-host).

## Status summary (2026-06-07)
| Item | Verdict | Evidence |
|------|---------|----------|
| Part A (nvmmconvert) | ✅ DONE | interpolation, dst-crop, flip/rotate, compute-mode shipped |
| B1 nvmmjpegenc/dec | ✅ PASS (reuse stock) | stock `nvjpegenc`/`nvjpegdec` do NVMM zero-copy, measured both hosts |
| B2 nvmmcompositor | ✅ DONE (v1.2.0) | element shipped + dual-host validated (PR #25) |
| B6 nvmmenc | ✅ PASS (reuse stock) | stock `nvv4l2h264enc` encodes from NVMM, measured both hosts |
| B3 nvmmofa (OFA) | ✅ DONE (v1.3.0, Orin-only) | element + flow-meta + nvmmflowstats shipped; OFA needs block-linear (nvvidconv default); validated on Orin (~46 fps 720p) |
| B4 nvmmcv (PVA) | ⚫ PARKED (no puller, 2026-06-09) | measured: PVA erode Orin-only (Xavier VPI2 `NOT_IMPLEMENTED`) + block-linear rejected → no clean dual-host zero-copy. Re-enter only if Orin-only consumer + PVA-beats-CUDA measurement |
| `nvmmremap` (remap/undistort) | ⚪ DORMANT candidate | no stock no-DS NVMM remap exists; would build CUDA-backed (both chips, NV12 zero-copy, no PVA) if lens-undistort demand appears |
| B5 nvmminfer (TRT) | 🟡 VIABLE, deferred | TensorRT 10.3 present; largest scope — needs user go-ahead |

Phases 1–3 (VIC + NVJPG/NVENC reuse + OFA) are done; the "reuse stock" items
(B1/B6) are measured-closed. The VPI probes (`vpi_pva_probe.cpp`,
`vpi_ofa_probe.cpp`, run on-device) confirmed zero-copy `NvBufSurface` wrapping
works on both chips, but the two VPI engines diverge sharply on *layout*: **OFA
requires block-linear** (the `nvvidconv` default → B3 shipped, Orin-only) while
**PVA morphology requires pitch-linear** and is unevenly implemented across
VPI 2/3 (→ B4 **NO-GO as specified**, would need re-scope to Orin-only +
pitch-linear or fall back to CUDA/VIC). B5 (TensorRT inference) remains the open
large item, pending user go-ahead.
