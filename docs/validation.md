# Jetson hardware validation

Validated on two Jetson platforms (both in Docker and native):

- **Jetson Xavier NX** — JetPack 5.1 (L4T R35.x), GStreamer 1.16.3
- **Jetson Orin NX** — JetPack 6 (L4T R36.x), GStreamer 1.20.3

## Test results

All 11 test suites pass on both Xavier NX and Orin NX (65 assertions + a fuzz run):

```
  1/11 nvmm_buffer         OK   10 passed   (create, map, move, release, export_fd, planes)
  2/11 nvmm_transform      OK   10 passed   (scale, crop, convert, flip, rotate 90/270, interpolation, compute-mode, null safety)
  3/11 gst_nvmm_allocator  OK    9 passed   (create, alloc, surface map, per-plane, roundtrip, pool video-meta strides)
  4/11 fuzz_shm_header     OK              (200k random NvmmShmHeader inputs through the consumer's validation — no crash/OOB/UB)
  5/11 optical_flow_meta   OK    4 passed   (api type, add/get, S10.5 decode, copy transform)
  6/11 nvmm_compositor     OK    4 passed   (create, output props, request pads, pad placement props)
  7/11 nvmm_sink           OK    5 passed   (create, properties, pool-size guard, state, shm lifecycle)
  8/11 nvmm_appsrc         OK    2 passed   (create, properties)
  9/11 gstcheck_elements   OK    8 passed   (discovery, state, properties, caps, pipeline)
 10/11 nvmm_det_meta       OK    7 passed   (wire layout/segment size, slot pointer math, add/get roundtrip, empty + count-clamped objects, survives buffer copy)
 11/11 integration         OK    6 passed   (multi-shm, dynamic props, pipeline bin, alloc stress, protocol, missing-shm)
Ok: 11   Fail: 0
```

> The `nvmm_det_meta` suite is DeepStream-free POD + `GstMeta` (no `NvBufSurface`),
> so it is hardware-agnostic; re-confirmed 7/7 on Orin NX (JP6) and the x86 dev
> image. Note: inside the Jetson build container the NVMM-allocating suites fail
> with `NvRmMemInitNvmap … Memory Manager Not supported` (no `/dev/nvmap` in the
> container) — build in the container, run the NVMM suites on the host.

Run the suite under sanitizers with `./scripts/run-sanitizers.sh` (ASan+UBSan,
and TSan on a privileged container / bare host).

11 pipeline tests also pass via `scripts/jetson-test.sh`:
passthrough, flip-180, rotate-90, rotate-270, scale, crop, format-convert,
decoder, tee-2way, 30f-throughput, and the two-process IPC pipeline
(`nvmmsink` → `nvmmappsrc`, verified frames cross the process boundary).

The compositor logic (`aggregate()` → `NvBufSurfTransform` `CROP_DST`) needs real
NVMM buffers to run, so it is covered on-hardware rather than in the mock unit
test: a 2-input composite was run on **both** hosts (`videotestsrc` smpte + ball
→ `nvmmcompositor` → JPEG, rc=0). On Orin a side-by-side layout was visually
confirmed — smpte in the left tile (`sink_0`, xpos=0), ball in the right
(`sink_1`, xpos=640); on Xavier the default-fill composite was visually confirmed
(VIC-scaled input). These two runs are the +2 on-hardware pipeline tests over the
jetson-test.sh set.

## Stress tests

| Test | Result |
|------|--------|
| State changes x100 (NULL→READY→NULL) | PASS |
| 500f pool stress (1080p→720p, flip) | PASS (21s) |
| 50 rapid pool recreate cycles | PASS |
| tee x3 with different transforms | PASS |
| Caps renegotiation (4 resolution changes) | PASS |

## Throughput / sustained-load (nvmmconvert, compute-mode)

1000 frames 1080p→480p through `nvmmconvert` in one run (sustained-load stress +
per-engine throughput), `fakesink sync=false`. All runs completed with rc=0 — no
crash, stall, or leak over the run.

| Host | compute-mode | 1000 frames | Throughput |
|------|--------------|-------------|------------|
| Xavier NX (JP5.1.2) | vic | 26.4 s | ~38 fps |
| Xavier NX (JP5.1.2) | gpu | 20.2 s | ~50 fps |
| Orin NX (JP6) | vic | 18.0 s | ~56 fps |
| Orin NX (JP6) | gpu | 14.4 s | ~69 fps |

The GPU engine is faster than the VIC for this downscale on both platforms; pick
`compute-mode=vic` instead to keep the GPU free for other work. Orin is ~1.4×
Xavier across the board.

## Throughput / sustained-load (nvmmcompositor)

600 frames of a 2-input composite (two 960×1080 inputs → one 1920×1080 NVMM
frame, VIC `CROP_DST` per pad) in one run, `fakesink sync=false`. Both runs
completed rc=0 — no crash, stall, or leak.

| Host | inputs → output | 600 frames | Throughput |
|------|-----------------|------------|------------|
| Xavier NX (JP5.1.2, Docker) | 2×960×1080 → 1920×1080 | 8.8 s | ~68 fps |
| Orin NX (JP6, native) | 2×960×1080 → 1920×1080 | 3.7 s | ~164 fps |

Two VIC transforms per output frame; Orin sustains real-time 1080p compositing of
two streams with headroom to spare.

## Optical flow (nvmmofa, OFA — Orin only)

`nvmmofa` runs VPI dense optical flow on the Orin **OFA** engine from zero-copy
NVMM, attaching the motion-vector field as `NvmmOpticalFlowMeta`. Xavier has no
OFA hardware (documented N/A, like NVENC). Validated end-to-end on Orin NX (JP6):

- **Produce → consume:** `videotestsrc → nvvidconv → NVMM NV12 → nvmmofa →
  nvmmflowstats`. The consumer read a `160×120` grid-4 flow field per frame; the
  first frame has no predecessor and correctly carries no flow (19/20 frames with
  flow). `grid-size=1` produces a dense per-pixel `640×480` field.
- **Responds to scene motion** — mean field magnitude over the same pipeline with
  three patterns (15 frames each):

    | pattern | motion | avg mean magnitude |
    |---|---|---|
    | `smpte` | ~static (small animated patch) | 4.59 px |
    | `ball`  | a translating ball | 5.12 px |
    | `snow`  | full-frame random noise | 6.93 px |

    Magnitude rises monotonically with scene motion, confirming the element runs
    OFA on the right frames and the field tracks real content. The non-zero
    *static* baseline (4.59 px) is the dense-flow artifact expected of OFA in
    textureless/low-texture regions (aperture problem) — i.e. this validates that
    a flow field is produced and is motion-responsive, **not** that every cell is
    an accurate vector. Treat the field as approximate; threshold/smooth for
    analytics.
- **Throughput:** 300 frames 720p, `grid-size=4`, sustained **~46 fps**, rc=0.

There is no mock/CI unit test for `nvmmofa`: it is VPI-gated and OFA is Orin-only,
so it is exercised on-device (the documented exception, like the NVENC Xavier
gap). The OFA `(format, backend)` gate is recorded by `probes/vpi_ofa_probe.cpp`.

## Detection metadata side-channel (IPC)

The optional [metadata side-channel](metadata-ipc.md) carries flat detection
records (`NvmmFrameMeta`) alongside each frame so a consumer can recover them
without re-running inference. Validated on Orin NX (JP6, L4T R36.4.3):

- **Wire format + `GstMeta` (unit, host):** `nvmm_det_meta` 7/7 — segment-size
  math with/without the metadata region, per-slot pointer arithmetic, add/get
  round-trip, empty and count-clamped object lists, and survival across
  `gst_buffer_copy` (copy-transform only; non-copy transforms drop the meta).
- **Cross-process surface path (E2E, host):** the protocol-v3 bump does not
  regress zero-copy IPC — a two-process `videotestsrc → nvvidconv → NVMM NV12 →
  nvmmsink` / `nvmmappsrc → nvvidconv → fakesink` run delivered **20/20** frames
  across the boundary, with and without `export-metadata`/`import-metadata`.
- **Properties + graceful degradation:** `nvmmsink export-metadata` and
  `nvmmappsrc import-metadata` are exposed; on a build **without**
  `-Denable_deepstream_meta`, `export-metadata=true` is a documented no-op
  (warns once, `meta_active=FALSE`) and frames still flow — confirmed.

!!! warning "Not yet exercised on hardware: the DeepStream→flat extraction"
    The producer-side `NvDsBatchMeta → NvmmFrameMeta` serialization is gated
    behind `-Denable_deepstream_meta` and needs DeepStream (`nvdsmeta.h`,
    `nvinfer`). The validation Orin has no DeepStream installed, so the full
    `nvinfer → nvmmsink(export) → nvmmappsrc(import) → GstNvmmDetMeta` path with
    **real detections crossing** has not been run end-to-end on hardware. Until a
    DeepStream-equipped box is available, that extraction step is covered only by
    the unit tests above. Everything downstream of the wire record (consumer
    attach, transform, graceful no-op) is validated.

## Sanitizer results

Run with `./scripts/run-sanitizers.sh` (mock build in the dev container).

| Sanitizer | Suites | Result |
|-----------|--------|--------|
| ASan + UBSan | all 9 (`libasan` preloaded) | Clean |
| ThreadSanitizer | 4 core (buffer, transform, allocator, fuzz) | Clean |

The element tests (`nvmm_compositor`, `nvmm_sink`, `gstcheck_elements`,
`integration`) load plugins via `dlopen`, which trips ASan's *"runtime does not
come first"* check unless `libasan` is `LD_PRELOAD`ed — the runner does this, so
all nine suites pass clean under ASan+UBSan; the loader leak is GStreamer's, not
this code. Under **TSan** those same dlopen tests can't run (the unsanitized
plugin scanner can't load a sanitized `.so`), so the `plugin` suite is excluded
and the atomic-heavy IPC paths are covered by the core + fuzz suites; TSan needs
`setarch -R` (a privileged container or bare host) to disable ASLR.

## Benchmark results

1000 iterations each. VIC transform includes hardware sync.

**Xavier NX (JetPack 5.1)**

| Operation | Resolution | Avg (us) | Min (us) | Max (us) |
|-----------|-----------|----------|----------|----------|
| alloc/free | NV12 1080p | 591 | 128 | 2291 |
| alloc/free | RGBA 1080p | 2095 | 1072 | 2129 |
| alloc/free | NV12 4K | 3245 | 2091 | 2701 |
| alloc/free | RGBA 4K | 10104 | 7110 | 9164 |
| map/unmap | NV12 1080p | 231 | 222 | 493 |
| VIC transform | 1080p -> 480p | **1947** | 1577 | 4752 |
| VIC transform | 1080p -> 720p | **1655** | 1594 | 1826 |
| VIC transform | 4K -> 1080p | **4002** | 3938 | 4913 |

**Orin NX (JetPack 6)**

| Operation | Resolution | Avg (us) | Min (us) | Max (us) |
|-----------|-----------|----------|----------|----------|
| alloc/free | NV12 1080p | 117 | 14 | 1551 |
| alloc/free | RGBA 1080p | 366 | 33 | 1072 |
| map/unmap | NV12 1080p | 298 | 275 | 374 |
| map/unmap | NV12 480p | 49 | 39 | 61 |
| VIC transform | 1080p -> 480p | **35** | 27 | 49 |
| VIC transform | 1080p -> 720p | **95** | 85 | 114 |
| VIC transform | 4K -> 1080p | **285** | 217 | 459 |
| VIC transform | 4K -> 480p | **31** | 26 | 67 |

Orin allocation is **5x faster** than Xavier NX. VIC transform **14-56x faster**
depending on resolution (e.g. 1080p->480p: 1947 us -> 35 us).

## VIC hardware accelerator verification

Evidence that the Tegra VIC (Video Image Compositor) hardware engine is engaged:

1. **NvBufSurfTransform defaults to VIC compute on Jetson** — the API selects `NvBufSurfTransformCompute_Default`, which maps to VIC on Tegra (not GPU or CPU).
2. **Transform latency confirms hardware acceleration** — 35 us per 1080p-to-480p scale on Orin NX (table above). A CPU scale at 1080p would take milliseconds; ~28,500 FPS is only achievable via dedicated hardware.
3. **NVMM SURFACE_ARRAY memory type** — tests use `NVBUF_MEM_DEFAULT` → `NVBUF_MEM_SURFACE_ARRAY` on Jetson (physically contiguous, VIC/NVDEC-managed). Tests FAIL with `NVBUF_MEM_SYSTEM` for hardware-only operations, proving the hardware path.
4. **DMA-buf fd export works** — `export_fd()` returns a valid fd from `bufferDesc`.
5. **VIC device node** — `/dev/nvhost-vic` is present and accessible.

## Transfer path verification

| Path | Pipeline | Result |
|------|----------|--------|
| **CPU -> GPU** | `videotestsrc ! nvvidconv ! NVMM ! nvmmsink` | OK |
| **GPU -> GPU** | `nvv4l2decoder(NVMM) ! nvvidconv ! NVMM(scaled) ! nvmmsink` | OK |
| **GPU -> CPU** | `nvv4l2decoder(NVMM) ! nvvidconv ! x-raw ! jpegenc ! file` | OK |

## Resolution verification

| Resolution | Alloc | Map | Transform (to 480p) | Pipeline |
|------------|-------|-----|---------------------|----------|
| **FHD** 1920x1080 | 3103 us | 77 us | 5324 us | OK (133 KB JPEG) |
| **4K** 3840x2160 | 263 us | 105 us | 17028 us | OK (491 KB JPEG) |

NvmmBuffer API at both resolutions: NV12 = 2 planes (Y+UV); FHD data_size
3,407,872 B; 4K data_size 12,582,912 B; DMA-buf fd export works at both.

## nvmmconvert pipeline proof

All operations verified via `gst-launch-1.0` on Jetson Xavier NX:

| Operation | Output |
|-----------|--------|
| Passthrough | ![passthrough](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_passthrough.jpg) |
| Flip 180° | ![flip180](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_flip180.jpg) |
| Rotate 90° CW (640×480→480×640) | ![rotate90](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_rotate90.jpg) |
| Rotate 270° CCW (640×480→480×640) | ![rotate270](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_rotate270.jpg) |
| Flip horizontal | ![flipH](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_flipH.jpg) |
| Scale 1080p→480p | ![scale](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_scale.jpg) |
| Crop (100,50,800,600) | ![crop](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/convert_crop.jpg) |

## Test outputs

All images generated on Jetson Xavier NX with real NVMM hardware:

| Image | Description |
|-------|-------------|
| ![smpte](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/smpte_1080p.jpg) | **smpte_1080p.jpg** — 1920x1080 SMPTE test pattern |
| ![gpu2cpu](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/gpu2cpu_1080p.jpg) | **gpu2cpu_1080p.jpg** — 1080p GPU→CPU transfer |
| ![4k](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/4k_roundtrip.jpg) | **4k_roundtrip.jpg** — 3840x2160 CPU→NVMM→CPU |
| ![ipc480](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/ipc_480p.jpg) | **ipc_480p.jpg** — IPC consumer via nvmmsink→shm→nvmmappsrc |
| ![shm](https://raw.githubusercontent.com/PavelGuzenfeld/gst-nvmm-cpp/main/test_output/shm_consumer_frame.jpg) | **shm_consumer_frame.jpg** — standalone C shm reader (ROS2-style) |

## Reproducing on Jetson

```bash
git clone https://github.com/PavelGuzenfeld/gst-nvmm-cpp.git
cd gst-nvmm-cpp

# Build (CMake; meson equivalent in Getting started)
cmake -S . -B build-cmake
cmake --build build-cmake -j$(nproc)

# Test + benchmark
ctest --test-dir build-cmake --output-on-failure
./build-cmake/benchmarks/bench_nvmm

# Install + use
sudo cmake --install build-cmake
rm -f ~/.cache/gstreamer-1.0/registry.*.bin   # force a GStreamer rescan
gst-inspect-1.0 nvmmconvert
```
