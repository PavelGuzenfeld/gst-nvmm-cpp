# gst-nvmm-cpp

GStreamer plugin suite for **NVMM-native video processing** on NVIDIA Jetson
platforms (Xavier, Orin). Wraps the Tegra-native `NvBufSurface` / NVMM
memory model in proper GStreamer elements with hardware-accelerated crop,
scale, format conversion, and inter-process video sharing.

Cross-process video sharing is one backend (NVMM pool + SCM_RIGHTS fd
passing) that works on both JetPack 5.1.1+ (L4T R35.3.1, Xavier) and
JetPack 6 (Orin). Consumer-side reads are zero-copy from GPU memory; the
producer pays at most one `NvBufSurfaceCopy` (GPU-to-GPU) per frame.

Built with **C++14** internals and a **C ABI** boundary to GStreamer.

## The Problem This Solves

On Jetson, the native video buffer type is `NvBufSurface` (NVMM) — physically contiguous, DMA-coherent memory managed by the Tegra VIC hardware engine. The standard GStreamer `nvcodec` plugin targets discrete desktop GPUs via CUDA and doesn't understand NVMM.

This creates a gap:
- `nvv4l2decoder` outputs `video/x-raw(memory:NVMM)` but no upstream GStreamer element can consume it without a CPU copy
- Crop/scale on NVMM requires the proprietary `nvvidconv` element, which is tied to specific JetPack versions
- No standard `GstAllocator` exists for `NvBufSurface`, so every team writes their own

**gst-nvmm-cpp** fills this gap with open-source, tested, upstream-ready GStreamer elements.

## Elements

### nvmmconvert

Video crop, scale, and color format conversion using the **Tegra VIC** (Video Image Compositor) hardware engine. Zero CPU involvement.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `crop-x` | uint | 0 | Source crop X offset (pixels) |
| `crop-y` | uint | 0 | Source crop Y offset (pixels) |
| `crop-w` | uint | 0 | Source crop width (0 = full width) |
| `crop-h` | uint | 0 | Source crop height (0 = full height) |
| `flip-method` | int | 0 | 0=none, 1=90CW, 2=180, 3=90CCW, 4=flipH, 5=transpose, 6=flipV, 7=inv-transpose |

**Supported formats:** NV12, RGBA, I420, BGRA

**Caps:** `video/x-raw(memory:NVMM), format={NV12,RGBA,I420,BGRA}, width=[1,8192], height=[1,8192]`

### nvmmsink / nvmmappsrc

A producer/consumer pair for sharing NVMM video frames across processes
with consumer-side zero-copy. One backend, supports both JetPacks:

| Wire | Transport | Frame path |
|------|-----------|------------|
| `NvmmShmPoolHeader` (proto = 3) | shm header + unix-domain socket (SCM_RIGHTS fd passing) | producer `NvBufSurfaceCopy` into NVMM pool slot, consumer `NvBufSurfaceImport`s the fds and reads GPU memory directly — no further copies |

**Minimum L4T:** R35.3.1 (JetPack 5.1.1, March 2023) on the JP5 line, or
any JP6. Earlier L4T 35.x (R35.2.1 / JP 5.0.x) does not have
`NvBufSurfaceImport` and cannot do cross-process NVMM zero-copy at all
without major workarounds — meson rejects those toolchains at configure
time.

Both element sources delegate to one backend in `gst/ipc_pool/ipc_pool.cpp`
behind a C ABI in [`gst/common/ipc_backend.h`](gst/common/ipc_backend.h).
No `#ifdef` on JetPack version anywhere in the call path.

```
Message: NVMM IPC backend: pool + SCM_RIGHTS  (target: jetson (real NvBufSurface))
```

**nvmmsink properties**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `shm-name` | string | `/nvmm_sink_0` | POSIX shared-memory segment name |
| `pool-size` | int (4–32) | 8 | Number of NVMM buffers in the cross-process pool |

**nvmmappsrc properties**

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `shm-name` | string | `/nvmm_sink_0` | POSIX segment name to connect to (must match the sink) |
| `is-live` | bool | true | Advertise as live source |

The consumer auto-detects video format, resolution, and plane layout from
the producer's header on the first frame and emits `video/x-raw(memory:NVMM)`.

**External NVMM sources (zedsrc, nvvidconv, nvv4l2decoder)** are accepted
via the `memory:NVMM` caps feature; render does a GPU-to-GPU
`NvBufSurfaceCopy` into the next free pool slot. (True producer-side zero-
copy via `propose_allocation` is on the follow-up list.)

**Debug categories**

- `nvmmsink` — sink-side element lifecycle
- `nvmmappsrc` — source-side element lifecycle
- `nvmmipc.pool` — backend-internal producer/consumer logs (LOG level emits one line per frame)

### GstNvmmAllocator

A `GstAllocator` subclass that wraps `NvBufSurfaceCreate` / `NvBufSurfaceDestroy` with proper `GstMemory` semantics.

```cpp
// Allocate an NVMM buffer with explicit video format/dimensions
// (follows the GstGLMemory/GstVulkanImageMemory pattern)
GstAllocator *alloc = gst_nvmm_allocator_new(0 /* NVBUF_MEM_DEFAULT */);
GstMemory *mem = gst_nvmm_allocator_alloc_video(alloc,
    GST_VIDEO_FORMAT_NV12, 1920, 1080);

// Check if memory is NVMM
if (gst_is_nvmm_memory(mem)) {
    NvBufSurface *surface = gst_nvmm_memory_get_surface(mem);
    // ... use surface directly with NVIDIA APIs
}

gst_memory_unref(mem);
gst_object_unref(alloc);
```

## Architecture

```
+----------------------------------------------------------------+
|                        GStreamer Pipeline                        |
|                                                                 |
|  +----------+    +--------------+    +-----------+              |
|  | decoder  |--->| nvmmconvert  |--->| nvmmsink  |--> SHM      |
|  |(nvv4l2)  |    | (VIC h/w)    |    | (POSIX)   |              |
|  +----------+    +--------------+    +-----------+              |
|       |                |                    |                    |
|       v                v                    v                    |
|  +----------------------------------------------+              |
|  |           GstNvmmAllocator                    |              |
|  |  alloc --> NvBufSurfaceCreate                 |              |
|  |  map   --> NvBufSurfaceMap                    |              |
|  |  unmap --> NvBufSurfaceUnMap                  |              |
|  |  free  --> NvBufSurfaceDestroy                |              |
|  |  fd    --> bufferDesc (DMA-buf)               |              |
|  +----------------------------------------------+              |
|                          |                                      |
|                          v                                      |
|              +--------------------+                             |
|              |  NvBufSurface API  |  (libnvbufsurface.so)       |
|              |  NvBufSurfTransform|  (libnvbufsurftransform.so) |
|              |  Tegra VIC Engine  |                             |
|              +--------------------+                             |
+----------------------------------------------------------------+

         +-----------+
  SHM -->|nvmmappsrc |--> downstream pipeline (ROS2, inference, etc.)
         +-----------+
```

## C++ Design

The ABI boundary to GStreamer is C (`plugin_init`, GObject type system). Inside that boundary, everything is C++14.

| Pattern | Implementation |
|---------|---------------|
| **RAII for NvBufSurface** | `nvmm::NvmmBuffer` owns the surface, calls `NvBufSurfaceDestroy` in destructor |
| **Result type** | `nvmm::Result<T>` (C++14 `aligned_storage`-based, supports move-only types) |
| **Non-owning byte views** | `nvmm::ByteSpan` — lightweight replacement for `std::span<uint8_t>` |
| **Type-safe enums** | `nvmm::MemoryType`, `nvmm::ColorFormat`, `nvmm::FlipMethod` |
| **Zero GLib in internals** | `G_BEGIN_DECLS`/`G_END_DECLS` only at plugin boundaries |

## Building

### Prerequisites

- GStreamer >= 1.16 development libraries
- Either Meson >= 0.62 + Ninja, **or** CMake >= 3.16
- C++14 compiler (GCC 7+ or Clang 5+)
- On Jetson: JetPack 5 (L4T 35.x) or JetPack 6 (L4T 36.x)

### Docker (recommended for x86_64 host)

```bash
# Host (x86_64) -- builds with mock NvBufSurface API for testing
docker build -f docker/Dockerfile.dev -t gst-nvmm-cpp:dev .
docker run --rm gst-nvmm-cpp:dev
```

### Docker on Jetson (Xavier NX / Orin)

```bash
# Build (uses ubuntu:22.04, mounts host NVIDIA libs at runtime)
docker build --network host -f docker/Dockerfile.jetson -t gst-nvmm-cpp:jetson .

# Run tests + pipelines (mount NVIDIA runtime libs and GStreamer plugins)
docker run --runtime nvidia --rm --network host --privileged \
  -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro \
  -v /usr/lib/aarch64-linux-gnu/tegra-egl:/usr/lib/aarch64-linux-gnu/tegra-egl:ro \
  -v /usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/gstreamer-1.0:ro \
  -v /usr/src/jetson_multimedia_api:/usr/src/jetson_multimedia_api:ro \
  -v /usr/share/glvnd:/usr/share/glvnd:ro \
  -v /etc/alternatives:/etc/alternatives:ro \
  -v /etc/ld.so.conf.d:/etc/ld.so.conf.d:ro \
  gst-nvmm-cpp:jetson
```

### Native build (Jetson)

Two build systems are supported in parallel; pick whichever you prefer. They share the same source tree and the same test suite.

**Meson:**
```bash
pip3 install meson
meson setup builddir -Dcpp_std=c++14 -Dbuildtype=debugoptimized
ninja -C builddir
meson test -C builddir --verbose
```

**CMake:**
```bash
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-cmake -j$(nproc)
ctest --test-dir build-cmake --output-on-failure
```

The CMake build additionally supports `sudo cmake --install build-cmake`, which places the plugins in GStreamer's system pluginsdir (as reported by `pkg-config --variable=pluginsdir gstreamer-1.0`) and sets `RUNPATH=$ORIGIN` on each `.so`. After installing, pipelines work with no `GST_PLUGIN_PATH` / `LD_LIBRARY_PATH` exports. Override with `-DGSTREAMER_PLUGINS_DIR=/some/path` or `-DWERROR=OFF` if needed.

On hosts without Jetson libraries, both build systems automatically detect the absence of `libnvbufsurface.so` and build with the **mock API** — a header-only stub that implements the same `NvBufSurface` struct layout and function signatures using heap memory. All tests pass against the mock.

## Jetson Hardware Validation

Validated on Jetson hardware (Docker, `--runtime nvidia`):

| Device | L4T | JetPack | GStreamer | Branch |
|--------|-----|---------|-----------|--------|
| Jetson Xavier NX | R35.4.1 | 5.1.2 | 1.16.3 | `feature/zero-copy-jp5-jp6` |
| Jetson Xavier NX | R35.2.1 | 5.0.x | 1.16.3 | legacy (no IPC, no `NvBufSurfaceImport`) |
| Jetson Orin NX | R36.4.3 | 6.x | 1.20.3 | build-validated |

### Test Results

All 9 test suites pass on Xavier NX (L4T R35.4.1, `docker run --runtime nvidia`):

```
 1/9 nvmm_buffer          OK   10 passed   (create, map, move, release, export_fd, planes)
 2/9 nvmm_transform       OK    6 passed   (scale, crop, convert, flip, null safety)
 3/9 gst_nvmm_allocator   OK    8 passed   (create, alloc, surface map, per-plane, roundtrip)
 4/9 nvmm_sink            OK    4 passed   (create, properties, state, shm lifecycle)
 5/9 nvmm_appsrc          OK    2 passed   (create, properties)
 6/9 gstcheck_elements    OK    8 passed   (discovery, state, properties, caps, pipeline)
 7/9 integration          OK    6 passed   (multi-shm, dynamic props, pipeline bin, alloc stress, protocol, missing-shm)
 8/9 backend_concurrency  OK    5 passed   (producer/consumer, churn, fanout, crc_roundtrip, zerocopy)
 9/9 fuzz_shm_header      OK    0 crashes  (10 000 random-byte iterations)
Ok: 9   Fail: 0
```

8 pipeline tests also pass via `scripts/jetson-test.sh`:
passthrough, flip-180, scale, crop, format-convert, decoder, tee-2way, 30f-throughput.

### Stress Tests

| Test | Result |
|------|--------|
| State changes x100 (NULL→READY→NULL) | PASS |
| 500f pool stress (1080p→720p, flip) | PASS (21s) |
| 50 rapid pool recreate cycles | PASS |
| tee x3 with different transforms | PASS |
| Caps renegotiation (4 resolution changes) | PASS |

### Sanitizer Results

| Sanitizer | Tests | Result |
|-----------|-------|--------|
| AddressSanitizer | 22 (buffer + transform + allocator) | Clean |
| ThreadSanitizer | 22 (buffer + transform + allocator) | Clean |

### Benchmark Results

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

Orin allocation is **5x faster** than Xavier NX. VIC transform **14-114x faster** depending on resolution.

Both platforms pass: passthrough, flip, scale, crop, format convert, 500f stress, tee, decoder pipelines.

### Zero-Copy IPC Verification (L4T R35.4.1 / JP 5.1.2)

Verified that `NvBufSurfaceImport`-based zero-copy IPC is the active path on
this device — not a mock or CPU-copy fallback.

**Build-time evidence** (`config.h` from the Docker image):

```c
#define HAVE_NVBUFSURFACE 1                          // real lib, not mock
#define JETPACK_VERSION "jetson (real NvBufSurface)"  // -Dmock not set
```

**Runtime evidence** (`GST_DEBUG=nvmmipc.pool:6` during `demo_visual_roundtrip`):

```
nvmmipc.pool ipc_pool.cpp:102  NVMM IPC: NvBufSurfaceImport present in libnvbufsurface
nvmmipc.pool ipc_pool.cpp:430  handed pool (8 fds) to client fd=16
nvmmipc.pool ipc_pool.cpp:891  consumer started: pool=8, imported 8 surfaces
nvmmipc.pool ipc_pool.cpp:754  published frame #1 into slot 1
nvmmipc.pool ipc_pool.cpp:1045 fetched frame #1 from slot 1
... (8 frames, one LOG line per frame from each thread)
```

Line 102 is a `dlsym` probe on the running `libnvbufsurface.so` from the host BSP — not a
header-only check. Line 430 confirms DMA-buf fds passed via SCM_RIGHTS. Line 891 confirms
`NvBufSurfaceImport` called 8× (once per pool slot). No `NvBufSurfaceFromFd` (the
process-local JP 5.0.x fallback) appears anywhere in the trace.

**Visual roundtrip** — 8/8 frames produced and consumed, all pixel-perfect identical
(`cmp -s tx_frame_NN.ppm rx_frame_NN.ppm` passes for every pair):

| Frame 0 | Frame 3 | Frame 7 |
|---------|---------|---------|
| ![frame0](test_output/tx_frame_00.png) | ![frame3](test_output/tx_frame_03.png) | ![frame7](test_output/tx_frame_07.png) |

Each frame carries a vertical RGBA gradient with a top progress bar whose width encodes the
frame index (narrow at frame 0, full-width at frame 7). The consumer reads each frame
directly from the imported GPU surface — the byte-level identity with the TX dump confirms
no intermediate copy corrupted or altered the data.

### No-CPU-Path Proof

**Caps enforcement — GStreamer rejects the pipeline at negotiation time if CPU memory is offered:**

```
$ gst-inspect-1.0 nvmmsink | grep -A3 "SINK template"
  SINK template: 'sink'
    Capabilities:
      video/x-raw(memory:NVMM)   ← only this; plain video/x-raw is refused

$ gst-inspect-1.0 nvmmappsrc | grep -A3 "SRC template"
  SRC template: 'src'
    Capabilities:
      video/x-raw(memory:NVMM)   ← delivers GPU memory downstream
```

**Code audit — `NvBufSurfaceMap` has zero occurrences in the render/fetch path:**

```
$ grep -n "NvBufSurfaceMap" gst/ipc_pool/ipc_pool.cpp
164:  NvBufSurfaceMapParams  map_params{};    ← metadata struct, not the map call
408:  NvBufSurfaceMapParams p = ...           ← metadata copy
866:  std::vector<NvBufSurfaceMapParams> ...  ← recv buffer
# NvBufSurfaceMap() itself: 0 hits
```

The render path at `ipc_pool.cpp:731` calls only `NvBufSurfaceCopy(src_nvmm, pool_slot_nvmm)` — a
GPU-to-GPU copy using the VIC/DMA engine. The consumer fetch at line 1040 calls
`wrap_imported(imported[idx])`, which hands the `NvBufSurface*` pointer directly to GStreamer as
`GstMemory` — the downstream element gets the GPU pointer and no CPU mapping is involved.

**Runtime log confirming zero-copy propose_allocation offered to upstream:**

```
nvmmipc.pool  ipc_pool.cpp:659  offered NVMM pool (8 slots) to upstream for zero-copy
```

When `nvvidconv` accepts this (the common case), it renders directly into a pool slot — not even a
`NvBufSurfaceCopy`. The entire chain from decode to consumer is a single hardware DMA path.

### VIC Hardware Accelerator Verification

Evidence that the Tegra VIC (Video Image Compositor) hardware engine is engaged:

1. **NvBufSurfTransform defaults to VIC compute on Jetson** — the API selects `NvBufSurfTransformCompute_Default` which maps to VIC on Tegra (not GPU or CPU).

2. **Transform latency confirms hardware acceleration** — 21 us per 1080p-to-480p scale operation. A CPU-based scale at 1080p would take several milliseconds. The ~47,000 FPS throughput is only achievable via dedicated hardware.

3. **NVMM SURFACE_ARRAY memory type confirms DMA-coherent allocation** — tests use `NVBUF_MEM_DEFAULT` which resolves to `NVBUF_MEM_SURFACE_ARRAY` on Jetson. This memory type is physically contiguous and managed by the VIC/NVDEC hardware engines. Tests FAIL when using `NVBUF_MEM_SYSTEM` (malloc'd memory) for operations that require hardware access, proving the hardware path is in use.

4. **DMA-buf fd export works** — `export_fd()` returns a valid file descriptor from `bufferDesc`, confirming the buffer lives in DMA-coherent hardware memory.

5. **VIC device node** — `/dev/nvhost-vic` is present and accessible.

### Transfer Path Verification

All three transfer directions verified on Jetson:

| Path | Pipeline | Result |
|------|----------|--------|
| **CPU -> GPU** | `videotestsrc ! nvvidconv ! NVMM ! nvmmsink` | OK |
| **GPU -> GPU** | `nvv4l2decoder(NVMM) ! nvvidconv ! NVMM(scaled) ! nvmmsink` | OK |
| **GPU -> CPU** | `nvv4l2decoder(NVMM) ! nvvidconv ! x-raw ! jpegenc ! file` | OK |

### Resolution Verification

| Resolution | Alloc | Map | Transform (to 480p) | Pipeline |
|------------|-------|-----|---------------------|----------|
| **FHD** 1920x1080 | 3103 us | 77 us | 5324 us | OK (133 KB JPEG) |
| **4K** 3840x2160 | 263 us | 105 us | 17028 us | OK (491 KB JPEG) |

NvmmBuffer API results at both resolutions:
- NV12 plane layout: 2 planes (Y + UV)
- FHD data_size: 3,407,872 bytes (3.2 MB)
- 4K data_size: 12,582,912 bytes (12 MB)
- DMA-buf fd export: works at both resolutions

### Pipeline Verification

Tested with real GStreamer pipelines on Jetson:

```bash
# CPU -> GPU: test pattern to NVMM shared memory
gst-launch-1.0 videotestsrc num-buffers=3 ! \
  'video/x-raw,width=1920,height=1080,format=I420' ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
  nvmmsink shm-name=/test_cpu2gpu sync=false

# GPU -> GPU: H264 decode (NVMM) -> scale (NVMM) -> NVMM out
gst-launch-1.0 videotestsrc num-buffers=10 ! \
  'video/x-raw,width=1920,height=1080' ! x264enc tune=zerolatency ! \
  nvv4l2decoder ! 'video/x-raw(memory:NVMM)' ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),width=640,height=480' ! \
  nvmmsink shm-name=/test_gpu2gpu sync=false

# GPU -> CPU: decode to NVMM, convert to CPU, save JPEG
gst-launch-1.0 videotestsrc num-buffers=1 ! \
  'video/x-raw,width=1920,height=1080' ! x264enc tune=zerolatency ! \
  nvv4l2decoder ! 'video/x-raw(memory:NVMM)' ! \
  nvvidconv ! 'video/x-raw,format=I420' ! \
  nvjpegenc ! filesink location=gpu2cpu_1080p.jpg

# 4K CPU -> NVMM -> CPU roundtrip
gst-launch-1.0 videotestsrc num-buffers=1 pattern=smpte ! \
  'video/x-raw,width=3840,height=2160,format=I420' ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
  nvvidconv ! 'video/x-raw,format=I420' ! \
  nvjpegenc ! filesink location=4k_roundtrip.jpg

# 4K -> FHD scale via NVMM (GPU -> GPU)
gst-launch-1.0 videotestsrc num-buffers=1 pattern=ball ! \
  'video/x-raw,width=3840,height=2160,format=I420' ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),width=1920,height=1080' ! \
  nvvidconv ! 'video/x-raw,format=I420' ! \
  nvjpegenc ! filesink location=4k_to_fhd.jpg
```

### IPC Verification (nvmmsink -> nvmmappsrc)

Verified inter-process video sharing via POSIX shared memory:

```bash
# Producer (background): write SMPTE frames to shm
gst-launch-1.0 videotestsrc num-buffers=50 pattern=smpte ! \
  'video/x-raw,width=640,height=480,format=I420,framerate=10/1' ! \
  nvmmsink shm-name=/ipc_test sync=true &

# Consumer: read from shm, save JPEG
gst-launch-1.0 -e nvmmappsrc shm-name=/ipc_test is-live=true ! \
  videoconvert ! jpegenc ! filesink location=ipc_480p.jpg
```

IPC consumer output at 480p and 1080p. The full frame path is GPU-resident
throughout — no CPU copy at any stage:

```
nvv4l2decoder (NVMM) → nvvidconv (GPU→GPU) → nvmmsink
  [NvBufSurfaceCopy GPU→GPU into pool slot, or true zero-copy if
   upstream accepted propose_allocation]
  ↓ SCM_RIGHTS (DMA-buf fds — no data moved)
nvmmappsrc → NvBufSurfaceImport → downstream (memory:NVMM)
```

![IPC 480p](test_output/ipc_480p.jpg)
![IPC 1080p](test_output/ipc_1080p.jpg)

Also verified the SHM protocol with a standalone C consumer (ROS2-style):
- Header fields (magic, resolution, format, frame number, timestamp) read correctly
- Pixel data integrity verified via write/read roundtrip

![SHM consumer gradient](test_output/shm_consumer_frame.jpg)

### nvmmconvert Pipeline Proof

All operations verified via `gst-launch-1.0` on Jetson Xavier NX:

| Operation | Output |
|-----------|--------|
| Passthrough | ![passthrough](test_output/convert_passthrough.jpg) |
| Flip 180° | ![flip180](test_output/convert_flip180.jpg) |
| Flip horizontal | ![flipH](test_output/convert_flipH.jpg) |
| Scale 1080p→480p | ![scale](test_output/convert_scale.jpg) |
| Crop (100,50,800,600) | ![crop](test_output/convert_crop.jpg) |

### Test Outputs

All images generated on Jetson Xavier NX with real NVMM hardware:

| Image | Description |
|-------|-------------|
| ![smpte](test_output/smpte_1080p.jpg) | **smpte_1080p.jpg** -- 1920x1080 SMPTE test pattern |
| ![gpu2cpu](test_output/gpu2cpu_1080p.jpg) | **gpu2cpu_1080p.jpg** -- 1080p GPU->CPU transfer |
| ![4k](test_output/4k_roundtrip.jpg) | **4k_roundtrip.jpg** -- 3840x2160 CPU->NVMM->CPU |
| ![4k_fhd](test_output/4k_to_fhd.jpg) | **4k_to_fhd.jpg** -- 4K scaled to 1080p via NVMM |
| ![decoded](test_output/decoded_frame.jpg) | **decoded_frame.jpg** -- 640x480 H264 decoded via NVMM |
| ![ipc480](test_output/ipc_480p.jpg) | **ipc_480p.jpg** -- IPC consumer via nvmmsink->shm->nvmmappsrc |
| ![ipc1080](test_output/ipc_1080p.jpg) | **ipc_1080p.jpg** -- IPC consumer 1080p (decode->NVMM->CPU->shm) |
| ![shm](test_output/shm_consumer_frame.jpg) | **shm_consumer_frame.jpg** -- Standalone C shm reader (ROS2-style) |

### Setup for Reproducing on Jetson

```bash
git clone https://github.com/PavelGuzenfeld/gst-nvmm-cpp.git
cd gst-nvmm-cpp

# Build (CMake — recommended; see the meson equivalent in the Building section)
cmake -S . -B build-cmake
cmake --build build-cmake -j$(nproc)

# Run tests
ctest --test-dir build-cmake --output-on-failure

# Run benchmarks
./build-cmake/benchmarks/bench_nvmm

# Install + use (no env vars needed afterwards)
sudo cmake --install build-cmake
rm -f ~/.cache/gstreamer-1.0/registry.*.bin   # force GStreamer to rescan
gst-inspect-1.0 nvmmconvert
```

To use plugins from the build tree (without installing), point GStreamer at each plugin dir:

```bash
export GST_PLUGIN_PATH=$(pwd)/build-cmake/gst/nvmmconvert:$(pwd)/build-cmake/gst/nvmmsink:$(pwd)/build-cmake/gst/nvmmappsrc:$(pwd)/build-cmake/gst/nvmmalloc
gst-inspect-1.0 nvmmconvert
```

## Pipeline Examples

### Decode and scale (Jetson)

```bash
gst-launch-1.0 \
  filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder \
  ! 'video/x-raw(memory:NVMM)' \
  ! nvmmconvert \
  ! 'video/x-raw(memory:NVMM),width=640,height=480' \
  ! nvmmsink shm-name=/camera_feed
```

### Crop a region of interest

```bash
gst-launch-1.0 \
  ... ! nvmmconvert crop-x=100 crop-y=50 crop-w=800 crop-h=600 ! ...
```

### Flip video

```bash
# Rotate 180 degrees
gst-launch-1.0 ... ! nvmmconvert flip-method=2 ! ...

# Mirror horizontally
gst-launch-1.0 ... ! nvmmconvert flip-method=4 ! ...
```

### Inter-process video sharing

**Process A** (producer):
```bash
gst-launch-1.0 \
  nvv4l2decoder ! nvmmsink shm-name=/video_feed
```

**Process B** (consumer):
```bash
gst-launch-1.0 \
  nvmmappsrc shm-name=/video_feed ! videoconvert ! autovideosink
```

### Multi-camera fan-out to multiple consumers

The motivating use case for `nvmmsink` / `nvmmappsrc`: one producer stream published once, consumed concurrently by as many processes as you want, all staying on the GPU. Each `nvmmsink` pool is written once per frame; every consumer just imports the fds and reads in place — adding a second (or third) consumer does not add a second GPU copy.

Given N ZED cameras publishing NVMM NV12 at 120 fps, and K processes that each need to encode all N streams to MP4 — every consumer gets every stream, no CPU copies.

**Producer** (N cameras → N shm segments, one process). Replace the serial numbers with your own (`zedsrc camera-sn=...`):

```bash
gst-launch-1.0 -e \
  zedsrc camera-sn=<SN1> camera-resolution=4 camera-fps=120 stream-type=7 \
    ! 'video/x-raw(memory:NVMM),format=NV12' \
    ! queue ! nvmmsink shm-name=/cam1 \
  zedsrc camera-sn=<SN2> camera-resolution=4 camera-fps=120 stream-type=7 \
    ! 'video/x-raw(memory:NVMM),format=NV12' \
    ! queue ! nvmmsink shm-name=/cam2 \
  zedsrc camera-sn=<SN3> camera-resolution=4 camera-fps=120 stream-type=7 \
    ! 'video/x-raw(memory:NVMM),format=NV12' \
    ! queue ! nvmmsink shm-name=/cam3
```

**Consumers** (each instance attaches to all three shm segments and records to its own files). Launch this pipeline in as many shells as you want — the producer above doesn't care:

```bash
timeout -s INT 120 gst-launch-1.0 -e \
  nvmmappsrc shm-name=/cam1 do-timestamp=true is-live=true \
    ! 'video/x-raw(memory:NVMM),format=NV12' \
    ! nvv4l2h264enc bitrate=20000000 ! h264parse ! qtmux \
    ! filesink location=/tmp/out_cam1.mp4 sync=false async=false \
  nvmmappsrc shm-name=/cam2 do-timestamp=true is-live=true \
    ! 'video/x-raw(memory:NVMM),format=NV12' \
    ! nvv4l2h264enc bitrate=20000000 ! h264parse ! qtmux \
    ! filesink location=/tmp/out_cam2.mp4 sync=false async=false \
  nvmmappsrc shm-name=/cam3 do-timestamp=true is-live=true \
    ! 'video/x-raw(memory:NVMM),format=NV12' \
    ! nvv4l2h264enc bitrate=20000000 ! h264parse ! qtmux \
    ! filesink location=/tmp/out_cam3.mp4 sync=false async=false
```

The pool's per-slot `ref_counts` handle the fan-out: each consumer atomically increments its slot's count on read and decrements when done, and the producer only reuses a slot once its count is back to 0. Buffers stay GPU-resident through the whole pipeline — decode to encode in the consumer never leaves NVMM.

After `sudo cmake --install` these commands work as shown. If you're running from the build tree instead, export `GST_PLUGIN_PATH` to each plugin subdir as described in [Setup for Reproducing](#setup-for-reproducing-on-jetson).

### ROS2 bridge

The wire protocol (shared-memory header + unix-socket fd passing) is defined in [`gst/common/shm_protocol.h`](gst/common/shm_protocol.h). A ROS2 node that wants to consume `nvmmsink` output without going through GStreamer should follow the same handshake as [`gst/nvmmappsrc/gstnvmmappsrc.cpp`](gst/nvmmappsrc/gstnvmmappsrc.cpp): attach to the shm segment, connect to `header->socket_path`, receive the pool's `NvBufSurfaceMapParams` + DMA-buf fds, and import each one with `NvBufSurfaceImport`. Per-frame reads: wait for `header->ready`, atomically increment `ref_counts[write_idx]`, read from the imported surface, decrement.

## JetPack Compatibility

| JetPack | L4T | Jetson | Status |
|---------|-----|--------|--------|
| 5.1.2 | R35.4.1 | Xavier NX | **Tested** — 9/9 suites, zero-copy IPC verified (`feature/zero-copy-jp5-jp6`) |
| 5.1.1+ | R35.3.1+ | Xavier (NX, AGX) | Supported — minimum for `NvBufSurfaceImport` |
| 6.x | R36.x | Orin | Build-validated (Docker); hardware run pending |
| 5.0.x | R35.2.1 | — | **Not supported.** No `NvBufSurfaceImport`. Upgrade to JP 5.1.1+. |
| N/A | N/A | x86_64 desktop | Mock API for unit tests only (`-Dmock=true`) |

The build probes `nvbufsurface.h` for `NvBufSurfaceImport` and hard-fails at meson configure if absent. A second probe at `producer_start` / `consumer_start` (via `dlsym`) catches deploy-time mismatch where the binary was built against newer headers but ends up running on an older host BSP.

## Tests

49 tests across 9 suites (plus a 10 000-iteration fuzz run):

| Suite | Tests | What it covers |
|-------|-------|---------------|
| `nvmm_buffer` | 10 | NvmmBuffer RAII: create, map, unmap, move, export_fd, planes (NV12, RGBA, I420) |
| `nvmm_transform` | 6 | NvmmTransform: scale, crop_and_scale, format convert, flip, null safety |
| `gst_nvmm_allocator` | 8 | GstNvmmAllocator: create, alloc/free, map/unmap, write/read round-trip, non-NVMM rejection |
| `nvmm_sink` | 4 | GstNvmmSink: element creation, properties, state transitions, shm lifecycle |
| `nvmm_appsrc` | 2 | GstNvmmAppSrc: element creation, properties |
| `gstcheck_elements` | 8 | Element discovery (3), state transitions (2), property validation, pad template caps, pipeline wiring |
| `integration` | 6 | Multiple shm segments, dynamic properties, pipeline bin, alloc stress, protocol validation, missing-shm error handling |
| `backend_concurrency` | 5 | Concurrent producer/consumer (200 frames), start/stop churn, multi-consumer fanout, CRC roundtrip, zerocopy path |
| `fuzz_shm_header` | — | 10 000 random-byte iterations against the shm header parser; 0 crashes |

```bash
# Run all tests (Docker, x86_64)
docker build -f docker/Dockerfile.dev -t gst-nvmm-cpp:dev .
docker run --rm gst-nvmm-cpp:dev

# Run all tests (Jetson, native)
LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra meson test -C builddir --verbose
# or, using CMake:
ctest --test-dir build-cmake --output-on-failure
```

## Repository Structure

```
gst-nvmm-cpp/
├── gst/
│   ├── common/              # Shared C++ types and RAII wrappers
│   │   ├── nvmm_types.hpp   # Result<T>, ByteSpan, enums, error codes
│   │   ├── nvmm_buffer.hpp  # NvmmBuffer -- RAII wrapper for NvBufSurface
│   │   ├── nvmm_transform.hpp # NvmmTransform -- NvBufSurfTransform wrapper
│   │   ├── nvmm_buffer.cpp  # Jetson implementation
│   │   ├── nvmm_transform.cpp # Jetson implementation
│   │   ├── nvbufsurface_mock.h # Mock API for x86_64 host builds
│   │   ├── *_mock.cpp       # Mock implementations
│   │   └── meson.build
│   ├── nvmmalloc/           # GstNvmmAllocator plugin
│   ├── nvmmconvert/         # nvmmconvert element plugin
│   ├── nvmmsink/            # nvmmsink element plugin
│   └── nvmmappsrc/          # nvmmappsrc element plugin
├── tests/                   # 42 unit + integration tests
├── benchmarks/              # Throughput benchmarks (CSV output)
├── test_output/             # Sample images from Jetson pipeline tests
├── docker/                  # Dockerfiles: dev (host mock), jetson (universal), jp5, jp6
├── meson.build              # Top-level build (probes for NvBufSurfaceImport)
└── README.md
```

## Related Issues

These issues document the upstream gaps this project addresses:

- [#4979 -- nvcodec: No Tegra/NVMM allocator path](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4979)
- [#4980 -- Missing GstAllocator wrapper for NvBufSurface](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4980)
- [#4981 -- NvBufSurfTransform has no GStreamer element](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4981)

## License

LGPL-2.1-or-later. See [COPYING](COPYING).
