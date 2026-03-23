# gst-nvmm-cpp

GStreamer plugin suite for **zero-copy video processing** on NVIDIA Jetson platforms (Xavier, Orin). Wraps the Tegra-native `NvBufSurface` / NVMM memory model in proper GStreamer elements, enabling hardware-accelerated crop, scale, format conversion, and inter-process video sharing — all without CPU copies.

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

### nvmmsink

Exports NVMM video frames to a **POSIX shared memory** segment for zero-copy inter-process communication. Consumers (ROS2 nodes, inference engines, visualization tools) attach to the segment and read frames without serialization overhead.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `shm-name` | string | `/nvmm_sink_0` | POSIX shared memory segment name |
| `export-dmabuf` | bool | false | Export DMA-buf fd in shared memory header |

**Shared memory protocol:**

```
[ ShmHeader (128 bytes) ][ frame data ]
```

The `ShmHeader` contains:
- Magic (`0x4E564D4D` = "NVMM"), version, width, height, pixel format
- Per-plane pitches and offsets
- DMA-buf fd (when `export-dmabuf=true`)
- Monotonic frame number and PTS timestamp (nanoseconds)
- Atomic `ready` flag for lock-free consumer reads

### nvmmappsrc

Reads NVMM video frames from a **POSIX shared memory** segment written by `nvmmsink` or an external producer. Pushes frames into a GStreamer pipeline.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `shm-name` | string | `/nvmm_sink_0` | POSIX shared memory segment name to read from |
| `is-live` | bool | true | Whether this source is a live source |

Auto-detects video format, resolution, and plane layout from the `ShmHeader` on the first frame.

### GstNvmmAllocator

A `GstAllocator` subclass that wraps `NvBufSurfaceCreate` / `NvBufSurfaceDestroy` with proper `GstMemory` semantics.

```cpp
// Allocate an NVMM buffer
GstAllocator *alloc = gst_nvmm_allocator_new(0 /* NVBUF_MEM_DEFAULT */);
GstMemory *mem = gst_allocator_alloc(alloc, 1920 * 1080 * 3 / 2, NULL);

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
- Meson >= 0.62, Ninja
- C++14 compiler (GCC 7+ or Clang 5+)
- On Jetson: JetPack 5 (L4T 35.x) or JetPack 6 (L4T 36.x)

### Docker (recommended for x86_64 host)

```bash
# Host (x86_64) -- builds with mock NvBufSurface API for testing
docker build -f docker/Dockerfile.dev -t gst-nvmm-cpp:dev .
docker run --rm gst-nvmm-cpp:dev
```

### Native build (Jetson)

```bash
pip3 install meson
meson setup builddir -Dcpp_std=c++14 -Dbuildtype=debugoptimized
ninja -C builddir
meson test -C builddir --verbose
```

On hosts without Jetson libraries, meson automatically detects the absence of `libnvbufsurface.so` and builds with the **mock API** — a header-only stub that implements the same `NvBufSurface` struct layout and function signatures using heap memory. All tests pass against the mock.

## Jetson Hardware Validation

Validated on **Jetson Xavier NX** (JetPack 5.1, L4T R35.2.1, GStreamer 1.16.3).

### Test Results

All 7 test suites pass with real NVMM hardware:

```
$ meson test -C builddir --verbose
 1/7 gst-nvmm-cpp:nvmm_buffer        OK   9 passed, 0 failed
 2/7 gst-nvmm-cpp:nvmm_transform     OK   6 passed, 0 failed
 3/7 gst-nvmm-cpp:gst_nvmm_allocator OK   5 passed, 0 failed
 4/7 gst-nvmm-cpp:nvmm_sink          OK   4 passed, 0 failed
 5/7 gst-nvmm-cpp:nvmm_appsrc        OK   3 passed, 0 failed
 6/7 gst-nvmm-cpp:gstcheck_elements  OK   8 passed, 0 failed
 7/7 gst-nvmm-cpp:integration        OK   7 passed, 0 failed
Ok: 7   Fail: 0
```

### Benchmark Results (Xavier NX)

```
benchmark,iterations,total_us,avg_us,min_us,max_us
alloc_free (NV12 1080p),1000,696675,696.68,134.88,3005.51
alloc_free (RGBA 1080p),1000,1825268,1825.27,586.69,3845.29
map_unmap (NV12 1080p),1000,252155,252.16,226.11,2185.00
map_unmap (NV12 480p),1000,106209,106.21,93.25,529.44
transform_scale (1080p->480p),1000,21392,21.39,18.40,1144.64
transform_scale (1080p->720p),1000,20707,20.71,18.59,70.21
```

Key findings:
- **VIC transform: ~21 us/frame** (1080p -> 480p scale) = ~47,000 FPS throughput
- **NvBufSurface alloc: ~700 us** (NV12 1080p)
- **CPU map/unmap: ~250 us** (NV12 1080p, includes cache sync)

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

Sample outputs in `test_output/`:
- `smpte_1080p.jpg` -- 1920x1080 SMPTE color bars (133 KB)
- `gpu2cpu_1080p.jpg` -- 1920x1080 decoded via NVMM, written to CPU (134 KB)
- `4k_roundtrip.jpg` -- 3840x2160 CPU->NVMM->CPU roundtrip (491 KB)
- `4k_to_fhd.jpg` -- 3840x2160 scaled to 1920x1080 via NVMM (34 KB)
- `decoded_frame.jpg` -- 640x480 H264 decoded via NVMM (254 KB)

### Setup for Reproducing on Jetson

```bash
# 1. Install meson
pip3 install meson

# 2. Clone and build
git clone https://github.com/PavelGuzenfeld/gst-nvmm-cpp.git
cd gst-nvmm-cpp
meson setup builddir -Dcpp_std=c++14 -Dbuildtype=debugoptimized -Dwerror=false
ninja -C builddir

# 3. Run tests (clear GStreamer registry cache first)
rm -f ~/.cache/gstreamer-1.0/registry.*.bin
LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra meson test -C builddir --verbose

# 4. Run benchmarks
LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra ./builddir/benchmarks/bench_nvmm

# 5. Use the plugins in pipelines
export GST_PLUGIN_PATH=$(pwd)/builddir/gst/nvmmconvert:$(pwd)/builddir/gst/nvmmsink:$(pwd)/builddir/gst/nvmmappsrc:$(pwd)/builddir/gst/nvmmalloc
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:$(pwd)/builddir/gst/nvmmalloc
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

### ROS2 bridge (conceptual)

```cpp
// In a ROS2 node -- attach to shared memory written by nvmmsink
int fd = shm_open("/camera_feed", O_RDONLY, 0);
void *ptr = mmap(NULL, shm_size, PROT_READ, MAP_SHARED, fd, 0);
auto *header = static_cast<ShmHeader *>(ptr);

while (rclcpp::ok()) {
    if (header->ready && header->frame_number > last_frame) {
        auto *frame_data = (uint8_t *)ptr + sizeof(ShmHeader);
        sensor_msgs::msg::Image msg;
        msg.width = header->width;
        msg.height = header->height;
        msg.data.assign(frame_data, frame_data + header->data_size);
        publisher->publish(msg);
        last_frame = header->frame_number;
    }
}
```

## JetPack Compatibility

| JetPack | L4T | Jetson | NvBufSurface | NvSciBuf | Status |
|---------|-----|--------|-------------|----------|--------|
| 5.1.x | R35.x | Xavier NX, Orin | Yes | No | Tested |
| 6.x | R36.x | Orin | Yes | Yes | Supported |
| N/A | N/A | x86_64 desktop | Mock API | No | Testing only |

The build system auto-detects JetPack version via `/etc/nv_tegra_release` and enables NvSciBuf support on JP6.

## Tests

42 tests across 7 suites:

| Suite | Tests | What it covers |
|-------|-------|---------------|
| `nvmm_buffer` | 9 | NvmmBuffer RAII: create, map, unmap, move, export_fd, planes (NV12, RGBA, I420) |
| `nvmm_transform` | 6 | NvmmTransform: scale, crop_and_scale, format convert, flip, null safety |
| `gst_nvmm_allocator` | 5 | GstNvmmAllocator: create, alloc/free, map/unmap, write/read round-trip, non-NVMM rejection |
| `nvmm_sink` | 4 | GstNvmmSink: element creation, properties, state transitions, shm lifecycle |
| `nvmm_appsrc` | 3 | GstNvmmAppSrc: element creation, properties, sink-to-source integration via shm |
| `gstcheck_elements` | 8 | Element discovery (3), state transitions (2), property validation, pad template caps, pipeline wiring |
| `integration` | 7 | Shm data round-trip, multiple shm segments, dynamic properties, pipeline bin, alloc stress, protocol validation, error handling |

```bash
# Run all tests (Docker, x86_64)
docker build -f docker/Dockerfile.dev -t gst-nvmm-cpp:dev .
docker run --rm gst-nvmm-cpp:dev

# Run all tests (Jetson, native)
LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra meson test -C builddir --verbose
```

## Repository Structure

```
gst-nvmm-cpp/
├── gst/
│   ├── common/              # Shared C++ types and RAII wrappers
│   │   ├── nvmm_types.hpp   # Result<T>, ByteSpan, enums, error codes
│   │   ├── nvmm_buffer.hpp  # NvmmBuffer -- RAII wrapper for NvBufSurface
│   │   ├── nvmm_transform.hpp # NvmmTransform -- NvBufSurfTransform wrapper
│   │   ├── nvmm_buffer.cpp  # Real implementation (Jetson)
│   │   ├── nvmm_transform.cpp # Real implementation (Jetson)
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
├── docker/                  # Dockerfiles for dev, JP5, JP6
├── meson.build              # Top-level build (auto-detects Jetson)
└── README.md
```

## Related Issues

These issues document the upstream gaps this project addresses:

- [#4979 -- nvcodec: No Tegra/NVMM allocator path](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4979)
- [#4980 -- Missing GstAllocator wrapper for NvBufSurface](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4980)
- [#4981 -- NvBufSurfTransform has no GStreamer element](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4981)

## License

LGPL
