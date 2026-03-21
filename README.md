# gst-nvmm-cpp

GStreamer plugin suite for **zero-copy video processing** on NVIDIA Jetson platforms (Xavier, Orin). Wraps the Tegra-native `NvBufSurface` / NVMM memory model in proper GStreamer elements, enabling hardware-accelerated crop, scale, format conversion, and inter-process video sharing — all without CPU copies.

Built with **C++17** internals and a **C ABI** boundary to GStreamer.

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
GstAllocator *alloc = gst_nvmm_allocator_new(GST_NVMM_MEM_SURFACE_ARRAY);
GstMemory *mem = gst_allocator_alloc(alloc, 1920 * 1080 * 3 / 2, NULL);

// Check if memory is NVMM
if (gst_is_nvmm_memory(mem)) {
    NvBufSurface *surface = gst_nvmm_memory_get_surface(mem);
    // ... use surface directly with NVIDIA APIs
}

// Map for CPU access (triggers NvBufSurfaceMap internally)
GstMapInfo info;
gst_memory_map(mem, &info, GST_MAP_READ);
// info.data points to the mapped buffer
gst_memory_unmap(mem, &info);

gst_memory_unref(mem);
gst_object_unref(alloc);
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GStreamer Pipeline                        │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐             │
│  │ decoder  │───▶│ nvmmconvert  │───▶│ nvmmsink  │──▶ SHM      │
│  │(nvv4l2)  │    │ (VIC h/w)    │    │ (POSIX)   │             │
│  └──────────┘    └──────────────┘    └───────────┘             │
│       │                │                    │                   │
│       ▼                ▼                    ▼                   │
│  ┌──────────────────────────────────────────────┐              │
│  │           GstNvmmAllocator                    │              │
│  │  alloc ─▶ NvBufSurfaceCreate                 │              │
│  │  map   ─▶ NvBufSurfaceMap                    │              │
│  │  unmap ─▶ NvBufSurfaceUnMap                  │              │
│  │  free  ─▶ NvBufSurfaceDestroy                │              │
│  │  fd    ─▶ NvBufSurfaceGetFd (DMA-buf)        │              │
│  └──────────────────────────────────────────────┘              │
│                          │                                      │
│                          ▼                                      │
│              ┌────────────────────┐                             │
│              │  NvBufSurface API  │  (libnvbufsurface.so)       │
│              │  NvBufSurfTransform│  (libnvbufsurftransform.so) │
│              │  Tegra VIC Engine  │                              │
│              └────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘

         ┌───────────┐
  SHM ──▶│nvmmappsrc │──▶ downstream pipeline (ROS2, inference, etc.)
         └───────────┘
```

## C++ Design

The ABI boundary to GStreamer is C (`plugin_init`, GObject type system). Inside that boundary, everything is modern C++17.

| Pattern | Implementation |
|---------|---------------|
| **RAII for NvBufSurface** | `nvmm::NvmmBuffer` owns the surface, calls `NvBufSurfaceDestroy` in destructor |
| **Result type** | `nvmm::Result<T>` (C++17 `std::variant`-based, inspired by `std::expected`) |
| **Non-owning byte views** | `nvmm::ByteSpan` — lightweight replacement for `std::span<uint8_t>` |
| **Type-safe enums** | `nvmm::MemoryType`, `nvmm::ColorFormat`, `nvmm::FlipMethod` |
| **Zero GLib in internals** | `G_BEGIN_DECLS`/`G_END_DECLS` only at plugin boundaries |

## Building

### Prerequisites

- GStreamer >= 1.20 development libraries
- Meson >= 0.62, Ninja
- C++17 compiler (GCC 9+ or Clang 10+)
- On Jetson: JetPack 5 (L4T 35.x) or JetPack 6 (L4T 36.x)

### Docker (recommended)

```bash
# Host (x86_64) — builds with mock NvBufSurface API for testing
docker build -f docker/Dockerfile.dev -t gst-nvmm-cpp:dev .
docker run --rm gst-nvmm-cpp:dev                              # run tests
docker run --rm gst-nvmm-cpp:dev bash -c './benchmarks/bench_nvmm'  # run benchmarks

# Jetson Xavier (JetPack 5)
docker build -f docker/Dockerfile.jetson-jp5 -t gst-nvmm-cpp:jp5 .
docker run --runtime nvidia --rm gst-nvmm-cpp:jp5

# Jetson Orin (JetPack 6)
docker build -f docker/Dockerfile.jetson-jp6 -t gst-nvmm-cpp:jp6 .
docker run --runtime nvidia --rm gst-nvmm-cpp:jp6
```

### Manual build

```bash
meson setup builddir -Dcpp_std=c++17
ninja -C builddir
meson test -C builddir --verbose
```

On hosts without Jetson libraries, meson automatically detects the absence of `libnvbufsurface.so` and builds with the **mock API** — a header-only stub that implements the same `NvBufSurface` struct layout and function signatures using heap memory. All tests pass against the mock.

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
// In a ROS2 node — attach to shared memory written by nvmmsink
int fd = shm_open("/camera_feed", O_RDONLY, 0);
void *ptr = mmap(NULL, shm_size, PROT_READ, MAP_SHARED, fd, 0);
auto *header = static_cast<ShmHeader *>(ptr);

while (rclcpp::ok()) {
    if (header->ready && header->frame_number > last_frame) {
        auto *frame_data = (uint8_t *)ptr + sizeof(ShmHeader);
        // Publish as sensor_msgs/Image or use DMA-buf fd for zero-copy
        sensor_msgs::msg::Image msg;
        msg.width = header->width;
        msg.height = header->height;
        msg.data.assign(frame_data, frame_data + header->data_size);
        publisher->publish(msg);
        last_frame = header->frame_number;
    }
}
```

### Using GstNvmmAllocator in custom elements

```cpp
#include "gstnvmmallocator.h"

// In your element's decide_allocation or buffer pool setup:
GstAllocator *alloc = gst_nvmm_allocator_new(GST_NVMM_MEM_SURFACE_ARRAY);

// Allocate a buffer with specific video format
GstVideoInfo info;
gst_video_info_set_format(&info, GST_VIDEO_FORMAT_NV12, 1920, 1080);
GstMemory *mem = gst_nvmm_allocator_alloc(GST_NVMM_ALLOCATOR(alloc), &info);

// Get the raw NvBufSurface for direct NVIDIA API access
NvBufSurface *surface = gst_nvmm_memory_get_surface(mem);

// Export DMA-buf fd for V4L2 or display sink interop
gint fd;
gst_nvmm_memory_get_fd(mem, &fd);
```

## JetPack Compatibility

| JetPack | L4T | Jetson | NvBufSurface | NvSciBuf | Status |
|---------|-----|--------|-------------|----------|--------|
| 5.1.x | R35.x | Xavier NX, Orin | Yes | No | Supported |
| 6.x | R36.x | Orin | Yes | Yes | Supported |
| N/A | N/A | x86_64 desktop | Mock API | No | Testing only |

The build system auto-detects JetPack version via `/etc/nv_tegra_release` and enables NvSciBuf support on JP6.

## Tests

35 tests across 6 suites, all passing:

| Suite | Tests | What it covers |
|-------|-------|---------------|
| `nvmm_buffer` | 9 | NvmmBuffer RAII: create, map, unmap, move, export_fd, planes (NV12, RGBA, I420) |
| `nvmm_transform` | 6 | NvmmTransform: scale, crop_and_scale, format convert, flip, null safety |
| `gst_nvmm_allocator` | 5 | GstNvmmAllocator: create, alloc/free, map/unmap, write/read round-trip, non-NVMM rejection |
| `nvmm_sink` | 4 | GstNvmmSink: element creation, properties, state transitions, shm lifecycle |
| `nvmm_appsrc` | 3 | GstNvmmAppSrc: element creation, properties, **sink→source integration via shm** |
| `gstcheck_elements` | 8 | Element discovery (3), state transitions (2), property validation, pad template caps, pipeline wiring |

Plus 1 benchmark target with 6 measurements (alloc/free, map/unmap, transform at 1080p and 4K).

```bash
# Run all tests
docker run --rm gst-nvmm-cpp:dev

# Run benchmarks
docker run --rm gst-nvmm-cpp:dev bash -c '/src/builddir/benchmarks/bench_nvmm'
```

## Repository Structure

```
gst-nvmm-cpp/
├── gst/
│   ├── common/              # Shared C++ types and RAII wrappers
│   │   ├── nvmm_types.hpp   # Result<T>, ByteSpan, enums, error codes
│   │   ├── nvmm_buffer.hpp  # NvmmBuffer — RAII wrapper for NvBufSurface
│   │   ├── nvmm_transform.hpp # NvmmTransform — NvBufSurfTransform wrapper
│   │   ├── nvbufsurface_mock.h # Mock API for x86_64 host builds
│   │   ├── *_mock.cpp       # Mock implementations
│   │   └── meson.build
│   ├── nvmmalloc/           # GstNvmmAllocator plugin
│   │   ├── gstnvmmallocator.h/.cpp
│   │   ├── plugin.cpp
│   │   └── meson.build
│   ├── nvmmconvert/         # nvmmconvert element plugin
│   │   ├── gstnvmmconvert.h/.cpp
│   │   ├── plugin.cpp
│   │   └── meson.build
│   ├── nvmmsink/            # nvmmsink element plugin
│   │   ├── gstnvmmsink.h/.cpp
│   │   └── meson.build
│   └── nvmmappsrc/          # nvmmappsrc element plugin
│       ├── gstnvmmappsrc.h/.cpp
│       └── meson.build
├── tests/                   # 35 unit + integration tests
├── benchmarks/              # Throughput benchmarks (CSV output)
├── docker/                  # Dockerfiles for dev, JP5, JP6
├── meson.build              # Top-level build (auto-detects Jetson)
└── README.md
```

## Related Issues

These issues document the upstream gaps this project addresses:

- [#4979 — nvcodec: No Tegra/NVMM allocator path](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4979)
- [#4980 — Missing GstAllocator wrapper for NvBufSurface](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4980)
- [#4981 — NvBufSurfTransform has no GStreamer element](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4981)

## License

MIT
