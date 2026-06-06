# gst-nvmm-cpp

Zero-copy **NVMM-native** GStreamer elements for NVIDIA Jetson (Xavier, Orin).
Wraps the Tegra-native `NvBufSurface` / NVMM memory model in proper GStreamer
elements — hardware-accelerated crop, scale, format conversion, rotate/flip, and
**inter-process video sharing with no CPU copies on the data path**.

Built with **C++14** internals and a **C-ABI** boundary to GStreamer. LGPL,
no DeepStream dependency.

> 📖 **Full documentation:** <https://pavelguzenfeld.com/gst-nvmm-cpp/>
> — elements & properties, getting started, the zero-copy IPC design, pipeline
> examples, and hardware validation/benchmarks.

## The problem this solves

On Jetson the native video buffer type is `NvBufSurface` (NVMM) — physically
contiguous, DMA-coherent memory managed by the Tegra VIC hardware engine. The
stock `nvcodec` plugin targets desktop GPUs via CUDA and doesn't understand
NVMM; DeepStream (which does) is heavyweight and single-process. This creates
gaps:

- `nvv4l2decoder` outputs `video/x-raw(memory:NVMM)` but no upstream GStreamer
  element can consume it without a CPU copy.
- No standard `GstAllocator` exists for `NvBufSurface`, so every team writes
  their own.
- There is **no** stock element for sharing NVMM frames across *independent*
  processes, zero-copy.

This project fills those gaps with small, open-source, tested, upstream-ready
elements.

## Elements

| Element | Role |
|---|---|
| [`nvmmconvert`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmconvert/) | Crop / scale / format-convert / rotate-flip on the VIC |
| [`nvmmsink`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmsink/) | Publish NVMM frames to a shared pool; pass DMA-buf fds to consumers |
| [`nvmmappsrc`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmappsrc/) | Import a producer's pool fds and read GPU memory in place |
| [`nvmmalloc`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmalloc/) | `GstAllocator` for `NvBufSurface` |

The headline capability NVIDIA ships no stock element for: **cross-process,
zero-copy NVMM sharing** (`nvmmsink` → `nvmmappsrc`) — single GPU-side copy on
the producer, zero-copy import on every consumer. See the
[Zero-copy IPC](https://pavelguzenfeld.com/gst-nvmm-cpp/ipc/) docs.

## Quick start

```bash
# Host (x86_64) — build + test with the mock NvBufSurface API, in Docker
docker build -f docker/Dockerfile.dev -t gst-nvmm-cpp:dev .
docker run --rm --shm-size=256m gst-nvmm-cpp:dev meson test -C builddir --verbose

# Jetson — native build
meson setup builddir -Dcpp_std=c++14 -Dbuildtype=debugoptimized -Dwerror=false
ninja -C builddir
```

Full build/install/run instructions, pipeline examples, and the IPC design are
in the [documentation site](https://pavelguzenfeld.com/gst-nvmm-cpp/).

## Status

Validated on Jetson Xavier NX (JP5.1.x) and Orin NX (JP6), in Docker and native:
**48 unit/integration tests + 11 on-hardware pipeline tests**, AddressSanitizer
and ThreadSanitizer clean. Full results, benchmarks, and evidence images:
[Validation & benchmarks](https://pavelguzenfeld.com/gst-nvmm-cpp/validation/).

| JetPack | L4T | Jetson | NvBufSurface | Status |
|---------|-----|--------|--------------|--------|
| 5.1.x | R35.x | Xavier NX | Yes | Tested |
| 6.x | R36.x | Orin NX | Yes | Tested |
| N/A | N/A | x86_64 desktop | Mock API | Testing only |

## Repository structure

```
gst-nvmm-cpp/
├── gst/
│   ├── common/      # Shared C++ types, RAII NvBufSurface wrappers, mock API
│   ├── nvmmalloc/   # GstNvmmAllocator plugin
│   ├── nvmmconvert/ # nvmmconvert element
│   ├── nvmmsink/    # nvmmsink element
│   └── nvmmappsrc/  # nvmmappsrc element
├── tests/           # unit + integration tests
├── benchmarks/      # throughput benchmarks
├── docs/            # documentation site (MkDocs)
├── docker/          # Dockerfiles for dev, JP5, JP6
└── meson.build      # top-level build (auto-detects Jetson; CMake also provided)
```

## Related upstream issues

- [#4979 — nvcodec: No Tegra/NVMM allocator path](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4979)
- [#4980 — Missing GstAllocator wrapper for NvBufSurface](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4980)
- [#4981 — NvBufSurfTransform has no GStreamer element](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/4981)

## License

LGPL-2.1-or-later. See [COPYING](COPYING).
