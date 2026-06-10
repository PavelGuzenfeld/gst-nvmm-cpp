# gst-nvmm-cpp

Zero-copy **NVMM-native** GStreamer elements for NVIDIA Jetson (Xavier, Orin).
Wraps the Tegra-native `NvBufSurface` / NVMM memory model in proper GStreamer
elements — hardware-accelerated video processing, TensorRT inference graphs
(detect → track → classify, optical flow, fusion), and **inter-process video
sharing with no CPU copies on the data path**.

Built with **C++14** internals and a **C-ABI** boundary to GStreamer. LGPL,
no DeepStream dependency.

> 📖 **Full documentation:** <https://pavelguzenfeld.com/gst-nvmm-cpp/>
> — elements & properties, getting started, inference graphs, the zero-copy
> IPC design, pipeline examples, and hardware validation/benchmarks.

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

**Memory & IPC** — the allocator and the cross-process transport.

| Element | Role |
|---|---|
| [`nvmmalloc`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmalloc/) | `GstAllocator` for `NvBufSurface`; underpins the suite's buffer pools |
| [`nvmmsink`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmsink/) | Publish NVMM frames to a shared pool; pass DMA-buf fds to consumers |
| [`nvmmappsrc`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmappsrc/) | Import a producer's pool fds and read GPU memory in place |

**Video processing** — 2D operations on the VIC.

| Element | Role |
|---|---|
| [`nvmmconvert`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmconvert/) | Crop / scale / format-convert / rotate-flip |
| [`nvmmcompositor`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmcompositor/) | Composite multiple NVMM inputs into one frame (mosaic / PiP) |

**Inference & analytics** — composable nodes for building
[inference graphs](https://pavelguzenfeld.com/gst-nvmm-cpp/inference-graphs/);
each passes the frame through untouched and reads or attaches metadata.

| Element | Role |
|---|---|
| [`nvmminfer`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmminfer/) | TensorRT object detection; attaches detection metadata |
| [`nvmmtracker`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmtracker/) | IOU multi-object tracking; assigns stable `tracker_id`s |
| [`nvmmofa`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmofa/) | Dense optical flow on the Orin OFA engine; flow rides as metadata |
| [`nvmmfusion`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmfusion/) | Joins detector + flow branches by PTS; computes per-object motion |
| [`nvmmsecondaryinfer`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmsecondaryinfer/) | Cascade classifier on detected objects, with per-track caching |
| [`nvmmdrawdet`](https://pavelguzenfeld.com/gst-nvmm-cpp/elements/nvmmdrawdet/) | Renders boxes, ids, motion and classification onto the frame |

Cross-process, zero-copy NVMM sharing (`nvmmsink` → `nvmmappsrc`) has no stock
NVIDIA equivalent: a single GPU-side copy on the producer, zero-copy import on
every consumer. See the
[Zero-copy IPC](https://pavelguzenfeld.com/gst-nvmm-cpp/ipc/) docs.

A full inference graph, wired entirely in GStreamer:

```
                ┌─ nvmminfer → nvmmtracker ─┐
src → tee ──────┤                           ├─ nvmmfusion → nvmmsecondaryinfer → nvmmdrawdet → sink
                └─ nvmmofa ─────────────────┘
```

The suite is extensible — see
[Creating a new element](https://pavelguzenfeld.com/gst-nvmm-cpp/extending/),
anchored by the in-tree example element in `gst/nvmmexample/`.

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

Validated on Jetson Xavier NX (JP5.1.x) and Orin NX (JP6), in Docker and
native. The unit/integration suite runs on x86 CI against a mock
`NvBufSurface` (in both C++14 and C++20 lanes) and on-device against the real
stack; on-hardware pipeline tests cover transforms, IPC, and the inference
graph. AddressSanitizer and ThreadSanitizer clean. Full results, benchmarks,
and evidence images:
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
│   ├── common/             # Shared types, metadata, RAII wrappers, mock API
│   ├── nvmmalloc/          # GstAllocator plugin for NvBufSurface
│   ├── nvmmconvert/        # VIC crop/scale/convert/rotate
│   ├── nvmmcompositor/     # VIC multi-input compositor
│   ├── nvmmsink/           # IPC producer (shared pool + fd passing)
│   ├── nvmmappsrc/         # IPC consumer (zero-copy import)
│   ├── nvmminfer/          # TensorRT detector (Jetson-only build)
│   ├── nvmmtracker/        # IOU tracker (pure host)
│   ├── nvmmofa/            # OFA optical flow (Orin/VPI only)
│   ├── nvmmfusion/         # PTS join of detector + flow branches
│   ├── nvmmsecondaryinfer/ # TensorRT cascade classifier (Jetson-only build)
│   ├── nvmmdrawdet/        # detection overlay (Jetson-only build)
│   └── nvmmexample/        # worked example element for docs/extending.md
├── tests/           # unit + integration tests
├── benchmarks/      # throughput benchmarks
├── probes/          # standalone hardware feasibility probes
├── scripts/         # Jetson validation, sanitizers, E2E test drivers
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
