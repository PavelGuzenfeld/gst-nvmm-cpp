# gst-nvmm-cpp

Zero-copy **NVMM-native** GStreamer elements for NVIDIA Jetson (Xavier, Orin):
hardware-accelerated video processing, TensorRT inference graphs, and
inter-process video sharing with no CPU copies on the data path. C++14
internals, C-ABI boundary to GStreamer, LGPL, no DeepStream dependency.

On Jetson the native buffer type is `NvBufSurface` (NVMM) — physically
contiguous, DMA-coherent memory operated on by the Tegra VIC. The stock
`nvcodec` plugin targets desktop GPUs and does not understand NVMM; DeepStream
does, but is heavyweight and single-process. This suite provides small
standalone elements for that memory model, including a cross-process zero-copy
NVMM transport (`nvmmsink` → `nvmmappsrc`) for which NVIDIA ships no stock
element. See [Zero-copy IPC](ipc.md).

## Elements

**Memory & IPC** — the allocator and the cross-process transport.

| Element | Role |
|---|---|
| [`nvmmalloc`](elements/nvmmalloc.md) | `GstAllocator` for `NvBufSurface`; underpins the suite's buffer pools |
| [`nvmmsink`](elements/nvmmsink.md) | Publish NVMM frames to a shared pool; pass DMA-buf fds to consumers |
| [`nvmmappsrc`](elements/nvmmappsrc.md) | Import a producer's pool fds and read GPU memory in place |

**Video processing** — 2D operations on the VIC.

| Element | Role |
|---|---|
| [`nvmmconvert`](elements/nvmmconvert.md) | Crop / scale / format-convert / rotate-flip |
| [`nvmmcompositor`](elements/nvmmcompositor.md) | Composite multiple NVMM inputs into one frame (mosaic / PiP) |

**Inference & analytics** — composable nodes for building
[inference graphs](inference-graphs.md); each passes the frame through
untouched and reads or attaches metadata.

| Element | Role |
|---|---|
| [`nvmminfer`](elements/nvmminfer.md) | TensorRT object detection; attaches `GstNvmmDetMeta` |
| [`nvmmtracker`](elements/nvmmtracker.md) | IOU multi-object tracking; assigns stable `tracker_id`s |
| [`nvmmofa`](elements/nvmmofa.md) | Dense optical flow on the Orin OFA engine; flow rides as metadata |
| [`nvmmfusion`](elements/nvmmfusion.md) | Joins detector + flow branches by PTS; computes per-object motion |
| [`nvmmsecondaryinfer`](elements/nvmmsecondaryinfer.md) | Cascade classifier on detected objects, with per-track caching |
| [`nvmmdrawdet`](elements/nvmmdrawdet.md) | Renders boxes, ids, motion and classification onto the frame |

## Where to start

- [Getting started](getting-started.md) — build and run the first pipeline.
- [Pipeline examples](pipelines.md) — transport, processing and encoding.
- [Inference graphs](inference-graphs.md) — combining the analytics elements.
- [Creating a new element](extending.md) — extend the suite with your own node.

## Status

Validated on Jetson Xavier NX (JP5.1.2) and Orin NX (JP6); the unit/integration
suite runs on x86 CI against a mock `NvBufSurface` and on-device against the
real stack — see [Validation & benchmarks](validation.md).
