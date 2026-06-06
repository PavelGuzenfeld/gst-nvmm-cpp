# gst-nvmm-cpp

Zero-copy **NVMM-native** GStreamer elements for NVIDIA Jetson (Xavier, Orin).
Wraps the Tegra-native `NvBufSurface` / NVMM memory model in proper GStreamer
elements — hardware-accelerated crop, scale, format conversion, rotate/flip, and
**inter-process video sharing with no CPU copies on the data path**.

- **C++14** internals, **C-ABI** boundary to GStreamer.
- A real `GstAllocator` for `NvBufSurface`.
- **No DeepStream dependency**, LGPL, upstream-able.

## Why this exists

On Jetson the native buffer type is `NvBufSurface` (NVMM) — physically
contiguous, DMA-coherent memory managed by the Tegra **VIC**. The stock
`nvcodec` plugin targets desktop GPUs via CUDA and doesn't understand NVMM, and
DeepStream (which does) is heavyweight and single-process. This suite fills the
gap with small open-source elements — and, uniquely, a **cross-process zero-copy
NVMM transport** (`nvmmsink` → `nvmmappsrc`) that NVIDIA ships no stock element
for. See [Zero-copy IPC](ipc.md).

## Elements

| Element | Role |
|---|---|
| [`nvmmconvert`](elements/nvmmconvert.md) | Crop / scale / format-convert / rotate-flip on the VIC |
| [`nvmmsink`](elements/nvmmsink.md) | Publish NVMM frames to a shared pool; pass DMA-buf fds to consumers |
| [`nvmmappsrc`](elements/nvmmappsrc.md) | Import a producer's pool fds and read GPU memory in place |
| [`nvmmalloc`](elements/nvmmalloc.md) | `GstAllocator` for `NvBufSurface` |

## Status

Validated on Jetson Xavier NX (JP5.1.2) and Orin NX (JP6): 47 unit/integration
tests + 10 on-hardware pipeline tests — see [Validation & benchmarks](validation.md).
What's next is tracked on the [Roadmap](PRODUCTION_PLAN.md).
