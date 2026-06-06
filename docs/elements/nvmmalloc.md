# nvmmalloc

A `GstAllocator` for `NvBufSurface` (NVMM) memory. Provides NVMM-backed
`GstMemory` so GStreamer buffer pools and elements can allocate, map, and pass
Tegra NVMM surfaces through the standard allocator interface — instead of every
team rolling their own.

It underpins the other elements (`nvmmconvert`, `nvmmsink`, `nvmmappsrc`) and the
buffer-pool plumbing. It has **no GObject properties** — it's a plain allocator,
configured through the standard `GstAllocationParams` / video-info path.

## Why it matters

There is no stock `GstAllocator` for `NvBufSurface`. Without one, NVMM interop
means hand-rolled buffer management in every project. `nvmmalloc` makes NVMM a
first-class GStreamer memory type with proper RAII lifetimes
(`NvBufSurfaceCreate`/`Destroy`, map/unmap), which is what lets the rest of the
suite stay zero-copy.
