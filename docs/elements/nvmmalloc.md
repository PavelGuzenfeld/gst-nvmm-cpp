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

## DeepStream interop

`nvmmalloc` has no place in a `gst-launch` line — it's a `GstAllocator`, not a
pipeline element — so there is no DeepStream "example" to run for it. The
relationship is at the memory layer: DeepStream allocates the very same
`NvBufSurface` type through its own internal pools, while `nvmmalloc` exposes that
surface type through the **standard `GstAllocator` interface**. Because both sides
speak `NvBufSurface`, buffers allocated here can flow into DeepStream elements
(`nvstreammux`, `nvinfer`, …) without a copy. On Orin NX (JP6 / DeepStream 7.1)
the whole suite builds against DeepStream's own `NvBufSurface` and co-registers in
one GStreamer registry alongside `nvinfer`/`nvstreammux`/`nvdsosd` — the interop in
[nvmmconvert](nvmmconvert.md#deepstream-interop),
[nvmmsink](nvmmsink.md#deepstream-interop), and
[nvmmappsrc](nvmmappsrc.md#deepstream-interop) is what `nvmmalloc` underpins.
