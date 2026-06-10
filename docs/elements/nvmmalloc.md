# nvmmalloc

A `GstAllocator` for `NvBufSurface` (NVMM) memory — there is no stock one.
It provides NVMM-backed `GstMemory` so buffer pools and elements can allocate,
map, share, and pass Tegra NVMM surfaces through the standard allocator
interface, with RAII lifetimes (`NvBufSurfaceCreate`/`Destroy`, map/unmap).

It underpins the other elements (`nvmmconvert`, `nvmmsink`, `nvmmappsrc`, the
inference graph) and the buffer-pool plumbing. It is not a pipeline element
and has no GObject properties; it is configured through the standard
`GstAllocationParams` / video-info path.

Memory is share-capable: `gst_buffer_make_writable` after a `tee` performs a
shallow copy — a new `GstBuffer` and meta list referencing the same
`NvBufSurface` — so per-branch metadata can be attached without duplicating
pixels.

## DeepStream interop

DeepStream allocates the same `NvBufSurface` type through its own internal
pools, so buffers allocated here flow into DeepStream elements
(`nvstreammux`, `nvinfer`, …) without a copy. On Orin NX (JP6 /
DeepStream 7.1) the suite builds against DeepStream's own `NvBufSurface` and
co-registers in one GStreamer registry alongside
`nvinfer`/`nvstreammux`/`nvdsosd`; see the interop sections of
[nvmmconvert](nvmmconvert.md#deepstream-interop),
[nvmmsink](nvmmsink.md#deepstream-interop), and
[nvmmappsrc](nvmmappsrc.md#deepstream-interop).
