# nvmmappsrc

Zero-copy NVMM IPC **source**. Connects to an `nvmmsink` (or compatible producer)
over its unix socket, receives the pool buffers' `NvBufSurfaceMapParams` + DMA-buf
fds, imports them with `NvBufSurfaceImport`, and pushes NVMM `GstBuffer`s
downstream — reading the producer's GPU memory **in place, no copy**. See
[Zero-copy IPC](../ipc.md).

Src caps: `video/x-raw(memory:NVMM)`, `{NV12, RGBA, I420, BGRA}`.

## Properties

| Property | Type | Default | Description |
|---|---|---|---|
| `shm-name` | string | `/nvmm_sink_0` | POSIX shared-memory segment to read from |
| `is-live` | bool | true | Whether this source is live |

## Example (consumer)

```bash
# NVMM -> system memory for JPEG (VIC via nvvidconv, not CPU videoconvert)
gst-launch-1.0 -e nvmmappsrc shm-name=/ipc_test is-live=true ! \
  'video/x-raw(memory:NVMM)' ! nvvidconv ! 'video/x-raw,format=I420' ! \
  nvjpegenc ! filesink location=ipc_out.jpg

# True zero-copy consumer: hardware encoder reads NVMM directly (Xavier — Orin NX has no NVENC)
gst-launch-1.0 -e nvmmappsrc shm-name=/ipc_test is-live=true ! \
  'video/x-raw(memory:NVMM),format=NV12' ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=out.mp4
```

> `nvmmappsrc` emits `memory:NVMM`. A CPU element (`videoconvert`) can't consume
> it — go through `nvvidconv` (VIC), or feed a hardware consumer like
> `nvv4l2h264enc` that takes NVMM directly.

## DeepStream interop

`nvmmappsrc` pushes `NvBufSurface`-backed NVMM buffers, which `nvstreammux`
accepts directly. So one producer published once (by `nvmmsink`) can fan out to
**multiple independent DeepStream inference processes**, each importing the pool
fds in place — no extra GPU copy per consumer. Feed `nvstreammux` (batch the
single stream), then `nvinfer`:

!!! success "Verified on Orin"
    Run on Orin NX / JetPack 6 (L4T R36.4.3) / DeepStream 7.1 (samples container):
    an `nvmmsink` producer feeding `nvmmappsrc ! nvstreammux ! nvinfer`
    (TrafficCamNet, `batch-size=1`) ran to EOS — `nvmmappsrc`'s NVMM negotiated into
    `nvstreammux` and inference ran on the **imported** buffers (headless test
    terminated in `fakesink` in place of `nv3dsink`). Adjust `nvstreammux`
    properties and the `nvinfer` `config-file-path` to your install; set
    `nvinfer batch-size=1` for a single stream (the stock config defaults to a
    heavy batch-30 INT8 engine).

```bash
# Zero-copy IPC source -> DeepStream detector
gst-launch-1.0 -e \
  nvmmappsrc shm-name=/ipc_test do-timestamp=true is-live=true \
  ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! nvstreammux batch-size=1 width=1920 height=1080 batched-push-timeout=40000 \
  ! nvinfer config-file-path=pgie.txt batch-size=1 \
  ! nvvideoconvert ! nvdsosd ! nv3dsink
```

!!! note "Only pixels cross the IPC boundary — not DeepStream metadata"
    `NvDsBatchMeta` and per-object metadata (bounding boxes, classes, tracker IDs)
    are attached to the `GstBuffer`, **not** to the `NvBufSurface`. `nvmmsink` /
    `nvmmappsrc` transfer only the surface, so metadata produced by an `nvinfer`
    in the *producer* process does **not** arrive here. Run inference in the
    consumer (as above), or — if you only need the *visualized* result — burn the
    overlay into the pixels with `nvdsosd` **before** `nvmmsink` in the producer.
