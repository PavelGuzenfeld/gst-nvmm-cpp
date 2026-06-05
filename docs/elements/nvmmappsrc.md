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
