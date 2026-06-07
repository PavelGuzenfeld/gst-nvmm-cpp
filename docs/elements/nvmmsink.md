# nvmmsink

Shares NVMM video frames across processes via a **GPU-copy pool**. Each incoming
frame is copied GPU-to-GPU into a fixed pool of NVMM buffers (via
`NvBufSurfTransform` on the VIC, which also de-tiles BLOCK_LINEAR input into the
pool's PITCH_LINEAR layout), and the pool buffers' DMA-buf fds are handed to
consumers over a unix-domain socket (`SCM_RIGHTS`). Consumers import the fds and
read GPU memory directly. See [Zero-copy IPC](../ipc.md).

Sink caps: `video/x-raw(memory:NVMM)`, `{NV12, RGBA, I420, BGRA}`.

## Properties

| Property | Type | Default | Description |
|---|---|---|---|
| `shm-name` | string | `/nvmm_sink_0` | POSIX shared-memory segment name (e.g. `/cam1`) |
| `pool-size` | int (13–16) | 16 | Number of NVMM buffers in the shared pool |

!!! note "Why the minimum is 13"
    `nvmmappsrc` holds a reference on up to `RELEASE_DELAY` (12) in-flight
    buffers (to cover the consumer's hardware-encoder pipeline depth). A pool of
    ≤ 12 lets a steady consumer hold every slot and starve the producer (the
    stream stalls after ~`pool-size` frames). The property therefore enforces a
    minimum of **13** (`RELEASE_DELAY + 1`); the default is 16.

## Example (producer)

```bash
gst-launch-1.0 videotestsrc num-buffers=300 pattern=smpte ! \
  'video/x-raw,width=640,height=480,format=I420,framerate=10/1' ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
  nvmmsink shm-name=/ipc_test sync=true
```

The producer makes **one** GPU-side copy into the shared pool; consumers then
import without further copies (single-copy producer, zero-copy consumers).

## DeepStream interop

`nvmmsink` takes `video/x-raw(memory:NVMM)` (NV12 / RGBA) — the same surfaces a
DeepStream pipeline carries — so you can **tap a DeepStream pipeline and publish
its frames to other processes**. Insert it on an NVMM branch after the stage you
want to share:

!!! success "Verified on Orin"
    Run on Orin NX / JetPack 6 (L4T R36.4.3) / DeepStream 7.1 (samples container):
    this pipeline (TrafficCamNet `nvinfer` + `nvdsosd`, `batch-size=1`) published
    annotated NVMM frames through `nvmmsink`, and a separate `nvmmappsrc` consumer
    imported and decoded **1422 frames** (re-encoded to JPEG) — confirming
    `nvmmsink` accepts post-DeepStream NVMM and consumers read the annotated pixels.
    Set
    `nvinfer batch-size=1` for a single stream (the stock config defaults to a
    heavy batch-30 INT8 engine); adjust `config-file-path` to your install.

```bash
# Run inference, burn the overlay into the pixels, publish annotated NVMM frames
gst-launch-1.0 \
  filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder \
  ! 'video/x-raw(memory:NVMM)' \
  ! nvstreammux batch-size=1 width=1920 height=1080 \
  ! nvinfer config-file-path=pgie.txt batch-size=1 \
  ! nvvideoconvert ! nvdsosd \
  ! nvvideoconvert ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! nvmmsink shm-name=/ds_annotated
```

!!! note "Only pixels cross the IPC boundary — not DeepStream metadata"
    `NvDsBatchMeta` and per-object metadata (boxes, classes, tracker IDs) live on
    the `GstBuffer`, **not** the `NvBufSurface`. `nvmmsink` ships only the surface,
    so downstream consumers receive the *frames* but **not** the inference results
    as metadata. To share the result, render it into the pixels with `nvdsosd`
    **before** `nvmmsink` (as above). Consumers then read finished, annotated
    frames; they cannot recover the structured detections.
