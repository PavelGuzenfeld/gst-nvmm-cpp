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
| `pool-size` | int (3–16) | 16 | Number of NVMM buffers in the shared pool |

!!! warning "Keep pool-size above the consumer's hold depth"
    `nvmmappsrc` holds a reference on up to `RELEASE_DELAY` (12) in-flight
    buffers (to cover the consumer's hardware-encoder pipeline depth). If
    `pool-size` ≤ 12, a steady consumer can hold every slot, starving the
    producer — the stream then stalls after ~`pool-size` frames. Keep the
    **default (16)**, or set `pool-size` ≥ 13 when a consumer is attached.

## Example (producer)

```bash
gst-launch-1.0 videotestsrc num-buffers=300 pattern=smpte ! \
  'video/x-raw,width=640,height=480,format=I420,framerate=10/1' ! \
  nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
  nvmmsink shm-name=/ipc_test sync=true
```

The producer makes **one** GPU-side copy into the shared pool; consumers then
import without further copies (single-copy producer, zero-copy consumers).
