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
