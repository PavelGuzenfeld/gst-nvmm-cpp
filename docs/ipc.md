# Zero-copy IPC (the differentiator)

The headline capability NVIDIA ships **no** stock element for: sharing GPU
(NVMM) video frames between **independent processes** without a CPU copy on the
consumer. DeepStream batches within a *single* process; this does it *across*
processes.

## Data path

```
producer process                          consumer process(es)
videotestsrc/camera/decoder
   → NVMM frame
   → nvmmsink                             nvmmappsrc
       • pool of NVMM buffers                • shm_open + connect socket
       • 1 GPU-side copy (VIC) per frame     • recv NvBufSurfaceMapParams + fds
       • publish via POSIX shm header        • NvBufSurfaceImport(fd)  ← zero copy
       • pass DMA-buf fds (SCM_RIGHTS) ─────▶ • read GPU memory in place
```

- **Producer:** one GPU/VIC copy of each frame into a stable, ref-counted shared
  pool (decouples lifetime from the transient upstream buffer, and de-tiles
  BLOCK_LINEAR → PITCH_LINEAR).
- **Consumer:** imports the pool buffer's fd and reads the **same physical
  memory** — verified byte-identical on hardware. No further copy.

So the suite is **single-copy in, zero-copy out**, with no CPU on the path.

## Wire protocol

A POSIX shared-memory header (`gst/common/shm_protocol.h`) carries magic,
version, resolution/format, per-plane pitches/offsets, a ring of `ref_counts`,
`write_idx`, frame number, timestamp, and the producer's unix-socket path.
fd passing uses `SCM_RIGHTS` over that socket.

## ROS2 / non-GStreamer consumers

A non-GStreamer consumer (e.g. a ROS2 node) can follow the same handshake as
`nvmmappsrc`: attach to the shm segment, connect to `header->socket_path`,
receive the pool's `NvBufSurfaceMapParams` + fds, `NvBufSurfaceImport` each, then
per frame: wait for `ready`, atomically increment `ref_counts[write_idx]`, read
the imported surface, decrement.

## Multi-consumer fan-out

One producer pool is written once per frame; every consumer just imports the fds
and reads in place — adding a second or third consumer adds **no** extra GPU copy.
Ideal for N cameras → K independent encoder/inference processes.
