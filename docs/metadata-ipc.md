# Detection metadata over IPC

By default the [zero-copy IPC](ipc.md) pair moves only the GPU **surface**:
`nvmmsink` hands consumers the pool buffers' DMA-buf fds and they import the
pixels in place. DeepStream's `NvDsBatchMeta` (bounding boxes, classes, tracker
IDs) lives on the producer's `GstBuffer` in host memory — it is **not** part of
the `NvBufSurface`, so it never crosses the boundary on its own.

The optional **metadata side-channel** carries a flat, self-contained copy of the
detections alongside each frame so a consumer can recover them without re-running
inference.

## What crosses — and what doesn't

The wire record is fixed-size POD (`NvmmDetObject` / `NvmmFrameMeta` in
`gst/common/shm_protocol.h`): per object a bbox, `class_id`, `confidence`,
`tracker_id`, and a label string. It does **not** carry:

- `NvDsUserMeta` / custom user metadata (it holds process-local function
  pointers that cannot cross processes),
- segmentation masks or tensor outputs (large, variable),
- classifier label hierarchies beyond the primary label.

If you need those, infer in the consumer instead (see [pipelines](pipelines.md)).

!!! warning "Coordinate space is part of the contract"
    `infer_width`/`infer_height` record the resolution the bboxes are expressed
    in. If the published surface is a different size, the consumer **must** rescale
    boxes by `surface_dim / infer_dim`. The producer stamps these as the published
    surface dimensions, so this matters when something rescales between `nvinfer`
    and `nvmmsink`. Single stream only (batch index 0); demux batched buffers
    first.

## How it travels

`nvmmsink` grows the shared segment by one `NvmmFrameMeta` record per pool buffer.
Slot *i* belongs to pool buffer *i* and is protected by the **same ref-count** as
the pixels, so the producer never overwrites metadata a consumer is still reading.
The record is written before the frame is published (behind the same memory
barrier as `write_idx`/`frame_number`), so any consumer that sees the new frame
also sees its metadata. No extra socket traffic, no new sync point.

## Using it

Build with the DeepStream bridge so `nvmmsink` can read `NvDsBatchMeta`:

```bash
meson setup builddir -Denable_deepstream_meta=true   # or cmake -DENABLE_DEEPSTREAM_META=ON
```

The flat wire format and the consumer side are **DeepStream-free** — only the
producer's `NvDsBatchMeta`→flat extraction needs this flag.

**Producer** — run inference, publish frames *and* detections:

```bash
gst-launch-1.0 \
  filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder \
  ! 'video/x-raw(memory:NVMM)' \
  ! nvstreammux batch-size=1 width=1920 height=1080 \
  ! nvinfer config-file-path=pgie.txt batch-size=1 \
  ! nvmmsink shm-name=/det export-metadata=true
```

**Consumer** — `nvmmappsrc` attaches a `GstNvmmDetMeta` to each output buffer:

```bash
gst-launch-1.0 nvmmappsrc shm-name=/det import-metadata=true ! ...
```

### Reading the detections

=== "Non-GStreamer (e.g. ROS2)"

    Read the flat records straight out of the shared segment — include
    `shm_protocol.h`, locate slot `write_idx` with `nvmm_shm_meta()`, and iterate
    `NvmmFrameMeta::objects`. No GStreamer, no DeepStream. See
    [Zero-copy IPC](ipc.md#ros2-non-gstreamer-consumers) for the frame handshake.

=== "GStreamer (non-DeepStream)"

    Downstream reads `GstNvmmDetMeta` off the buffer:

    ```c
    GstNvmmDetMeta *m = gst_buffer_get_nvmm_det_meta(buf);
    for (guint i = 0; m && i < m->num_objects; i++)
        use(&m->objects[i]);   /* bbox / class_id / confidence / tracker_id / label */
    ```

=== "DeepStream consumer"

    A consumer that is itself a DeepStream pipeline re-hydrates the flat record
    into `NvDsBatchMeta` from a pad probe after `nvstreammux` (rescaling bboxes by
    `surface_dim / infer_dim`), then `nvtracker`/`nvdsosd` work as usual.

## When *not* to use this

The side-channel adds a wire-format bump and an optional dependency. It pays off
for **infer-once / fan-out** (many consumers, inference too costly to repeat) or a
**non-GStreamer consumer** that just wants detections. If a single consumer can
re-infer cheaply, that path is simpler. For *visualization only*, burn the overlay
in with `nvdsosd` **before** `nvmmsink` and skip metadata entirely.

## Validation status

The wire format + `GstMeta` round-trip (unit), the cross-process surface path
(20/20 frames, no zero-copy regression), and the no-DeepStream graceful no-op are
validated on Orin NX (JP6). The producer-side `NvDsBatchMeta → NvmmFrameMeta`
extraction (`-Denable_deepstream_meta`) has **not** yet been run end-to-end on a
DeepStream-equipped board. See [Validation → Detection metadata side-channel](validation.md#detection-metadata-side-channel-ipc).
