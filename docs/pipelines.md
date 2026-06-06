# Pipeline examples

> `nvmmconvert`/`nvmmappsrc` work in `video/x-raw(memory:NVMM)`. Cross the NVMM
> boundary with `nvvidconv` (VIC) — not `videoconvert` (CPU) — or feed a hardware
> consumer (`nvv4l2h264enc`, `nvjpegenc`) that takes NVMM directly.

## Decode and scale (Jetson)

```bash
gst-launch-1.0 \
  filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder \
  ! 'video/x-raw(memory:NVMM)' \
  ! nvmmconvert \
  ! 'video/x-raw(memory:NVMM),width=640,height=480' \
  ! nvmmsink shm-name=/camera_feed
```

## Crop a region of interest

```bash
gst-launch-1.0 ... ! nvmmconvert crop-x=100 crop-y=50 crop-w=800 crop-h=600 ! ...
```

## Flip / rotate

```bash
gst-launch-1.0 ... ! nvmmconvert flip-method=rotate-180 ! ...      # 180°
gst-launch-1.0 ... ! nvmmconvert flip-method=horizontal-flip ! ... # mirror
gst-launch-1.0 ... ! nvmmconvert flip-method=rotate-90 ! \
  'video/x-raw(memory:NVMM),width=480,height=640' ! ...            # 90° (swaps dims)
```

## Choose a scaling filter

```bash
gst-launch-1.0 ... ! nvmmconvert interpolation=5-tap ! \
  'video/x-raw(memory:NVMM),width=640,height=480' ! ...
```

## Inter-process video sharing

**Process A** (producer — `nvv4l2decoder` emits NVMM, which `nvmmsink` takes directly):

```bash
gst-launch-1.0 ... ! nvv4l2decoder ! 'video/x-raw(memory:NVMM)' ! nvmmsink shm-name=/video_feed
```

**Process B** (consumer — `nvvidconv` brings NVMM to system memory for display):

```bash
gst-launch-1.0 nvmmappsrc shm-name=/video_feed ! nvvidconv ! videoconvert ! autovideosink
```

## Multi-camera fan-out to multiple consumers

The motivating use case: one producer stream published once, consumed by as many
processes as you want, all staying on the GPU. Each `nvmmsink` pool is written
once per frame; every consumer just imports the fds and reads in place — adding a
second or third consumer adds **no** extra GPU copy.

**Producer** (N ZED cameras → N shm segments, one process):

```bash
gst-launch-1.0 -e \
  zedsrc camera-sn=<SN1> camera-resolution=4 camera-fps=120 stream-type=7 \
    ! 'video/x-raw(memory:NVMM),format=NV12' ! queue ! nvmmsink shm-name=/cam1 \
  zedsrc camera-sn=<SN2> camera-resolution=4 camera-fps=120 stream-type=7 \
    ! 'video/x-raw(memory:NVMM),format=NV12' ! queue ! nvmmsink shm-name=/cam2 \
  zedsrc camera-sn=<SN3> camera-resolution=4 camera-fps=120 stream-type=7 \
    ! 'video/x-raw(memory:NVMM),format=NV12' ! queue ! nvmmsink shm-name=/cam3
```

**Consumers** (each instance attaches to all three segments and records — launch
in as many shells as you want; the hardware encoder reads NVMM directly, no copy):

```bash
timeout -s INT 120 gst-launch-1.0 -e \
  nvmmappsrc shm-name=/cam1 do-timestamp=true is-live=true \
    ! 'video/x-raw(memory:NVMM),format=NV12' \
    ! nvv4l2h264enc bitrate=20000000 ! h264parse ! qtmux \
    ! filesink location=/tmp/out_cam1.mp4 sync=false async=false \
  nvmmappsrc shm-name=/cam2 do-timestamp=true is-live=true \
    ! 'video/x-raw(memory:NVMM),format=NV12' \
    ! nvv4l2h264enc bitrate=20000000 ! h264parse ! qtmux \
    ! filesink location=/tmp/out_cam2.mp4 sync=false async=false \
  nvmmappsrc shm-name=/cam3 do-timestamp=true is-live=true \
    ! 'video/x-raw(memory:NVMM),format=NV12' \
    ! nvv4l2h264enc bitrate=20000000 ! h264parse ! qtmux \
    ! filesink location=/tmp/out_cam3.mp4 sync=false async=false
```

The pool's per-slot `ref_counts` handle the fan-out: each consumer atomically
increments its slot's count on read and decrements when done; the producer reuses
a slot only once its count is back to 0. Buffers stay GPU-resident end to end.

## ROS2 / non-GStreamer bridge

See [Zero-copy IPC](ipc.md#ros2-non-gstreamer-consumers) for the wire protocol and
handshake a non-GStreamer consumer follows (`shm_protocol.h` + `SCM_RIGHTS`).
