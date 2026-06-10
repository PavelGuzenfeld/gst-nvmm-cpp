# Combining elements: inference graphs

The analytics elements are composable nodes; GStreamer's pipeline is the graph.
Each node passes the NVMM frame through untouched and reads or attaches
metadata, so chains and fan-outs add no pixel copies. All pipelines on this
page run as written on Jetson Orin (JP6) — substitute your own engine, labels
and input.

## Building the engines

Engines are device- and TensorRT-version-specific; build them on the target:

```bash
# Detector: YOLO11n ONNX -> fp16 engine
/usr/src/tensorrt/bin/trtexec --onnx=yolo11n.onnx \
  --saveEngine=yolo11n_fp16.engine --fp16

# Classifier: the ResNet50 that ships with TensorRT (ImageNet-1000,
# class_labels.txt included in the same directory)
/usr/src/tensorrt/bin/trtexec \
  --onnx=/usr/src/tensorrt/data/resnet50/ResNet50.onnx \
  --saveEngine=resnet50_fp16.engine --fp16
```

## Detect and overlay

The smallest useful chain: detector plus on-frame boxes.

```bash
gst-launch-1.0 filesrc location=video.h264 ! h264parse ! nvv4l2decoder \
  ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! nvmminfer engine-file=yolo11n_fp16.engine \
  ! nvmmdrawdet ! videoconvert ! autovideosink
```

## Detect → track

`nvmmtracker` assigns each detection a stable `tracker_id` (greedy IOU,
pure host). The overlay shows the id in the label (`car #4 82%`).

```bash
... ! nvmminfer engine-file=yolo11n_fp16.engine ! nvmmtracker \
    ! nvmmdrawdet ! ...
```

## Detect → track → classify (cascade)

`nvmmsecondaryinfer` crops each detection's region out of the frame on the
VIC, classifies it with a second engine, and attaches the result. Tracked
objects are re-classified only every `infer-interval` frames; a per-track
cache serves the result in between — so the cost scales with the number of
*new or stale* tracks per frame, not with detections × frames.

```bash
gst-launch-1.0 filesrc location=video.h264 ! h264parse ! nvv4l2decoder \
  ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! nvmminfer engine-file=yolo11n_fp16.engine \
  ! nvmmtracker \
  ! nvmmsecondaryinfer engine-file=resnet50_fp16.engine \
      labels-file=/usr/src/tensorrt/data/resnet50/class_labels.txt \
      output-activation=none net-scale-factor=0.003921569 \
      offsets="0.485,0.456,0.406" std-values="0.229,0.224,0.225" \
      conf-threshold=0.3 infer-interval=10 \
  ! nvmmdrawdet ! videoconvert ! autovideosink
```

The overlay label becomes `cat #1 85% [tiger cat 55%]` — detector class,
tracker id, detector confidence, then the classifier's verdict.

Normalization belongs to the classifier model, not the element: the values
above are the torchvision ImageNet convention. A model exported with YOLO-style
preprocessing needs only the default `net-scale-factor=1/255` and no
offsets/std.

`GST_DEBUG=nvmmsecondaryinfer:6` shows the cache at work:

```
frame 0:  5 objects, 5 inferred, 5 tracks cached
frame 1:  5 objects, 0 inferred, 5 tracks cached   <- served from cache
...
frame 10: 5 objects, 5 inferred, 5 tracks cached   <- interval re-inference
```

## Parallel branches: detection + optical flow, fused

`tee` duplicates the NVMM frame reference (zero-copy) into two branches that
attach different metadata; `nvmmfusion` joins them back by PTS onto one buffer
and computes per-object motion from the flow under each box. Orin only
(`nvmmofa` uses the OFA engine).

```bash
gst-launch-1.0 filesrc location=video.h264 ! h264parse ! nvv4l2decoder \
  ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! tee name=t \
  t. ! queue ! nvmminfer engine-file=yolo11n_fp16.engine ! nvmmtracker ! queue ! f.detection \
  t. ! queue ! nvmmofa ! queue ! f.flow \
  nvmmfusion name=f ! nvmmdrawdet ! videoconvert ! autovideosink
```

Downstream of fusion, one buffer carries the detection, flow and motion metas
at one PTS. Moving objects are drawn with a heavier box and a `>>` suffix;
stationary ones are not.

## The full graph

Cascade and fusion compose — the classifier is a post-fusion node:

```
                ┌─ nvmminfer → nvmmtracker ─┐
src → tee ──────┤                           ├─ nvmmfusion → nvmmsecondaryinfer → nvmmdrawdet → sink
                └─ nvmmofa ─────────────────┘
```

```bash
gst-launch-1.0 filesrc location=video.h264 ! h264parse ! nvv4l2decoder \
  ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! tee name=t \
  t. ! queue ! nvmminfer engine-file=yolo11n_fp16.engine ! nvmmtracker ! queue ! f.detection \
  t. ! queue ! nvmmofa ! queue ! f.flow \
  nvmmfusion name=f \
  ! nvmmsecondaryinfer engine-file=resnet50_fp16.engine \
      labels-file=class_labels.txt output-activation=none \
      net-scale-factor=0.003921569 offsets="0.485,0.456,0.406" \
      std-values="0.229,0.224,0.225" conf-threshold=0.3 infer-interval=10 \
  ! nvmmdrawdet ! videoconvert ! autovideosink
```

## Consuming the results programmatically

Every result rides the buffer as metadata; any downstream element or pad probe
can read it. The in-tree example element prints all of it:

```bash
... ! nvmmsecondaryinfer ... ! nvmmdetlog per-object=true ! fakesink
```

```c
GstNvmmDetMeta    *det = gst_buffer_get_nvmm_det_meta(buf);
GstNvmmClassMeta  *cls = gst_buffer_get_nvmm_class_meta(buf);   /* cascade  */
GstNvmmMotionMeta *mot = gst_buffer_get_nvmm_motion_meta(buf);  /* fusion   */
/* cls->objects[i] and mot->objects[i] describe det->objects[i] */
```

See [Creating a new element](extending.md) for writing your own consumer or
analytics node, and [Detection metadata over IPC](metadata-ipc.md) for moving
detections to another process.

## Caps and ordering rules

- Every analytics element takes `video/x-raw(memory:NVMM), format=NV12`; put
  `nvvidconv` after the decoder if the format differs.
- `nvmmtracker` must run before `nvmmsecondaryinfer` for the cache to work —
  untracked detections are re-classified every frame.
- `nvmmsecondaryinfer` is a post-fusion node; never a fusion input.
- `nvmmdrawdet` outputs system-memory RGBA. Follow it with `videoconvert`
  (display, x264) or `jpegenc`; everything before it stays NVMM.
- Branches into `nvmmfusion` need a `queue` on each side of the slow elements,
  as in the examples.
