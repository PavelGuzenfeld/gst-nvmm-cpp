# nvmmsecondaryinfer

**TensorRT cascade classifier on detected objects (Phase 3.2).** An in-place
passthrough on `video/x-raw(memory:NVMM)` that reads the upstream
`GstNvmmDetMeta` (from [`nvmminfer`](nvmminfer.md), ideally with
[`nvmmtracker`](nvmmtracker.md) ids), VIC-crops each detection's ROI out of the
NV12 frame, stretch-resizes it to the classifier's input, runs a TensorRT
engine on it, and attaches the results as a **`GstNvmmClassMeta`** sibling meta
— index-aligned with the det meta, like the motion meta. No DeepStream.

**Multi-rate by design:** a track is re-inferred only every `infer-interval`
frames; between runs the per-`tracker_id` cache serves the last result
(`fresh=0` in the meta). Untracked detections (`tracker_id == 0`) cannot be
cached and re-infer every frame — put `nvmmtracker` upstream. Cached tracks
unseen for `max-track-age` frames are dropped.

Preprocess is the proven `nvmminfer` device-side path (VIC crop+resize → NPP
planarize/convert), extended with optional per-channel normalization:
`y = (x * net-scale-factor - offsets[c]) / std-values[c]`, all in the engine's
channel order. NPP runs through the `_Ctx` stream API so the element never
touches the process-global NPP stream that `nvmminfer` binds to its own stream.

The engine must be a classification head: input `1x3xHxW` FP32, output a
per-class score vector (`[1,C]`, `[1,C,1,1]` or `[C]`) FP32. ROIs are inferred
sequentially (batch-1 engine); batching is a deferred optimization — with the
interval cache the per-frame ROI count is small.

## Properties

| Property | Type | Default | Notes |
|---|---|---|---|
| `engine-file` | string | — | Serialized TensorRT classifier engine (required; build with `trtexec`) |
| `labels-file` | string | — | One class label per line (else `class<N>`) |
| `infer-interval` | uint | `10` | Re-classify a track every N frames (1 = every frame) |
| `max-track-age` | uint | `60` | Drop a cached track unseen this many frames |
| `min-roi-size` | uint | `16` | Skip detections narrower/shorter than this (surface px) |
| `net-scale-factor` | double | `1/255` | Pixel scale applied during preprocess |
| `offsets` | string | — | Per-channel mean `"v0,v1,v2"` subtracted after scaling (engine channel order) |
| `std-values` | string | — | Per-channel std `"v0,v1,v2"` divided after offsets |
| `color-order` | enum | `rgb` | `rgb` / `bgr` — channel order the engine expects |
| `output-activation` | enum | `softmax` | `softmax` over logits, or `none` if the engine already outputs probabilities |
| `conf-threshold` | double | `0.0` | Minimum top-1 score to attach (and cache) a result |

## Validated on Orin (JP6, TRT 10.3)

ResNet50 fp16 (built from the `/usr/src/tensorrt/data/resnet50/ResNet50.onnx`
that ships with TensorRT, torchvision normalization), behind the yolo11n
detector + IOU tracker: the COCO `cat` detection classifies as **"tiger cat"**
on TensorRT's `tabby_tiger_cat.jpg`, the bus in `bus.jpg` as **"recreational
vehicle"**; the interval cache shows `5 inferred` on frame 0 and 10, `0
inferred` in between. ~2.1 ms GPU per ROI (trtexec fp16 measurement).

```bash
gst-launch-1.0 ... ! nvmminfer engine-file=yolo11n_fp16.engine ! nvmmtracker \
  ! nvmmsecondaryinfer engine-file=resnet50_fp16.engine \
      labels-file=class_labels.txt output-activation=none \
      net-scale-factor=0.003921569 offsets="0.485,0.456,0.406" \
      std-values="0.229,0.224,0.225" conf-threshold=0.3 \
  ! nvmmdrawdet ! ...   # labels now show "cat #1 85% [tiger cat 55%]"
```

[`nvmmdrawdet`](nvmmdrawdet.md) renders the classification as a
`[label conf%]` suffix in the box label.
