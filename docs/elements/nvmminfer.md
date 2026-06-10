# nvmminfer

**TensorRT object detection straight from NVMM — no DeepStream.** `nvmminfer` is
an in-place transform: the NV12 frame passes through unchanged/zero-copy, and per
frame it runs VIC+NPP preprocessing → a TensorRT engine → a YOLO decode+NMS
parser, attaching the detections as a [`GstNvmmDetMeta`](../metadata-ipc.md)
(boxes in original-frame pixel space).

!!! warning "Jetson only"
    Built only where TensorRT + CUDA + NPP are present (skip-on-host, like
    `nvmmofa`). Validated on Orin JP6 (TensorRT 10.3, CUDA 12.x).

Sink/src caps: `video/x-raw(memory:NVMM), format=NV12`.

## Pipeline (device-only preprocess)

1. **VIC** (`NvBufSurfTransform`) letterboxes NV12 → RGBA at network size into a
   VIC-native surface; **EGL/CUDA interop** exposes it to CUDA, which works for
   every source memtype (decoder, `nvvidconv`, `imagefreeze`).
2. **NPP** splits planes, converts to float and normalizes into the input tensor.
3. **TensorRT** `enqueueV3` on the bound device buffers.
4. The host **YOLO parser** (pure CPU, unit-tested on x86 CI) decodes the
   `[1, 4+classes, proposals]` head, applies per-class NMS, and un-maps boxes from
   letterbox space back to frame pixels.

No host round-trip for pixels: surface → input tensor is one device pass.

## Properties

| Property | Type | Default | Notes |
|---|---|---|---|
| `engine-file` | string | — | Prebuilt TensorRT `.engine` (build on-target with `trtexec`) |
| `conf-threshold` | double | `0.25` | Minimum class score |
| `measure-latency` | bool | `false` | Log per-stage latency + FPS every 60 frames |

The engine must have FP32 I/O bindings, exactly one input (`1x3xHxW`) and one
channels-first YOLO output head; anything else is rejected loudly at start.
(`trtexec --fp16` keeps FP32 I/O, so fp16 engines work.)

## Validation

- **Golden test** (`scripts/nvmminfer_golden_test.sh`): cross-checks the TRT
  output box-by-box against an independent onnxruntime fp32 reference on the same
  ONNX — IoU ≥ 0.97, confidence delta ≤ 0.05 on the reference image.
- Measured on Orin (1080p, YOLO11n fp16): preprocess 7 ms, inference 13 ms,
  copy+parse 2 ms — **~23 ms/frame, ~43 FPS** detector capability.

```bash
gst-launch-1.0 filesrc location=video.h264 ! h264parse ! nvv4l2decoder \
  ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! nvmminfer engine-file=yolo11n_fp16.engine \
  ! nvmmdrawdet ! videoconvert ! autovideosink
```
