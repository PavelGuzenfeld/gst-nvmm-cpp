# nvmmconvert

VIC-accelerated 2D transform on NVMM surfaces: **crop, scale, color-format
convert, rotate/flip**, all GPU-to-GPU (no CPU on the data path). Runs on the
Tegra VIC via `NvBufSurfTransform`.

Sink/src caps: `video/x-raw(memory:NVMM)`, formats `{NV12, RGBA, I420, BGRA}`,
up to 8192×8192.

## Properties

| Property | Type | Default | Description |
|---|---|---|---|
| `crop-x` | uint | 0 | Source crop X offset (pixels) |
| `crop-y` | uint | 0 | Source crop Y offset |
| `crop-w` | uint | 0 | Source crop width (0 = full) |
| `crop-h` | uint | 0 | Source crop height (0 = full) |
| `flip-method` | enum | 0 | 0=none, 1=rotate-90 (CW), 2=rotate-180, 3=rotate-270 (CCW), 4=horizontal-flip, 5=transpose, 6=vertical-flip, 7=inverse-transpose |
| `interpolation` | enum | 6 | VIC scaling filter: 0=nearest, 1=bilinear, 2=5-tap, 3=10-tap, 4=smart, 5=nicest, 6=default |
| `compute-mode` | enum | 0 | Engine that runs the transform: 0=default (VIC on Tegra), 1=gpu, 2=vic |

Output dimensions come from the downstream caps. `rotate-90`/`270` and
`transpose`/`inverse-transpose` swap width and height.

## Examples

```bash
# Scale 1080p -> 480p with the 5-tap filter
... ! nvmmconvert interpolation=5-tap ! 'video/x-raw(memory:NVMM),width=640,height=480' ! ...

# Force the GPU engine instead of the VIC (e.g. when the VIC is saturated)
... ! nvmmconvert compute-mode=gpu ! 'video/x-raw(memory:NVMM),width=640,height=480' ! ...

# Crop a region
... ! nvmmconvert crop-x=100 crop-y=50 crop-w=800 crop-h=600 ! \
      'video/x-raw(memory:NVMM),width=800,height=600' ! ...

# Rotate 90 CW (640x480 -> 480x640)
... ! nvmmconvert flip-method=rotate-90 ! 'video/x-raw(memory:NVMM),width=480,height=640' ! ...
```

## Relationship to `nvvidconv`

`nvmmconvert` overlaps NVIDIA's stock `nvvidconv`; the value here is being
open-source and integrated with this suite's NVMM allocator/IPC pool — not new 2D
capability.

## DeepStream interop

`nvmmconvert` emits `video/x-raw(memory:NVMM)` backed by `NvBufSurface` (NV12 /
RGBA), the same memory DeepStream consumes — so it drops in as an **open-source
preprocessing stage** (crop / scale / rotate on the VIC) ahead of inference. You
*can* feed DeepStream; you aren't *required* to.

Put it **before** `nvstreammux` — `nvinfer` runs on the batched mux output, so any
per-stream ROI crop or resize happens upstream of the mux:

!!! success "Verified on Orin"
    Run on Jetson Orin NX, JetPack 6 (L4T R36.4.3), inside the
    `nvcr.io/nvidia/deepstream:7.1-samples` container — the elements built against
    DeepStream's own `NvBufSurface`, all four `nvmm*` plugins co-registered with
    `nvinfer`/`nvstreammux`/`nvdsosd`. The pipeline below (TrafficCamNet detector)
    ran end-to-end to EOS with real inference (headless test terminated in
    `fakesink` in place of `nv3dsink`). `nvstreammux` property names and the
    `nvinfer` config layout vary by DS release — adjust `config-file-path` to your
    install.

```bash
# Crop a region of interest + scale, then batch and infer
gst-launch-1.0 \
  filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder \
  ! 'video/x-raw(memory:NVMM)' \
  ! nvmmconvert crop-x=320 crop-y=180 crop-w=1280 crop-h=720 \
  ! 'video/x-raw(memory:NVMM),width=640,height=640,format=NV12' \
  ! nvstreammux batch-size=1 width=640 height=640 \
  ! nvinfer config-file-path=pgie.txt batch-size=1 \
  ! nvvideoconvert ! nvdsosd ! nv3dsink
```

!!! tip "Set `nvinfer batch-size=1` for a single stream"
    The stock DeepStream `config_infer_primary.txt` defaults to a **batch-30** INT8
    engine (built for the multi-stream `deepstream-app`). Building that engine needs
    far more GPU memory than one stream warrants and can OOM on an 8 GB Orin NX.
    Overriding `nvinfer batch-size=1` builds a small single-batch engine instead.

`nvmmconvert` is interchangeable with `nvvideoconvert` for this 2D step; the value
is that it shares this suite's NVMM allocator/IPC pool, so a cropped frame can be
published over [IPC](../ipc.md) without leaving the GPU.
