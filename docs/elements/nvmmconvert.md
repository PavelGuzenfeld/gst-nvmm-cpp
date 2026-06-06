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
