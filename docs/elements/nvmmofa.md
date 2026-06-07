# nvmmofa

Dense **optical flow on the Orin OFA engine** (Optical Flow Accelerator) via VPI,
straight from NVMM — no CPU on the frame path. `nvmmofa` is a passthrough
transform: the NV12 frame flows downstream **unchanged and zero-copy**, and for
every consecutive pair of frames it runs VPI dense optical flow on the dedicated
OFA hardware and attaches the motion-vector field as an
[`NvmmOpticalFlowMeta`](#flow-metadata) for in-process analytics.

!!! warning "Orin only"
    OFA is dedicated hardware on Orin; **Xavier has no OFA** and cannot run this
    element (documented N/A, like NVENC's Xavier-only encode). Built only where
    VPI is present.

Sink/src caps: `video/x-raw(memory:NVMM), format=NV12`, 32×32 … 8192×8192. The
input must be **block-linear** NV12 — exactly what `nvvidconv` emits — which is
what OFA requires (a pitch-linear surface is rejected).

## Properties

| Property | Type | Default | Values |
|---|---|---|---|
| `grid-size` | enum | `4` | `1` (dense), `2`, `4`, `8` — pixels per motion vector |
| `quality` | enum | `medium` | `low`, `medium`, `high` |

The output flow field is `ceil(width/grid_size) × ceil(height/grid_size)` cells.
Smaller grids are denser but cost more; `grid-size=1` gives a per-pixel field.

## Flow metadata

`nvmmofa` attaches an `NvmmOpticalFlowMeta` to each output buffer (from the second
frame on — the first has no predecessor). OFA only writes a VPI-native `2S16_BL`
image (a wrapped NVMM surface is rejected), so the field is locked to host and
copied — tightly packed — into a host buffer the meta carries:

- `mv_width × mv_height` cells, each two `int16` (dx, dy) in **S10.5** fixed point
  (divide by 32 for pixels).
- The video frame itself is still passed through zero-copy; only the small flow
  field (e.g. 160×120 → ~75 KB) is copied out.

The meta is **in-process only** — it does not cross the `nvmmsink`→`nvmmappsrc`
zero-copy IPC boundary (GstMeta is per-process). For cross-process flow you would
serialize it through the shm protocol instead.

## nvmmflowstats — example consumer

`nvmmflowstats` is a sink that reads the flow meta and reports per-frame mean/max
motion-vector magnitude (in pixels) — a worked example of consuming the metadata.

```bash
# Dense optical flow on a live camera feed, with flow magnitude logged
gst-launch-1.0 -e \
  nvarguscamerasrc ! 'video/x-raw(memory:NVMM),format=NV12,width=1280,height=720' \
  ! nvmmofa grid-size=4 quality=medium ! nvmmflowstats

# Synthetic check (moving ball)
gst-launch-1.0 -e \
  videotestsrc num-buffers=30 pattern=ball ! 'video/x-raw,width=640,height=480' \
  ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! nvmmofa ! nvmmflowstats
# [nvmmflowstats] frame N: 160x120 grid=4 mean=5.11 px max=12.08 px
```

Because the frame passes through untouched, you can also tee it to an encoder or
display while `nvmmflowstats` (or your own meta consumer) reads the flow.

## Relationship to DeepStream

DeepStream exposes OFA via `nvof` / `nvofvisual`. `nvmmofa` gives the same OFA
optical flow as a standalone, open-source element with no DeepStream dependency,
keeping the frame NVMM-native and zero-copy.
