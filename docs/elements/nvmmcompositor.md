# nvmmcompositor

VIC-composited multi-input NVMM mixer: places each input frame into a rectangle
of a single output frame — **mosaics, side-by-side, picture-in-picture, overlays**
— all GPU-to-GPU (no CPU on the data path). Each request sink pad is blitted into
the output via `NvBufSurfTransform` (Tegra VIC) with a `CROP_DST` rectangle.

Built on `GstAggregator` (not `GstVideoAggregator`, whose `GstVideoFrame`
mapping does not understand NVMM memory). The output frame is allocated from this
suite's NVMM allocator, so it stays zero-copy into `nvmmsink`, `nvmmconvert`, or
the IPC pool downstream.

Sink (request `sink_%u`) / src caps: `video/x-raw(memory:NVMM)`, formats
`{NV12, RGBA}`, up to 8192×8192.

## Properties

### Element

| Property | Type | Default | Description |
|---|---|---|---|
| `width` | int | 1280 | Output frame width (pixels) |
| `height` | int | 720 | Output frame height |

### Request pad (`sink_%u`)

| Property | Type | Default | Description |
|---|---|---|---|
| `xpos` | int | 0 | Left offset of this input in the output |
| `ypos` | int | 0 | Top offset of this input in the output |
| `width` | int | 0 | Tile width (0 = fill to output right edge) |
| `height` | int | 0 | Tile height (0 = fill to output bottom edge) |

Each pad's frame is scaled by the VIC to fit its `width`×`height` rectangle at
`(xpos, ypos)`. The default `0` width/height fills the remaining output, so a
single pad with default placement scales its input to the full output frame.

!!! note "Full-coverage assumption"
    Each pad writes only its own rectangle; regions no pad covers are left
    **undefined** (the output buffer is not cleared). Lay out the pads so they
    tile the entire output (mosaic), or put a full-frame background pad behind
    the others for partial layouts (PiP/overlay).

## Examples

Pad properties are set on the request pads. With `gst-launch-1.0` you request the
pads by linking to `sink_0`/`sink_1` and set their properties via the pad — or
drive it from code (see the Python snippet below).

```bash
# Side-by-side: two 640-wide tiles into a 1280x720 output.
# (gst-launch links the pads; placement defaults make sink_0 fill the left and
#  sink_1 the right when widths are set programmatically — see Python below.)
gst-launch-1.0 -e \
  nvmmcompositor name=c width=1280 height=720 \
  c. ! 'video/x-raw(memory:NVMM)' ! nvmmsink shm-name=/mosaic \
  videotestsrc pattern=smpte ! 'video/x-raw,width=640,height=720' ! \
    nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! c.sink_0 \
  videotestsrc pattern=ball  ! 'video/x-raw,width=640,height=720' ! \
    nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! c.sink_1
```

```python
# Side-by-side placement set on the pads (Python/gi).
import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst
Gst.init(None)
p = Gst.parse_launch(
    "nvmmcompositor name=c width=1280 height=720 "
    "c. ! video/x-raw(memory:NVMM) ! nvvidconv ! jpegenc ! filesink location=mosaic.jpg "
    "videotestsrc num-buffers=1 pattern=smpte ! video/x-raw,width=640,height=720 ! "
    "  nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! c.sink_0 "
    "videotestsrc num-buffers=1 pattern=ball ! video/x-raw,width=640,height=720 ! "
    "  nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! c.sink_1")
c = p.get_by_name("c")
c.get_static_pad("sink_0").set_property("xpos", 0);   c.get_static_pad("sink_0").set_property("width", 640)
c.get_static_pad("sink_1").set_property("xpos", 640); c.get_static_pad("sink_1").set_property("width", 640)
p.set_state(Gst.State.PLAYING)
```

## Relationship to DeepStream `nvstreammux`/`nvmultistreamtiler`

NVIDIA's tiling lives in DeepStream (`nvmultistreamtiler`) and `nvcompositor`.
`nvmmcompositor` gives the common mosaic/PiP case as a standalone, open-source
element with no DeepStream dependency, composited on the VIC and integrated with
this suite's zero-copy NVMM allocator and cross-process IPC.
