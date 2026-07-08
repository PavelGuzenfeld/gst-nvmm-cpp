# nvmmsamurai

**Single-object visual tracker (SAM2.1 / SAMURAI) on TensorRT.** An in-place
passthrough on `video/x-raw(memory:NVMM), format=NV12` that runs the
[SAMURAI](https://yangchris11.github.io/samurai/) motion-aware variant of
Meta's **SAM 2.1** as five TensorRT engines, tracks one target across the
stream, and attaches a `GstNvmmTrackMeta` (box + object score + valid flag) to
each buffer. Pixels are untouched.

The tracker seeds itself from an upstream detector's `GstNvmmDetMeta` (e.g.
[`nvmminfer`](nvmminfer.md) running YOLO) — picking the highest-confidence
detection of `target-class`, or the one nearest frame center with
`seed-prefer-center` — or from a fixed `seed-roi` you supply directly. Once
seeded it runs SAM2.1 every frame; between full inferences it can coast on a
constant-velocity Kalman step (`max-kf`) for throughput. It listens for an
upstream `nvmm-reseed` event (emitted by [`nvmmfusekf`](nvmmfusekf.md) on
track loss) to re-acquire.

Sink/Src: `video/x-raw(memory:NVMM), format=NV12` → same (passthrough + track meta).

## Requirements

This element is **Jetson-only** — it needs TensorRT + CUDA at runtime. It loads
five engines and one constants file from `engine-dir`/`consts-file`:

- `image_encoder_bplus_512.engine`, `prompt_encoder.engine`,
  `mask_decoder.engine`, `memory_encoder.engine`, `memory_attention.engine`
- `samurai_consts.bin` — learned constants packed out of the model

These are built from the public SAM 2.1 `base_plus` checkpoint and the public
SAMURAI config; the export + build + pack chain is documented in
[Building the SAMURAI engines](../building-engines.md). Nothing model-specific is
shipped in this repo.

## Properties

| Property | Type | Default | Notes |
|---|---|---|---|
| `engine-dir` | string | — | Directory with the 5 SAMURAI `.engine` files (required) |
| `consts-file` | string | — | Packed learned constants, `samurai_consts.bin` (required) |
| `crop-size` | int (64–2048) | `512` | Square encoder input size; the engine set must be exported to match (see note) |
| `max-kf` | int (0–30) | `2` | Max consecutive Kalman-only frames between full inferences |
| `kf-score-weight` | double (0–1) | `0.25` | Stable-regime score: `w·kf_iou + (1−w)·iou` |
| `iou-threshold` | double (0–1) | `0.5` | Min selected-candidate IoU to accept a Kalman update |
| `kf-min-area` | double | `25` | Min Kalman box area (px²) to accept an update |
| `stable-frames-threshold` | int | `10` | Frames of stable IoU before the "stable" regime |
| `target-class` | int | `0` | YOLO class id to seed/track |
| `seed-conf` | double (0–1) | `0.25` | Min YOLO confidence to auto-seed |
| `seed-prefer-center` | bool | `false` | Seed the detection nearest frame center (vs most confident) |
| `seed-roi` | string | — | Force the initial seed at `"x,y,w,h"` (pixels), bypassing YOLO |
| `seed-delay` | uint | `0` | Don't auto-seed before this frame (skip an unstable lead-in) |
| `gmc` | bool | `false` | Camera-motion compensation — shift KF/crop to cancel camera translation (handheld / moving camera) |

!!! note "Non-default `crop-size`"
    `crop-size` drives the encoder token grid (`(crop/16)²`) at runtime, so a
    non-512 crop is supported — but the five engines must be exported/built at
    that size (`image_encoder` is spatial-dynamic and rebuilds at any size; the
    others are baked per crop — see [building engines](../building-engines.md)).
    Tracking *quality* is unchanged from the 512 default only if the target
    stays in the crop: on a moving camera enable `gmc`, and pair the tracker
    with a detector so it re-seeds. A frame whose mask-decoder objectness comes
    back non-finite (target fully out of crop) coasts on the last box and skips
    the memory update — it never corrupts tracker state.

## Logging

Categories are layered so a normal run is silent:

```bash
GST_DEBUG=nvmmsamurai:4   # lifecycle: engines loaded, init, seed acquired
GST_DEBUG=nvmmsamurai:5   # + state changes: reseed requests, seed-condition eval
GST_DEBUG=nvmmsamurai:6   # + per-frame: track box, camera-motion shift
```

## Example

```bash
... ! nvmminfer engine-file=yolo.engine ! \
    nvmmsamurai engine-dir=/o/trt consts-file=/o/trt/samurai_consts.bin \
                max-kf=2 seed-prefer-center=true ! \
    nvmmfusekf target-class=0 ! nvmmdrawdet ! ...
```

See the [tracker pipeline walkthrough](../tracker-pipeline.md) for the full
detector → tracker → fusion → overlay → encode graph.
