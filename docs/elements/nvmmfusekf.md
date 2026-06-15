# nvmmfusekf

**Master constant-velocity Kalman filter that fuses the visual tracker with the
detector.** A pure-host, in-place passthrough on `video/x-raw(memory:NVMM),
format=NV12` that reads both the `GstNvmmTrackMeta` (from
[`nvmmsamurai`](nvmmsamurai.md)) and the `GstNvmmDetMeta` (from
[`nvmminfer`](nvmminfer.md)) and maintains a single fused track box, written
back as the authoritative `GstNvmmTrackMeta` that [`nvmmdrawdet`](nvmmdrawdet.md)
renders.

It touches no pixels and uses no CUDA, so it **builds and runs on the x86
host/CI build** — the Kalman core ([`kalman_box`](nvmmtracker.md)) is
dependency-free and unit-tested directly.

How it fuses each frame:

- **SAMURAI drives** — its box is the primary measurement.
- **YOLO refines** — a detection of `target-class` above `det-conf` is fused
  only if its center is within `gate-threshold` pixels of the prediction
  (Euclidean gate, not Mahalanobis: measurement noise scales with box size, so
  for a few-pixel target the Mahalanobis gate degenerates).
- **On loss** — after `max-lost` frames with no measurement the track is marked
  lost and the element emits an upstream `nvmm-reseed` custom event back to
  `nvmmsamurai` to re-acquire, rate-limited by `reseed-cooldown`.

Sink/Src: `video/x-raw(memory:NVMM), format=NV12` → same (passthrough + fused track meta).

## Properties

| Property | Type | Default | Notes |
|---|---|---|---|
| `target-class` | int | `0` | YOLO class id to fuse |
| `det-conf` | double (0–1) | `0.25` | Min YOLO confidence to fuse |
| `gate-threshold` | double (px) | `100` | Max YOLO-to-prediction center distance to fuse |
| `max-lost` | int | `30` | Consecutive no-measurement frames before the track is lost |
| `reseed-cooldown` | int | `15` | Frames between successive upstream reseed emissions |

## Per-frame CSV

Set `NVMMFUSEKF_CSV=<path>` in the environment to dump a per-frame row
(`frame,valid,left,top,width,height,score,yolo_fused`) for offline analysis.
This is a data sink, independent of the GStreamer log.

## Logging

```bash
GST_DEBUG=nvmmfusekf:5   # state changes: track lost + reseed emitted
GST_DEBUG=nvmmfusekf:6   # + per-frame fuse decision (sam/yolo/updated/lost)
```

## Example

```bash
... ! nvmmsamurai engine-dir=/o/trt consts-file=/o/trt/samurai_consts.bin ! \
    nvmmfusekf target-class=0 gate-threshold=100 max-lost=30 ! \
    nvmmdrawdet ! ...
```
