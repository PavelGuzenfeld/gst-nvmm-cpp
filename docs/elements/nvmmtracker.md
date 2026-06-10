# nvmmtracker

**Multi-object tracking on detection metadata — pure host, no CUDA.** An in-place
passthrough on `video/x-raw(memory:NVMM)` that reads the `GstNvmmDetMeta`
(e.g. from [`nvmminfer`](nvmminfer.md)) and assigns each detection a stable
`NvmmDetObject.tracker_id` by greedy per-class **IOU matching** against
prior-frame tracks. Unmatched detections start new tracks; tracks unseen for
`max-age` frames expire.

Pixels are untouched, so the element **builds and runs on the x86 host/CI build**
too — the `nvmm::Tracker` core is dependency-free and unit-tested directly
(matching, ID persistence, class separation, expiry, reset).

## Properties

| Property | Type | Default | Notes |
|---|---|---|---|
| `iou-threshold` | double | `0.3` | Minimum IOU (same class) to continue a track |
| `max-age` | int | `30` | Frames a track survives with no match |

IDs are 1-based, stable within a stream session, and `0` means "no tracker"
(the meta's default). Greedy-IOU is the Phase-2 algorithm; a motion model
(SORT/Kalman) is a later internal change behind the same class boundary.

```bash
... ! nvmminfer engine-file=yolo.engine ! nvmmtracker iou-threshold=0.3 \
    ! nvmmdrawdet ! ...   # labels now show "car #4 82%"
```
