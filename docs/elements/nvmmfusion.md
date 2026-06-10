# nvmmfusion

**Join the detector and optical-flow branches by PTS** — a `GstAggregator` with
two named sink pads that unions both branch metas onto a single output buffer:

- `detection` — NVMM NV12 carrying `GstNvmmDetMeta`
  ([`nvmminfer`](nvmminfer.md) → [`nvmmtracker`](nvmmtracker.md))
- `flow` — NVMM NV12 carrying `NvmmOpticalFlowMeta`
  ([`nvmmofa`](nvmmofa.md))

Per output frame the **detection buffer is the carrier** (zero-copy — same
`NvBufSurface`, no pixel work, no CUDA) and the flow meta is added onto it, so a
single buffer downstream holds **both** metas at one PTS. PTS is the join key:
`tee` copies timestamps verbatim so the queue heads pair naturally; on a mismatch
(a branch dropped a frame) the older head is dropped and the join resyncs.

Either branch reaching EOS ends the fused stream (inner-join semantics).

!!! note "Phase 2: structural join only"
    Fusion co-locates the data; it does not yet *use* it. The cross-modal payoff
    — marking which detected boxes are actually moving from the flow field —
    is Phase 3.

```bash
gst-launch-1.0 filesrc location=video.h264 ! h264parse ! nvv4l2decoder \
  ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! tee name=t \
  t. ! queue ! nvmminfer engine-file=yolo.engine ! nvmmtracker ! queue ! f.detection \
  t. ! queue ! nvmmofa ! queue ! f.flow \
  nvmmfusion name=f ! fakesink
# GST_DEBUG=nvmmfusion:6 logs:  fused: 8 detection(s) + flow=yes on one buffer
```

A downstream consumer verifies the union by reading both metas off one buffer:
`gst_buffer_get_nvmm_det_meta(buf)` **and**
`gst_buffer_get_nvmm_optical_flow_meta(buf)` are both non-NULL after fusion.
