# B5 — `nvmminfer` & the NVMM inference-graph family (design)

> Status: **Phases 0–2 SHIPPED, Orin-validated.** Phase 0 gate cleared; Phase 1
> (`nvmminfer` detector + `nvmmdrawdet` overlay + golden validation) merged via
> PRs #37–#40; Phase 2 (share-capable `nvmmalloc` #44, `nvmmtracker` #45/#46,
> `nvmmfusion`) complete. Phase 3 (cascade + the "mark moving objects" payoff)
> is next. Supersedes the single-element framing in `HW_ACCEL_EXPLORATION.md`
> (B5). Target: Orin first.

## What this is

Not "one element." A small **family of composable GStreamer node-elements** that
let a user wire an arbitrary **inference graph** — parallel models on the same
frame (b) *and* cascades that re-infer on another node's output (c) — with no
DeepStream dependency. GStreamer itself is the graph engine; we add nodes.

Driving use-case (the first real graph):

```
src → tee ─┬─ nvmminfer(detector) → nvmmtracker ─┐
           └─ nvmmofa → nvmmflowstats ───────────┴→ nvmmfusion → nvmmsecondaryinfer(classifier) → consumer
```

- **Fast tracker** = `nvmminfer` (TRT detector, every frame) + **`nvmmtracker`**
  (new, *non-TRT* IOU/Kalman ID assignment; fills `NvmmDetObject.tracker_id`).
- **Movement detector** = **reuse shipped B3** `nvmmofa → nvmmflowstats` (Orin OFA).
  Essentially free on Orin; no new TRT work.
- **Slower classifier** = **`nvmmsecondaryinfer`** (TRT cascade, type c): reads
  upstream object meta, crops + batches ROIs, re-infers on an *interval*, caches
  results per `tracker_id`. A **post-fusion** node — never a fusion input.
- **`nvmmfusion`** = `GstAggregator` (sibling of `nvmmcompositor`): joins the
  every-frame branches by PTS, unions their metas onto one buffer, pushes it.

The genuinely new **TRT** work is only the detector + the secondary classifier.

## Decisions (resolved)

| # | Decision | Choice |
|---|---|---|
| Intent | Portfolio vs production | **Both** — one real graph, built upstream-quality |
| Topology | Graph in pipeline vs orchestrator element | **Graph = the GStreamer pipeline**; composable node-elements; user wires it |
| Fan-out | `tee` zero-copy? | `tee`'s push is zero-copy, **but** per-branch meta-attach forces `make_writable` → copy. Today `nvmmalloc` sets `GST_MEMORY_FLAG_NO_SHARE` with **no copy fn** → broken/expensive. Must fix. |
| Zero-copy fix | How to keep fan-out zero-copy | **(a) Make `nvmmalloc` share-capable** — implement `mem_share` (full-surface only), drop `NO_SHARE`. `make_writable` then does a *shallow* copy: new `GstBuffer`+meta list, **same `NvBufSurface` by ref**. Safe — inference reads pixels only. Fallbacks documented: (b) out-of-band results keyed by frame-number, (c) shared single batch-meta (DeepStream's trick). |
| Meta model | Hierarchy now vs later | **Sibling metas, deferred.** Phase 1 ships on existing `det_meta`, **zero new meta types**. Tensor/classifier siblings added when a concrete model demands them. Future tensor meta must **ref, not deep-copy** device data on copy-transform — template already exists: `nvmm_optical_flow_meta` holds `GstMemory*` by ref. |
| Topology/rates | Gated vs parallel | **Parallel + fused (b)**, Orin first. Multi-rate confined to the classifier (interval + per-track cache); fusion stays same-rate. |
| Preprocess | Where + how | **Inside `nvmminfer`** (like `nvinfer` — "just give it a frame"; do **not** revive the unshipped A.4 `NORMALIZE`). Mechanism: reuse proven **VIC** (`nvmm_transform`/`NvBufSurfTransform`) for resize + NV12→RGBA, then **NPP** (`nppi`) for normalize+planarize+cast → device tensor → TRT bind. NPP over a custom kernel: no `nvcc`/`.cu` added, ships with CUDA. Props: `network-size`, `mean`/`std` (or `net-scale-factor`), `color-order`. |
| "Zero-copy" honesty | What it means for inference | **No host/CPU round-trip** — *not* binding the camera surface directly. Surface→input-tensor is one device pass (VIC→CUDA, one sync/frame, like DeepStream). Project's IPC/`tee` zero-copy claims unaffected. |
| Fusion compute | Phase 1 vs later | **(a) structural join only** in Phase 1/2 (co-locate `det_meta` + flow meta by PTS). The cross-modal "mark moving objects" payoff lands in **Phase 3** with the fusion-result sibling meta. |
| Join key | How fusion aligns branches | **PTS** — `tee` copies timestamps verbatim, so branch PTS are identical. *(As-built correction: plain `GstAggregator` does NOT align by timestamp — only `GstVideoAggregator` does — so `nvmmfusion` pairs queue heads and, on a PTS mismatch, drops the older head to resync.)* Add a frame-id stamper only if hardware shows PTS collisions/reorder. Fusion latency = slower branch (OFA); the timeout emit-with-flag is deferred until a live use-case needs it. |
| CUDA/TRT | New deps | B5 is the **first CUDA in the repo** (TRT *is* CUDA). Pulls in `cudart` + `nvinfer` + `nppi`. |
| CI / mock | Stub vs skip | **Skip-on-host**, following the VPI/`nvmmofa` precedent: probe `cudart`/`nvinfer`/`nppi` → `have_tensorrt`; TRT elements build only on Jetson, validated on hardware. **Host-CI-able** (mock `NvBufSurface`, no CUDA): the share-capable allocator, `nvmmtracker`, most of `nvmmfusion`. |
| Engine artifacts | Source + precision | Mirror `nvinfer`: `engine-file` (prebuilt `.engine`) and/or `onnx-file` (build + disk-cache on first run; engines are device+TRT-version specific, non-portable). `dla-core` (0/1/-1=GPU), `precision` = **fp16 first**; **int8 deferred** (needs calibration cache). |
| First model | Concrete detector | **Default: YOLO11n/v8n** ONNX → engine; its decode+NMS parser behind a clean `parse(output_tensors) → vector<NvmmDetObject>` interface so a 2nd model is a new function, not a new element. |

## Phase 0 — the go/no-go gate — ✅ CLEARED (GO)

**Standalone on-Jetson reproducer** (`probes/trt_nvbufsurface_probe.cpp`):
confirm an `NvBufSurface` surface's device pointer flows into **NPP and TRT with
no host round-trip**. This single result decides whether *any* `nvmminfer` is
worth building.

**Result (Orin JP6, TensorRT 10.3.0.30 + CUDA 12.5, run on host):** every step
passed → **GO**.

- `NvBufSurface` `dataPtr` is **CUDA device-addressable** (`cudaPointerGetAttributes`
  → `memoryType=device`) for a **pitch-linear RGBA** surface (the VIC-preprocess
  output layout) — *and*, notably, for a **block-linear NV12** surface too (the
  decoder default), so addressability is not the constraint; the de-tile to planar
  (VIC) before TRT is.
- **NPP** (`nppiSet_8u_C4R`) operates on `dataPtr` in place — device→device, no copy.
- **TensorRT** builds an engine, accepts `setInputTensorAddress`/`setTensorAddress`
  on device pointers **we own**, and runs `enqueueV3` + sync — confirming the
  raw-device-pointer binding model the element relies on.

Build/run:
```
g++ -std=c++17 -O2 probes/trt_nvbufsurface_probe.cpp -o trt_nvbufsurface_probe \
  -I/usr/src/jetson_multimedia_api/include -I/usr/local/cuda/include \
  -L/usr/lib/aarch64-linux-gnu/tegra -L/usr/local/cuda/lib64 -L/usr/lib/aarch64-linux-gnu \
  -lnvbufsurface -lnvinfer -lcudart -lnppc -lnppidei -lnppig
./trt_nvbufsurface_probe   # on the HOST (needs /dev/nvmap), not a container
```

## Phasing

- **Phase 1 — `nvmminfer` detector (v1.4.0). ✅ DONE** (PRs #37–#40). Single
  element: VIC+NPP preprocess → TRT engine (fp16) → YOLO parser → `det_meta`,
  plus `nvmmdrawdet` overlay, Dockerised E2E and the golden cross-check
  (onnxruntime fp32 vs TRT: IoU ≥ 0.97). ~23 ms/frame (~43 FPS) at 1080p on Orin.
- **Phase 2 — graph plumbing (v1.5.0). ✅ DONE.** Share-capable `nvmmalloc`
  (#44; on-Jetson probe shows `tee`+`make_writable` keep the same NvBufSurface) +
  `nvmmtracker` (#45/#46; pure-host IOU tracker, ids drawn by the overlay) +
  `nvmmfusion` (PTS join unioning `det_meta` + `nvmmofa` flow meta on one
  buffer; 590/591 frames fused with flow on Orin). Also fixed a latent
  cross-`.so` meta-registration race by making `nvmm_common` a shared library.
- **Phase 3 — cascade + payoff (v1.6.0).** `nvmmsecondaryinfer` (ROI crop +
  re-batch + per-track cache) **and** the sibling metas (classifier result +
  fusion motion-annotation). The "mark moving objects" headline lands here.

## Validation assets

- Download a **public-domain test clip** (traffic/pedestrian, e.g. a CC0 street
  scene) into a fixtures location; decode via `nvv4l2decoder` for a deterministic,
  repeatable hardware input.
- Generate a **golden reference**: run the same YOLO weights on host
  (ultralytics/onnxruntime) to produce expected detections; compare Jetson TRT
  output box-by-box within an fp16 tolerance (IoU + confidence deltas). Guards
  against silent preprocess/parser bugs.

## Open / deferred

- Exact YOLO variant + class set (default YOLO11n/v8n; pin at Phase-1 build).
- Tracker algorithm detail (IOU vs SORT/Kalman) — Phase 2.
- Sibling-meta schemas (tensor + per-object classifier/motion) — Phase 3, model-driven.
- INT8 calibration — post-fp16.
- C++20 baseline via GCC ≥ 14.x + CUDA ≥ 12.6 — see [PRODUCTION_PLAN](PRODUCTION_PLAN.md) Phase 5.
- Xavier motion path (no OFA → classical bg-subtraction) — out of scope (Orin-first).
- Why `NO_SHARE` was originally set: appears to be a conservative default (initial
  allocator commit, no rationale/IPC tie). Re-confirm nothing depends on the broken
  copy path before dropping it; re-run full suite (allocator underpins every element).
