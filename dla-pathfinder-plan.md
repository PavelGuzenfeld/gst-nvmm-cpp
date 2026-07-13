# DLA pathfinder — partition SAMURAI engines across DLA/GPU

## Goal

Increase `nvmmsamurai` throughput on Jetson Orin NX by partitioning the SAMURAI
model graphs themselves (the ONNX sources → multiple sub-engines/loadables) so
parts run on the DLA while the GPU works on the next frame, chained by
zero-copy device buffers on CUDA streams. Pathfinder target: `memory_encoder`
on DLA end-to-end, plus an async-tail restructure of the tracker that overlaps
the memory-update tail of frame N with frame N+1's crop + image encoder.

## Non-goals

- INT8 anywhere (no calibration pipeline; accuracy risk with no recovery path).
- DLA for `memory_attention`, `mask_decoder`, `prompt_encoder` (pure/mostly
  transformer — DLA-unsupported).
- Op substitution (GELU→ReLU etc.) — changes the math; no fine-tuning loop in
  this repo.
- The `image_encoder` Hiera split (patch-embed / FPN neck → DLA). That is the
  eventual payoff but is gated on this pathfinder's verdict; not designed here.
- Cross-process or non-CUDA consumers (no NvSci standalone-mode plumbing).

## Decisions

1. Scope is intra-model partitioning: cut the SAMURAI ONNX sources into
   sub-engines/loadables placed per-part on DLA or GPU, chained by zero-copy
   device buffers.
2. All DLA work runs FP16; INT8 is excluded.
3. The pathfinder engine is `memory_encoder`; the `image_encoder` split is
   attacked only after the pathfinder delivers a verdict.
4. The DLA runtime is standalone cuDLA (explicit loadables +
   `cudlaMemRegister`), not TRT-managed DLA.
5. cuDLA runs in hybrid mode (`CUDLA_CUDA_DLA`): DLA tasks submitted on CUDA
   streams, buffers are registered CUDA device memory, sync is CUDA events.
6. Loadables are compiled through the TRT builder
   (`trtexec --buildDLAStandalone --useDLACore=0 --fp16`) on-Jetson inside the
   jp6 container; cuDLA replaces only the runtime.
7. `memory_encoder`'s LayerNorm2d/GELU layers (DLA-unsupported) are handled by
   cutting the ONNX at op boundaries: dense conv chunks become DLA loadables;
   the norm/activation glue runs as CUDA kernels hosted in
   `samurai_kernels.cu`, on the same registered buffers.
8. Kill-criterion: if the DLA chunk chain (submits + glue kernels + reformats,
   end-to-end) is slower than today's whole-engine GPU `memory_encoder`
   enqueue, write the verdict and stop — do not integrate a regression.
9. Work is two gated phases: Phase A runs the DLA chunks inline (serialized,
   today's order) to prove compile/parity/latency; Phase B (async tail) proceeds
   only after Phase A's numbers exist — and is worth landing even if DLA loses,
   as a GPU-only overlap win.
10. The graph-cut script lives in `tools/samurai/` next to the existing
    export/pack scripts.
11. Loadables land in `engine-dir` as `memory_encoder_dla_chunk{0..N}.loadable`,
    discovered by directory like the five engines.
12. `docs/building-engines.md` gains an optional "DLA loadables" section; no
    separate experimental doc tree.
13. New element property `dla-core` (int, default -1 = off; 0/1 selects a core);
    core count validated at init via `cudlaDeviceGetCount`, not hardcoded.
14. No silent fallback: with `dla-core >= 0`, any missing loadable, cuDLA init
    failure, or chunk-shape mismatch fails `init()` with a clear error — never
    a quiet GPU downgrade.
15. `SamuraiConfig` gains `int dla_core = -1`; the chunk-chain runner is a new
    unit `samurai_dla.{hpp,cpp}` (not inlined into `samurai_tracker.cpp`).
16. Phase A acceptance: tensor parity on `maskmem_feat`/`maskmem_pos` vs the GPU
    FP16 baseline via the `SAMURAI_DUMP_DIR` dump path — max abs diff ≤ 1e-2
    and mean abs diff ≤ 1e-3.
17. Phase A acceptance: behavioral parity on a full clip — per-frame box IoU vs
    the GPU run ≥ 0.99 median, no frame < 0.9, no valid-flag flips.
18. Phase A acceptance: latency measured with CUDA events around the chunk chain
    vs the same events around today's GPU `memory_encoder` enqueue; the
    kill-criterion (decision 8) applies to this number.
19. The results (parity + latency + verdict) are written up regardless of
    outcome — the verdict doc is a deliverable even on a kill.
20. Async tail (Phase B), fully designed: only device work defers to a second
    CUDA stream (stream B) — mask-prep kernels (`k_bilinear`/`k_sigmoid_scale`),
    `memory_encoder` (DLA chunks or GPU), and the ring-slot copy (switched from
    sync `cudaMemcpy` to `cudaMemcpyAsync`). The host obj_ptr MLP and ring
    bookkeeping stay synchronous inside `track(N)`.
21. Async tail join point: `track(N+1)` waits `evTailDone` immediately before
    memory-assemble (step 2 of `track_frame`); VIC crop, `image_encoder`, and
    the curr/curr_pos transposes for frame N+1 overlap the tail. All
    `d_dmasks`/`d_high`/scratch reuse is ordered behind that event.
22. pix_feat hazard fix: stream B's first op is a ~1 MB D2D copy
    `out6 → d_pix_feat_tail`; stream A waits `evPixCopied` before enqueueing
    `image_encoder(N+1)` (the VIC crop proceeds regardless); `memory_encoder`
    binds `d_pix_feat_tail` permanently.
23. Exception paths stay synchronous: the rare occlusion `no_obj_embed` host
    fixup and the non-finite-objectness coast path force an immediate join.
24. `seed()`, reseed, and the destructor join stream B before touching rings or
    engines.
25. Deferred tail errors surface at the next join and hard-fail `track(N+1)`,
    same error contract as today.
26. The compile falsifier runs first, before any tooling or runtime code: a
    rough hand-driven cut of `memory_encoder.onnx` + per-chunk DLA compile.
    Its output (unsupported-op list, chunk fragmentation, proxy latency) can
    trigger the kill-criterion on day one; the cut script, glue kernels, and
    cuDLA runner are written only if it passes.

## Open questions

- Chunk count and cut points — unknown until `memory_encoder.onnx` is actually
  surgeried; the LayerNorm/GELU density decides how fragmented the DLA work is
  (and heavily influences the kill-criterion outcome). Answered by step 1.
- DLA FP16 tensor-format cost (`CHW16`/linear reformats at each chunk boundary)
  — measured, not estimated, in step 1.

Resolved since first draft:

- Deployment target is **Orin NX 16 GB (2 DLA cores)** — the pathfinder uses
  core 0; core 1 stays free for a future `image_encoder` split or a DLA YOLO.
  Runtime still validates via `cudlaDeviceGetCount` (decision 13).
- Throughput bar is **"measurably better"** — no fixed fps number. Phase B
  (async tail) lands on any statistically clean `pipeline_bench.py` A/B win;
  the DLA path is justified only by its own A/B.

## Steps

1. **Compile falsifier** (on-Jetson, jp6 container, no repo code yet): rough
   hand-driven cut of `memory_encoder.onnx` at LayerNorm2d/GELU boundaries
   (`onnx.utils.extract_model`, throwaway notebook/script), then per chunk:
   - `trtexec --buildDLAStandalone --useDLACore=0 --fp16` → proves the chunk
     compiles to a loadable at all (no fallback: definitive unsupported-op
     list);
   - the same chunk as a regular DLA engine (`--useDLACore=0 --fp16`, no
     `--allowGPUFallback`) timed with `trtexec` → per-chunk DLA latency proxy
     without writing any cuDLA runtime code.
   Deliverable: chunk count, op list, summed proxy latency vs the GPU
   whole-engine number. **Gate: kill-criterion (decision 8) — a fragmented or
   slow result ends the DLA effort here with the verdict doc, before steps
   2-6.**
2. **Cut script** (`tools/samurai/`): productionize the falsifier's cut —
   graph-surgeon `memory_encoder.onnx` → N conv-chunk ONNXs + a manifest
   (chunk I/O names/shapes/order), then compile
   `engine-dir/memory_encoder_dla_chunk*.loadable` as in step 1.
3. **Glue kernels**: implement the LayerNorm2d/GELU segments as CUDA kernels in
   `samurai_kernels.cu` (weights come from the cut manifest / consts pack).
4. **Runner** (`samurai_dla.{hpp,cpp}`): cuDLA hybrid-mode device init,
   loadable load, `cudlaMemRegister` of the shared device buffers, and a
   `run(stream)` that alternates `cudlaSubmitTask` / glue kernels down the
   chunk chain. Property plumbing: `dla-core` on the element →
   `SamuraiConfig::dla_core`; hard-fail init policy (decision 14).
5. **Phase A integration (inline)**: behind `dla-core >= 0`, replace the
   `mem_encoder->infer(stream)` calls in seed and track paths with the chunk
   chain, serialized exactly where the GPU engine runs today.
6. **Phase A measurement**: parity dumps GPU vs DLA (decision 16), full-clip
   behavioral parity (decision 17), CUDA-event latency A/B (decision 18).
   Write the verdict doc (decision 19). **Gate: kill-criterion.**
7. **Phase B async tail** (proceeds on GPU even if DLA killed): second CUDA
   stream + `evTailDone`/`evPixCopied` events in `SamuraiTracker::Impl`;
   `d_pix_feat_tail` buffer + permanent `memory_encoder` pix_feat rebind;
   defer mask-prep + memory_encoder + async ring copy to stream B; join before
   memory-assemble; sync exception paths; join in seed/reseed/destructor;
   deferred-error surfacing.
8. **Phase B measurement**: pipeline-level fps A/B with `pipeline_bench.py`
   (tail-async on/off × DLA on/off), plus full-clip behavioral parity re-run.
   Fold results into the verdict doc; update `docs/elements/nvmmsamurai.md`
   (property table + a DLA/async note) and `docs/building-engines.md`.

## Risks / rejected alternatives

- **TRT-managed DLA (rejected in favor of standalone cuDLA)** — one-line build
  change, near-zero integration, would have answered feasibility cheaply; user
  chose explicit cuDLA control. Risk accepted: a few hundred lines of runtime
  code before the first measurement.
- **cuDLA standalone mode (NvSciBuf/NvSciSync)** — rejected; only pays off for
  non-CUDA/cross-process consumers, and everything else here lives on CUDA
  streams.
- **GELU→ReLU / op substitution** — rejected outright; changes the math with
  no fine-tuning path to recover accuracy.
- **Ping-pong encoder output buffers** (pix_feat hazard alternative) —
  rejected; makes every enc_out consumer track frame parity (the tail's only
  cross-frame read is out6 ≈ 1 MB, so ping-pong buys no memory saving over the
  single 1 MB copy + one event, at the cost of parity bookkeeping everywhere).
- **Graceful GPU fallback when DLA init fails** — rejected; repo policy is no
  silent fallbacks. You asked for DLA; you get DLA or a diagnosis.
- **Whole `memory_encoder` as one DLA loadable** — expected to fail compile
  (LayerNorm2d/GELU are DLA-unsupported); the op-boundary cut (decision 7) *is*
  the mitigation. If the conv chunks come out too fragmented, the
  kill-criterion (decision 8) ends it with a written verdict.
- **Fundamental risk, recorded**: DLA on Orin is INT8-first and roughly 5-10×
  slower per-op than the iGPU in FP16 even on supported convs, and
  `memory_encoder` is only ~3-5% of a frame. The pathfinder's realistic upside
  is toolchain knowledge for the `image_encoder` split, not fps. The async
  tail (Phase B) is the part expected to yield real throughput on its own.
- **Dependency-loop constraint** (why frames can't naively pipeline): frame
  N+1's `memory_attention` consumes frame N's memory-bank write, and frame
  N+1's crop is centered on frame N's box. The async tail exploits the only
  legal overlap window without going speculative; a KF-predicted/stale crop
  (speculative encoder) was discussed and deferred, not designed.
