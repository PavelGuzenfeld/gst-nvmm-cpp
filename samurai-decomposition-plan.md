# SAMURAI decomposition + per-accelerator placement — CLOSED by measurement

**Outcome: there is nowhere to place anything.** The two probes this plan hinged
on were run and both close their branch:

- **SM headroom is ~10%** (`--streams` on the deployed encoder engine), so
  multi-frame pipelining is capped at +10% for major surgery with quality risk.
- **DLA takes nothing**, measured on 4 of 5 graphs. `image_encoder` alone pushes
  5732 layers back to GPU and hits the 16-subgraph limit 92 times.

Since the placement space for NN layers is exactly {GPU, DLA} (VIC and PVA cannot
execute NN layers at all), and DLA is closed while the GPU has 10% of slack, the
per-accelerator placement question is answered: **keep everything where it is.**

The per-module roofline table (step 1) was never built and is now **largely moot** —
its purpose was to guide placement, and there is no placement decision left. Its
residual value would be guiding a *model* change (smaller Hiera, smaller crop),
which is a different lever and was scoped out of this effort.

Related: detector decimation surfaced during the same interview and is written up
separately in [detector-decimation-verdict.md](detector-decimation-verdict.md) —
+57.8% measured, currently blocked on an `nvmmfusekf` flush-BB coupling.

## Objective

Maximize throughput of the single live-stream the labelled evaluation set pipeline, holding tracking
quality invariant. Latency, memory, and resource utilization are the currency we
may spend — latency is effectively unbounded (the consumer tolerates lag; the
payoff is closing the frame-drop gap against the camera rate). Zero-copy is an
optimization to be justified by measurement, not a rule.

## Decisions

1. **Target graphs**: `image_encoder` @512 (67% of frame) and `memory_attention`
   + `mask_decoder` (30% pre-box). `memory_encoder` and `prompt_encoder` are
   excluded — already characterized in the DLA pathfinder.
2. **Two deliverables**: (a) the per-op DLA-eligibility measurement the DLA
   verdict *argued* rather than measured for these three graphs; (b) the
   component decomposition itself, as a reference artifact.
3. **Order**: decomposition first, then hardware probes, so probes target the
   modules that actually carry FLOPs rather than whole graphs.
4. **Table shape**: per-module rows (~30/graph), columns = measured ms | params |
   activation MB | arithmetic intensity | bound | DLA-hostile op count. Rows come
   from ONNX node-name prefixes.
5. `--streams=1` vs `=2` per-module deltas are **columns in that same table**,
   not a deferred probe — that is what validates the `bound` column against
   reality.
6. **Roofline knee derived empirically** on the box (a matmul-heavy and a
   copy-heavy layer through `trtexec`, achieved bandwidth via `tegrastats`), not
   from a spec-sheet TFLOPS figure.
7. **Quality bar** = DLA-pathfinder decisions 16–17, reused: placement-only
   changes (same math) need tensor parity via `SAMURAI_DUMP_DIR`, max abs diff
   ≤1e-2 / mean ≤1e-3; math-changing changes need behavioural parity — per-frame
   IoU vs baseline ≥0.99 median, no frame <0.9, no valid-flag flips. The gate is
   `tools/trajectory_compare.py`, which now exists.
8. **INT8 out of scope.** FP16 throughout, placement and queues only.

## Hardware constraint — settles the CPU/GPU/DLA/VIC/PVA framing

**VIC and PVA cannot execute NN layers.** VIC does 2D ops (rescale / convert /
composite); PVA runs a fixed VPI algorithm set. For the ~65 ms of NN work the
placement space is exactly **{GPU, DLA}**. VIC already has the crop, PVA already
has a GMC backend, the CPU already has the obj_ptr MLP and selector. The OFA is
idle but no NN layer can use it.

DLA core 0 is **occupied** by `detector_ir` in the deployed pipeline; core 1 is
free.

## Facts established during the interview

- **ONNX node names preserve module hierarchy**, so the table is mechanical:
  - `image_encoder`: `/enc/trunk/blocks.{0..23}` (Hiera-B+, stages 2/3/16/3),
    `/enc/neck/convs.{0..3}`, `/enc/neck/position_encoding`
  - `memory_attention`: `/ma/layers.{N}/{self_attn,linear1,linear2,Relu}` — note
    **ReLU, not GELU**, materially DLA-friendlier than the encoder, and the one
    genuine remaining DLA candidate
  - `mask_decoder`: **partially flat** — named heads (`/hyper/`,
    `/iou_prediction_head/`, `/conv_s*`) but the transformer body sits at root as
    bare `/Add_*`, `/Einsum*`; needs an `<unnamed-root>` bucket
- **The serialization point is step 7, not step 9.** `last = out.box` commits at
  `samurai_tracker.cpp:568`; nothing in steps 8–9 writes it. So
  `image_encoder(N+1)` is unblocked at ~96.6% of frame N, which is the DLA
  pathfinder's 3.4% overlap window confirmed from the other direction. Speculative
  crop therefore has no correctness hazard in the tail — only quality drift.
- **Speculative-crop machinery already exists**: `apply_gmc` (`:300`) shifts
  `last` by measured camera motion before the crop, and the KF already runs
  `predict()`.
- **Element-level GStreamer queues are already fully exploited** — `run.sh:43-48`
  and every deployed script have `! queue !` between every element (the diagram in
  `docs/tracker-pipeline.md:104` omits them for readability). A fully-threaded
  pipeline still yields the serialized sum, which is itself evidence for the GPU
  saturation the DLA verdict only asserted.

## The arithmetic gate

Of a 67.8 ms tracker frame: `image_encoder` 45.4 ms (GPU), pre-box 20.1 ms (GPU),
tail 2.27 ms — **96.6% GPU-bound**. Producer/consumer queues convert latency into
throughput only across *different* engines. If nothing leaves the GPU, perfect
multi-frame pipelining with zero bubbles buys ≤3.4%, the number Phase B already
measured.

So the plan lives or dies on **SM headroom, not DLA op support**.

### MEASURED (step 2 done) — SM headroom is ~10%, and that closes the pipelining branch

`trtexec --loadEngine=image_encoder_bplus_512.engine --fp16 --iterations=100
--streams=N`, Orin NX MAXN:

| --streams | throughput | gain | mean latency |
|---|---|---|---|
| 1 | 22.60 qps | — | 46.9 ms |
| 2 | **24.86 qps** | **+10.0%** | 82.2 ms |
| 3 | 24.94 qps | +10.3% | 102.6 ms |

The GPU is **~90% saturated by one encoder stream**, and concurrency saturates
completely at 2 streams. The DLA verdict's asserted premise ("a GPU the encoder
already saturates") is ~90% true, not absolutely true.

**Consequence: +10% is the hard ceiling on anything multi-frame pipelining can
deliver for the encoder.** That beats Phase B's 3.4%, but it costs speculative
crop, double-buffered state, and quality risk — against +57.8% from a single
`nvmminfer` property. Step 4 of this plan (speculative crop + 2-frame pipeline) is
therefore **not worth building**; it is closed by arithmetic, not by op support.

Consistent with the detector A/B's incidental finding that ~5 ms of ~20 ms of
detector work overlapped tracker work when the tracker was busy every frame.

### MEASURED (step 3, partial) — the DLA generalization is CONFIRMED, and for a different reason than expected

`trtexec --onnx=<g> --useDLACore=1 --fp16 --allowGPUFallback --verbose
--skipInference` on the onnx512 set:

| graph | build | layers switched to GPU | 16-subgraph-limit hits | unsupported ops |
|---|---|---|---|---|
| `memory_attention` | **FAILS (exit 1)** | 973 | **26** | SQRT ×24, DIV ×8 |
| `mask_decoder` | ok | 1154 | 1 | DIV ×28, SQRT ×22, AVG-Reduce ×2, ERF ×2, POW ×1 |
| `image_encoder` (the 67% graph) | ok | **5732** | **92** | DIV ×175, FLOOR ×129, SQRT ×72, ERF ×24, SIN ×6, COS ×6, LESS, GREATER |

`image_encoder` is the worst of all: 5732 layers pushed back to GPU and the
16-subgraph limit hit 92 times. FLOOR ×129 and SIN/COS ×6 come from Hiera's window
partitioning and positional-encoding interpolation — op classes DLA refuses that
are not even LayerNorm.

**The DLA branch is now closed by measurement on 4 of 5 graphs** (`memory_encoder`
in the prior pathfinder; these three here). Only `prompt_encoder` is unmeasured and
it is 17 KB / 119 KB — irrelevant to throughput. No further DLA work is warranted,
and the claim no longer rests on any architectural argument.

(Per-layer "running on DLA" greps are unreliable — they match summary lines. The
switched-to-GPU counts and limit hits are the trustworthy figures.)

**The ReLU-FFN hypothesis in this plan was wrong.** `memory_attention` was
predicted to be the one genuine DLA candidate because its FFN uses ReLU rather
than GELU. It is the *worst* of the three — it does not even build with GPU
fallback, hitting the 16-subgraph limit 26 times. The wall is **LayerNorm**
(SQRT/DIV/ReduceMean), which sits between every block of every transformer.
GELU-vs-ReLU is irrelevant when each LayerNorm forces a DLA↔GPU boundary.

So the DLA verdict's generalization now rests on measurement for 3 of 5 graphs
(`memory_encoder` previously, these two now) rather than on architectural
argument. Step 3 below is closed for these two; `image_encoder` measured
separately.

## Steps

1. Per-module decomposition table for the three graphs (decisions 4–6), built
   from `trtexec --dumpProfile --separateProfileRun` on the engines already on the
   box, aggregated by ONNX node-name prefix, plus params/activation bytes from the
   ONNX. Reuse `tools/samurai/dla-falsifier/inspect_memenc.py` (takes a path arg).
2. `--streams=1/2/3` on `image_encoder_bplus_512.engine` for the saturation
   number, as table columns.
3. Per-op DLA-eligibility for the three never-measured graphs —
   `trtexec --useDLACore=1 --allowGPUFallback --verbose`, counting layers that
   actually land. Reuse the grep set in `dla-falsifier/exp_ab2.sh`.
   `memory_attention` first: it is the only ReLU-FFN candidate.
4. Only if 2–3 show exploitable headroom: speculative crop + 2-frame pipeline,
   gated on `tools/trajectory_compare.py`.

## Honest expected outcome

The placement half will most likely produce a second kill. Element-level queues
are exhausted; intra-frame overlap measured 3.4%; cross-frame pipelining needs
both speculative crop *and* SM headroom that the fully-queued-yet-serialized
pipeline suggests is near zero; and DLA is measured dead on the most conv-friendly
of the five graphs. `memory_attention`'s ReLU FFN is the one genuine candidate.
Step 3 is cheap and closes the DLA question with data rather than the
generalization the DLA verdict currently rests on.
