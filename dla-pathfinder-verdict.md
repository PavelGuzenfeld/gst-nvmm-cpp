# DLA pathfinder — verdict

**Outcome: KILL the DLA path for SAMURAI (12× regression), and — after
measuring it — DROP the GPU-only async tail too (~1 % hideable, 3.4 % ceiling).**
The Phase A compile falsifier triggered the kill-criterion on day one, exactly
as [the plan](dla-pathfinder-plan.md) (decisions 8, 9, 26) intended; the Phase B
falsifier (below) then showed the surviving lever isn't worth the surgery
either. Net: no code change to the tracker is warranted; the real throughput is
in the `image_encoder` (67 % of the frame), not the memory-update tail.

## Setup

- Board: **Jetson Orin NX 16 GB** (Engineering Reference Dev Kit) — **2 DLA
  cores** confirmed (`Builder.num_DLA_cores == 2`).
- L4T R36.4.3 (JetPack 6), TensorRT **10.3.0.30** / CUDA 12.5, container
  `gst-nvmm-infer:jp6`.
- Target: `memory_encoder.onnx` (512-crop export,
  `workspace/onnx/onnx512/onnx/memory_encoder.onnx`, 5.6 MB).
- All three passes via `trtexec`, FP16, on-Jetson in the container.

## Results

| Pass | Config | Build | GPU compute (mean) | End-to-end latency | Throughput |
|---|---|---|---|---|---|
| 2 | **GPU-only** `--fp16` (kill-criterion denominator) | ✅ | **1.41 ms** (median 1.38) | 1.58 ms | 706 qps |
| 1 | DLA core 0 `--fp16 --allowGPUFallback` (TRT's own optimal mixed placement) | ✅ | 17.25 ms | 17.64 ms | 57.6 qps |
| 3 | Whole graph forced DLA, **no** fallback (standalone-loadable feasibility) | ❌ | — | — | — |

- **Pass 1 is 12.2× slower than the GPU baseline** even though TRT is free to
  place as much as possible on DLA. The kill-criterion (decision 8: DLA chain
  time > whole-engine GPU time) is met by more than an order of magnitude.
- **Pass 3 fails to build** — first refusal at
  `/e/mask_downsampler/encoder/encoder.1/ReduceMean`: *"DLA cores do not
  support AVG Reduce operation."* The engine cannot be a single DLA loadable.

## Root cause — SAM2.1's `memory_encoder` is conv-*shaped*, not conv-*pure*

The mask_downsampler and the ConvNeXt fuser interleave DLA-unsupported ops
**between every convolution**:

- **LayerNorm2d** expands to `ReduceMean → Sub → Pow → ReduceMean → Add →
  Sqrt → Div → Mul → Add` — DLA supports none of the reduce/pow/sqrt/div ops
  (12× AVG-Reduce refusals, 16× DIV, 6× POW, 4× FLOOR observed).
- **GELU** expands to `Div → Erf → Add → Mul` — also unsupported.
- Total: **312 layers "Unsupported on DLA"** across `encoder.0..12` + fuser.

Consequences when TRT cuts around them (Pass 1):

- The graph fragments past the **hard limit of 16 DLA subgraphs per core** —
  TRT hit it 7 times and dumped the overflow back to GPU.
- The DLA↔GPU boundary generated **~12,000 reformat layers** (CHW16↔linear),
  which dominate the runtime — this is the entire 12× regression.

The conv "chunks" between the norm/activation ops are individually tiny (often
a single conv), so even a hand-cut that puts only the largest pure-conv chunk
on DLA cannot win: inline (Phase A) it is strictly additive (GPU still runs the
rest + 2 reformats + the serial DLA chunk), and the whole engine is only
1.41 ms on GPU to begin with.

**Scope of the measurement (honest disclosure).** Pass 1 measures TRT's
whole-graph auto-placement, not the plan's per-chunk *isolated* DLA timing
(decision 18). We did **not** hand-cut and time a single conv chunk in
isolation — deliberately, because it cannot flip the verdict: the DLA↔GPU
boundary is crossed at *every* LayerNorm/GELU regardless of how the graph is
sliced (each conv is its own DLA island), so the reformat/subgraph explosion is
intrinsic to the architecture, not an artifact of greedy `--allowGPUFallback`
placement. The ceiling argument closes it without the per-chunk number: the
whole engine is 1.41 ms on GPU, DLA FP16 convs are slower per-op and, inline,
run serially — so no isolated chunk can beat the GPU whole-engine time.

## Generalization — the DLA path is dead for all of SAMURAI, not just this engine

`memory_encoder` was the *best* DLA candidate (the most conv-heavy of the five
engines). It failed. The eventual payoff target, `image_encoder`, is a Hiera
**attention** transformer (LayerNorm/GELU/softmax throughout) — categorically
worse for DLA. `memory_attention` and `mask_decoder` are pure transformers.
There is no SAMURAI engine that maps to DLA in FP16 without a reformat-dominated
regression. **No further DLA work is warranted.**

The transferable knowledge (the real deliverable of a killed pathfinder): on
Orin + TRT 10.3, DLA FP16 placement is viable only for graphs that are conv/pool
*without* interleaved LayerNorm/GELU/softmax and that stay under 16 subgraphs
per core. Modern transformer/ConvNeXt vision blocks violate both.

## Phase B (async tail), GPU-only — MEASURED, and it's a NO-GO too

Phase B was scoped (plan decision 9) to be worth landing **even if DLA loses**,
so before writing it off we measured it — same falsifier-first discipline that
gated Phase A. nsys is not installed on the box (no binary, no `.deb`), so we
instrumented `track()` directly with CPU timestamps + a `cudaEvent` around
`memory_encoder`, rebuilt the plugin, and ran 250 full-inference frames
(`max-kf=0`) on a 1080p clip. Steady-state (200 frames after warmup):

| Stage | mean ms | % of frame |
|---|---|---|
| `image_encoder` (`run_encoder`) | 45.4 | **67 %** |
| pre-box (memory_attention + mask_decoder + candidate loop) | 20.1 | **30 %** |
| **tail (steps 8–9, deferrable by Phase B)** | **2.27** | **3.4 %** |
| &nbsp;&nbsp;└ `memory_encoder` GPU compute | 1.53 | — |
| &nbsp;&nbsp;└ **hideable bubble** (tail − memenc) | **0.74** | **1.1 %** |
| frame total | 67.8 | 100 % |

(The measured `memenc_gpu` = 1.53 ms matches the trtexec 1.41 ms baseline — the
instrumentation is sound. Numbers are at the board's current `nvpmodel`, not a
forced MAXN clock; the conclusion is power-mode-invariant because all GPU stages
scale together, so the tail/frame *ratio* holds regardless of absolute ms.)

**Why this kills Phase B.** `memory_encoder(N)` and `image_encoder(N+1)` both run
on the **GPU**; separate CUDA streams do not make them run concurrently on a GPU
the encoder already saturates — their kernels serialize on the SMs. So the clean
win is only the tail's GPU-*idle* bubble (host obj_ptr MLP, selector logic,
D2H/H2D copies, the `cudaStreamSynchronize` at `samurai_tracker.cpp:606`), which
`encoder(N+1)` can fill: **0.74 ms, ~1.1 % of a frame.** Even the optimistic
ceiling — pretending the *entire* tail hides for free — is **3.4 %.** That is
not worth the "real surgery" the plan describes (double-buffered maskmem ring
slot, two events, pix_feat tail copy, occlusion/reseed edge cases). With the
default `max-kf=2` the tail runs only every third frame, so the real-run payoff
is smaller still.

**Where the time actually is.** The frame is 67 % `image_encoder` and 30 % the
attention/decoder pre-box stages. Any throughput work on this tracker belongs
there — a lighter/faster encoder, a smaller crop, or the `max-kf` coasting that
already exists — not in the memory-update tail.

## Answering the original question — "zero-copy queues between parts"

The premise was to split the engine into parts on DLA/GPU with zero-copy queues
between them *for parallelism*. That structure yields no win here, for a
structural reason independent of the DLA op-support numbers:

- Splitting a **sequentially-dependent** engine across DLA/GPU creates **no
  intra-frame concurrency** — chunk 2 consumes chunk 1's output *of the same
  frame*, so the queue between them is a hand-off, not an overlap.
- The SAMURAI **frame dependency chain** blocks cross-frame pipelining of
  `memory_encoder`: it can't start for frame N until that frame's mask exists
  (decoder → selector → mask), and its output feeds frame N+1's
  `memory_attention`. The *only* legal overlap is `memory_encoder(N)` running
  behind `image_encoder(N+1)` — which is exactly the Phase B tail overlap,
  measured above at a 3.4 % ceiling.

So the DLA-with-queues design is **strictly dominated** by the GPU-only async
tail we already rejected: it targets the same 3.4 % overlap window, but makes
that window *slower* (DLA FP16 convs) and *reformat-laden* (12k CHW16↔linear
copies). There is no arrangement of DLA parts + zero-copy queues that beats
leaving `memory_encoder` on the GPU.

## Recommendation

Close the DLA effort **and** the async-tail effort. Neither the DLA path
(12× regression, kill-criterion) nor the GPU-only async tail (~1 % hideable,
3.4 % ceiling) is worth building. The plan's Phase B design (decisions 20–25)
is left on record but not pursued. If tracker throughput is revisited, target
the `image_encoder` (67 % of the frame) or the pre-box transformer stages
(30 %).

## Aside — the one DLA lever left alive (out of scope)

The only SAMURAI-adjacent engine that *would* map cleanly to DLA is the
**detector** (`nvmminfer`/YOLO — conv-pure, DLA-friendly), which would free the
GPU for SAMURAI. That is outside the SAMURAI-engine partitioning scope set for
this pathfinder; noted, not pursued.

## Reproduce

Falsifier scripts + raw logs live on the Jetson at
`nvidia@10.0.0.41:/home/nvidia/personalspace/dla-pathfinder/` (`falsify.sh`,
`falsify_out/{dla_fallback,gpu_baseline,dla_only}.log`). Rerun:

```bash
docker run --rm --runtime nvidia --network host \
  -v /home/nvidia/workspace/onnx/onnx512/onnx:/onnx \
  -v /home/nvidia/personalspace/dla-pathfinder:/work \
  gst-nvmm-infer:jp6 bash /work/falsify.sh
```
