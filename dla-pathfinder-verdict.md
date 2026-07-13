# DLA pathfinder — verdict (Phase A, compile falsifier)

**Outcome: KILL the DLA path for SAMURAI. Pivot to Phase B (GPU-only async
tail).** The Step-1 compile falsifier triggered the kill-criterion on day one,
exactly as [the plan](dla-pathfinder-plan.md) (decisions 8, 9, 26) intended.

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

## What survives — Phase B (async tail), GPU-only — but verify before building

Phase B was scoped (plan decision 9) to be worth landing **even if DLA loses**.
That is still the only in-scope throughput lever — but the falsifier does **not**
license the naive framing that overlapping the 1.41 ms `memory_encoder` tail
with the next frame's encoder saves ~1.41 ms.

**Why the win is bounded, and by what.** `memory_encoder(N)` and
`image_encoder(N+1)` both run on the **GPU**. Putting them on separate CUDA
streams does not make them run concurrently on a GPU the encoder already
saturates — their kernels serialize on the SMs. The real win from the async
tail is only the GPU-*idle* time in today's tail that `encoder(N+1)` can fill:
the host obj_ptr MLP, the SamuraiSelector logic, the D2H/H2D copies, and the
two explicit `cudaStreamSynchronize` bubbles (`samurai_tracker.cpp:485` decoder
sync, `:606` memenc sync). That bubble could be near-zero (tail is
compute-bound) or a solid fraction of a frame (tail is bubble-bound) — the
falsifier can't distinguish the two.

**Apply the same falsifier-first discipline before the Impl surgery.** The plan
itself calls the async-tail restructure "real surgery." Before committing to it,
run the cheap falsifier: an **`nsys` trace of one `track()`** shows the tail's
GPU-idle bubble size directly. That number decides whether Phase B is worth the
restructure — measure it *before* building, exactly as this DLA falsifier
gated Phase A.

**Recommendation:** close the DLA effort with this verdict; make Phase B's
next action an `nsys` bubble-size measurement, then (if the bubble is
worthwhile) the GPU-only async-tail restructure, judged by a `pipeline_bench.py`
A/B. The pix_feat hazard fix and event/stream design in plan decisions 20–25
stand unchanged if we proceed.

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
