# DLA pathfinder — verdict

**Outcome: KILL the DLA path for SAMURAI (12× regression), DROP the GPU-only
async tail (~1 % hideable, 3.4 % ceiling), and — after testing them —
the two follow-up throughput ideas (static `image_encoder` rebuild, moving the
GPU-bound detector to DLA) are dead too.** The Phase A compile falsifier
triggered the kill-criterion on day one, exactly as
[the plan](dla-pathfinder-plan.md) (decisions 8, 9, 26) intended; the Phase B
falsifier then showed the surviving lever isn't worth the surgery either; the
two follow-up experiments (below) close out the remaining ideas. Net: **no
code or engine change moves this tracker's throughput** on this board — it is
at the GPU-FLOPS ceiling.

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

Close the whole pathfinder. In order tested:

1. DLA partitioning of any SAMURAI engine — **12× regression**, kill-criterion.
2. GPU-only async tail — **~1 % hideable, 3.4 % ceiling**, not worth the surgery.
3. Static rebuild of `image_encoder` — **no gain**, already static/optimal.
4. Moving the GPU-bound detector to DLA core 1 — **NO GO**, 0 layers map at
   this resolution, 6.7–9× slower than GPU.

The plan's Phase B design (decisions 20–25) is left on record but not pursued.
This tracker, on this board, is at the **GPU-FLOPS ceiling**: 67 % of a frame
is `image_encoder`, 30 % is the attention/decoder pre-box stages. The only
remaining throughput levers change the *model or workload*, not its placement:
a smaller Hiera checkpoint, a smaller crop (384 engines already exist on this
box), more aggressive `max-kf` coasting, or more GPU (Orin AGX).

## Follow-up A — static `image_encoder` rebuild: NO GAIN

Hypothesis: `image_encoder` is documented as spatial-dynamic, so a dynamic-shape
TensorRT engine might leave tactic-selection performance on the table vs a
build specialized to the one shape actually used (512×512).

Checked the deployed engine first: `image_encoder_bplus_512.engine` already has
`Profile: Disabled` and a single fixed `1×3×512×512` binding — it is *already*
static. Confirmed by re-building straight from the ONNX (which is itself a
fixed `[1,3,512,512]` graph, no shape flags needed):

| Build | GPU compute (mean) |
|---|---|
| Deployed engine (baseline) | 42.88 ms |
| Freshly built, `--fp16`, no shape flags | 44.02 ms |

No gap to close — the deployed engine was already the static/optimal build.
**No gain available here.**

## Follow-up B — move the GPU-bound detector to DLA: NO GO (9× slower, not 0 layers moved)

Hypothesis: `yolo26n_1088x1920` runs on the **GPU** every frame in the real
pipeline, competing with the SAMURAI encoder for the same SMs. `yolo_ir_640`
already runs on DLA core 0; core 1 is idle. Moving `yolo26n` (or its ONNX
sibling `yolov8n`) to DLA core 1 would free ~21 ms/frame of GPU time for
SAMURAI, at zero cost to tracking quality — this was flagged as "the one DLA
lever left alive" in the first draft of this verdict.

Tested both detector ONNXs (native fixed shape `1×3×576×1920`; deployed runs
at `1088×1920`, ~2× the pixels) on **DLA core 1, FP16, GPU fallback allowed**:

| Model | DLA layers used | Layers forced to GPU | GPU compute (DLA+fallback) | Clean GPU-only build (@576×1920) |
|---|---|---|---|---|
| `yolo26n` | **0** | 71 (all) | 93.5 ms | 10.5 ms |
| `yolov8n` | **0** | 40 (all) | 67.6 ms | ~10 ms (est.) |

**Zero layers landed on DLA for either model** — the whole graph falls back to
GPU, but *slower* than a clean GPU build (9× and 6.7× respectively), because of
the DLA validation overhead plus fallback-path reformats.

Root cause, from the build logs:

- `Dimension: 3 (22680) exceeds maximum allowed size for DLA: 8192` — the
  detection-head flatten (`num_anchors × grid_cells`) blows past DLA's hard
  per-dimension limit at this resolution. At the deployed 1088×1920 (~2× the
  pixels of the 576×1920 tested here) this is worse, not better.
  `yolo_ir_640` avoids this because 640-class input keeps the head's flattened
  dimension under the limit.
- Both models also have attention-family ops in the head/backbone
  (`/model.22/...Softmax`, `/model.10/m/m.0/attn/Softmax` for yolo26n) that DLA
  refuses independent of size.

**Conclusion:** DLA offload of a detector works only at small input resolution
(as `yolo_ir_640` already does) — it does not generalize to a high-resolution,
attention-augmented detector like `yolo26n`/`yolov8n` at 1088×1920. This lever
is **not available** for the GPU-bound detector; no further action.

## Reproduce

All scripts under `tools/samurai/dla-falsifier/`; raw logs also live on the
Jetson at `nvidia@10.0.0.41:/home/nvidia/personalspace/dla-pathfinder/`.

**Phase A (memory_encoder DLA compile falsifier)** — `falsify.sh`,
`falsify_out/{dla_fallback,gpu_baseline,dla_only}.log`:

```bash
docker run --rm --runtime nvidia --network host \
  -v /home/nvidia/workspace/onnx/onnx512/onnx:/onnx \
  -v /home/nvidia/personalspace/dla-pathfinder:/work \
  gst-nvmm-infer:jp6 bash /work/falsify.sh
```

**Phase B (async-tail bubble measurement)** — `instrument.py` (patches
`samurai_tracker.cpp`), `run_timing.sh`, `analyze.py`, `timing.log`:

```bash
python3 tools/samurai/dla-falsifier/instrument.py gst/nvmmsamurai/samurai_tracker.cpp
# rebuild the plugin, then:
docker run --rm --runtime nvidia --network host \
  -v $PWD:$PWD -v <clip-dir>:/o -v <workdir>:/work \
  -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro \
  gst-nvmm-infer:jp6 bash /work/run_timing.sh
# revert samurai_tracker.cpp afterward (instrument.py's changes are throwaway)
```

**Follow-up A/B (static encoder, detector→DLA)** — `exp_ab2.sh`:

```bash
docker run --rm --runtime nvidia --network host \
  -v /home/nvidia/samurai-onnx:/enc \
  -v /home/nvidia/personalspace/pr-verify/samurai-engines/yolo:/y \
  -v /home/nvidia/personalspace/dla-pathfinder:/work \
  gst-nvmm-infer:jp6 bash /work/exp_ab2.sh
```
