# Building the SAMURAI engines

[`nvmmsamurai`](elements/nvmmsamurai.md) loads five TensorRT engines plus a
constants file, and [`nvmminfer`](elements/nvmminfer.md) loads a YOLO engine.
**None of these are bundled in the repo** — you build them yourself from public,
open-source weights. This page is the end-to-end recipe. Everything runs in Docker.

The scripts live in `tools/samurai/`:

| Script | What it does |
|---|---|
| `export_onnx.py` | Export the 5 sub-models (image/prompt/mask-decoder/memory-encoder/memory-attention) to ONNX. Wraps the **stock** public SAM2 modules with synthetic trace inputs — no captured data; matches the `out1..out6` / dynamic-`sparse` contracts the C++ binds. |
| `pack_consts.py` | Gather the out-of-engine learned constants (temporal pos-enc, no-mem/no-obj embeddings, obj-ptr projections, image PE, empty-prompt sparse/dense) into the self-describing `samurai_consts.bin`. |
| `build_engines.sh` | `trtexec` the five ONNX → fp16 engines on the Jetson; profiles the mask-decoder dynamic `sparse` axis for `Np∈{2,3}`. |

## Open-source inputs

| Artifact | Source |
|---|---|
| SAM 2.1 `base_plus` checkpoint | Meta — [facebookresearch/sam2](https://github.com/facebookresearch/sam2) · [`sam2.1_hiera_base_plus.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) |
| SAMURAI model + configs | [yangchris11/samurai](https://github.com/yangchris11/samurai) ([project page](https://yangchris11.github.io/samurai/)) |
| YOLO detector | [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) (`yolo26n`) |

The tracker runs the SAM 2.1 image encoder at the `nvmmsamurai crop-size`
(default **512×512**, not SAM2's native 1024), so the export overrides
`image_size` to match. At the 512 default that gives `image_embed` 32×32,
`feat_s0` 128×128, `feat_s1` 64×64; a non-512 `crop-size` needs the whole engine
set re-exported at that size (the token grid scales as `crop/16`) — see the
[`crop-size` note](elements/nvmmsamurai.md).

## 1. Fetch the open-source repo + checkpoint

```bash
git clone --recurse-submodules https://github.com/yangchris11/samurai.git
mkdir -p ckpt && wget -O ckpt/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

## 2. Export the SAM2.1 sub-models to ONNX + pack the constants

ONNX export is hardware-agnostic, so it runs in a stock PyTorch container (CPU is
fine). Install SAM2 with the CUDA extension disabled (`SAM2_BUILD_CUDA=0`) — the
export needs only the Python graph:

```bash
docker run --rm -v "$PWD":/work -v <repo>/tools/samurai:/scripts \
  -w /work/samurai/sam2 pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime bash -c '
    export SAM2_BUILD_CUDA=0
    pip install -e /work/samurai/sam2 onnx loguru scipy
    python3 /scripts/export_onnx.py  --ckpt /work/ckpt/sam2.1_hiera_base_plus.pt --out /work/onnx --device cpu
    python3 /scripts/pack_consts.py  --ckpt /work/ckpt/sam2.1_hiera_base_plus.pt --out /work/onnx/samurai_consts.bin --device cpu
'
```

This writes `image_encoder.onnx`, `prompt_encoder.onnx`, `mask_decoder.onnx`,
`memory_encoder.onnx`, `memory_attention.onnx`, and `samurai_consts.bin` to
`onnx/`. `export_onnx.py` wraps the **stock** public modules (no upstream patch):
it flattens the image-encoder dict to the `out1..out6` order the C++ binds, and
re-implements the mask-decoder hypernetwork batched so it exports cleanly to TRT
(candidate selection stays in the C++).

## 3. Export the YOLO detector to ONNX

```bash
docker run --rm -v "$PWD":/work -w /work \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime bash -c '
    pip install ultralytics
    yolo export model=yolo26n.pt format=onnx opset=17 imgsz=1088,1920
'
```

Pick `imgsz` to match your input video; see the
[Ultralytics export docs](https://docs.ultralytics.com/modes/export/). Any
Ultralytics detector works — `nvmminfer` only needs the engine.

## 4. Build the TensorRT engines (on the Jetson)

`trtexec` must be the version the runtime links against, so the engine build runs
in `gst-nvmm-infer:jp6` on the Orin. Copy the `onnx/` dir to the Jetson, then:

```bash
docker run --rm --runtime nvidia --network host \
  -v <repo>:/src -v <onnx_dir>:/onnx -v <out_dir>:/out gst-nvmm-infer:jp6 \
  bash /src/tools/samurai/build_engines.sh
```

This builds the five SAMURAI engines (fp16) into `<out_dir>`, with the
mask_decoder's dynamic `sparse` axis profiled for `Np∈{2,3}` (empty-prompt
tracking vs box-seed), and copies `samurai_consts.bin` alongside. Build the YOLO
engine with `trtexec --onnx=yolo26n.onnx --fp16 --saveEngine=/out/yolo.engine`.

Point the pipeline at `<out_dir>` via `nvmmsamurai engine-dir=…` /
`nvmminfer engine-file=…` — see the
[tracker pipeline walkthrough](tracker-pipeline.md).

## Generic engine builder

`build_engines.sh` above is SAMURAI-specific (it names the five segments). For
anything else — a YOLO detector, a re-exported encoder at a different crop, any
one-off ONNX — `tools/build_engine.sh` is a thin `trtexec` wrapper that turns a
single `.onnx` into an `.engine` on the box the element loads it on:

```bash
tools/build_engine.sh model.onnx model.engine            # fp16 by default
# dynamic input axis (name it as the ONNX declares it):
tools/build_engine.sh mask_decoder.onnx mask_decoder.engine \
  --minShapes=sparse:1x2x256 --optShapes=sparse:1x3x256 --maxShapes=sparse:1x3x256
# encoder exported with dynamic spatial dims — pin the crop at build time:
tools/build_engine.sh image_encoder_384.onnx image_encoder.engine \
  --shapes=input:1x3x384x384
```

Anything after the two paths is passed straight to `trtexec`. Because a
serialized engine is locked to the TensorRT version and GPU arch that built it,
run this on the target (or a container matching its TensorRT), not on a
build host — an engine built elsewhere fails to deserialize with a version-tag
mismatch.

## Note on licences

The model weights are governed by their upstream licences (SAM 2.1 and the YOLO
weights you choose); review them before redistributing any engines you build.
