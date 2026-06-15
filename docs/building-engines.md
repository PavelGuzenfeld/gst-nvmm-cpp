# Building the SAMURAI engines

[`nvmmsamurai`](elements/nvmmsamurai.md) loads five TensorRT engines plus a
constants file, and [`nvmminfer`](elements/nvmminfer.md) loads a YOLO engine.
**None of these are bundled in the repo** — you build them yourself from public,
open-source weights. This page is the end-to-end recipe; the scripts live in
[`tools/samurai/`](https://github.com/PavelGuzenfeld/gst-nvmm-cpp/tree/main/tools/samurai).

Everything runs in Docker.

## Open-source inputs

| Artifact | Source |
|---|---|
| SAM 2.1 `base_plus` checkpoint | Meta — [facebookresearch/sam2](https://github.com/facebookresearch/sam2) · [`sam2.1_hiera_base_plus.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) |
| SAMURAI model + configs | [yangchris11/samurai](https://github.com/yangchris11/samurai) ([project page](https://yangchris11.github.io/samurai/)) |
| YOLO detector | [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) (`yolo26n`) |

The tracker runs the SAM 2.1 image encoder at a **512×512 crop** (not SAM2's
native 1024), so the export overrides `image_size=512` → `image_embed` 32×32,
`feat_s0` 128×128, `feat_s1` 64×64.

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

## Note on licences

The model weights are governed by their upstream licences (SAM 2.1 and the YOLO
weights you choose); review them before redistributing any engines you build.
