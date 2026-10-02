#!/usr/bin/env bash
# Test engines for nvmmdetgate (XFeat + LighterGlue) and nvmmsamurai (SAM 2.1 hiera tiny), from pinned Apache-2.0 releases.
# export <work>: any Docker host; downloads, checks sha256, writes <work>/onnx. build <onnx> <engines>: on the Jetson.
# Tests read the engine dir from NVMM_TEST_ENGINE_DIR.
set -euo pipefail

XFEAT_REPO=verlab/accelerated_features
XFEAT_SHA=e92685f57f8318b18725c5c8c0bd28c7fe188d9a
SAMURAI_REPO=yangchris11/samurai
SAMURAI_SHA=76ba195984892b0d1e3db5d9c9f90bb62175680a
SAM2_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
EXPORT_IMAGE=python:3.12-slim
PIP_PINS="torch==2.4.1 torchvision==0.19.1 onnx==1.16.2 onnxruntime==1.19.2 kornia==0.7.2 numpy==1.26.4 hydra-core==1.3.2 iopath==0.1.10 tqdm pillow loguru scipy"

FILES=(
  "xfeat.pt|https://raw.githubusercontent.com/$XFEAT_REPO/$XFEAT_SHA/weights/xfeat.pt|0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"
  "xfeat-lighterglue.pt|https://raw.githubusercontent.com/$XFEAT_REPO/$XFEAT_SHA/weights/xfeat-lighterglue.pt|766102df37f11189efe5b0811d1f47c72b22629b79bfabfcfff9d2a2f84654b8"
  "sam2.1_hiera_tiny.pt|$SAM2_URL|7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
  "accelerated_features.tar.gz|https://codeload.github.com/$XFEAT_REPO/tar.gz/$XFEAT_SHA|2a567feda3bf4ed2a919f96a8bb07013b778bad207bb2693d0a65b121b074a36"
  "samurai.tar.gz|https://codeload.github.com/$SAMURAI_REPO/tar.gz/$SAMURAI_SHA|e5af32ac796062414ec778e25b3ef121a2d91a9a10bef8b86f3bc1d9345a7ec4"
)
ENGINES=(xfeat.engine lightglue.engine image_encoder_bplus_512.engine prompt_encoder.engine
         mask_decoder.engine memory_encoder.engine memory_attention.engine samurai_consts.bin)

TOOLS="$(cd "$(dirname "$0")" && pwd)"

fetch() {
  local dl="$1/dl"
  mkdir -p "$dl"
  for f in "${FILES[@]}"; do
    IFS='|' read -r name url sha <<<"$f"
    [ -s "$dl/$name" ] || curl -fsSL -o "$dl/$name" "$url"
    echo "$sha  $dl/$name" | sha256sum -c -
  done
  rm -rf "$1/src" && mkdir -p "$1/src/xfeat" "$1/src/samurai"
  tar -xzf "$dl/accelerated_features.tar.gz" -C "$1/src/xfeat" --strip-components=1
  tar -xzf "$dl/samurai.tar.gz" -C "$1/src/samurai" --strip-components=1
  cp "$dl/xfeat.pt" "$dl/xfeat-lighterglue.pt" "$1/src/xfeat/weights/"
}

export_onnx() {
  local work; work="$(cd "$1" && pwd)"
  fetch "$work"
  mkdir -p "$work/onnx"
  docker run --rm -v "$work":/work -v "$TOOLS":/tools:ro -w /work "$EXPORT_IMAGE" bash -euc "
    pip install -q --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple $PIP_PINS
    export PYTHONPATH=/work/src/samurai/sam2 PYTHONDONTWRITEBYTECODE=1
    cfg=configs/samurai/sam2.1_hiera_t.yaml ckpt=/work/dl/sam2.1_hiera_tiny.pt
    python3 /tools/xfeat/export_onnx.py --xfeat-src /work/src/xfeat --out /work/onnx
    python3 /tools/samurai/export_onnx.py --ckpt \$ckpt --config \$cfg --out /work/onnx --device cpu
    python3 /tools/samurai/pack_consts.py --ckpt \$ckpt --config \$cfg --out /work/onnx/samurai_consts.bin
    chown -R $(id -u):$(id -g) /work/onnx"
  ls -la "$work/onnx"
}

build_engines() {
  local onnx="$1" out="$2"
  mkdir -p "$out"
  ONNX="$onnx" OUT="$out" bash "$TOOLS/samurai/build_engines.sh"
  bash "$TOOLS/build_engine.sh" "$onnx/xfeat.onnx" "$out/xfeat.engine"
  bash "$TOOLS/build_engine.sh" "$onnx/lightglue.onnx" "$out/lightglue.engine" \
    --minShapes=desc0:1x9x64,desc1:1x9x64,nkpts0:1x9x2,nkpts1:1x9x2 \
    --optShapes=desc0:1x512x64,desc1:1x512x64,nkpts0:1x512x2,nkpts1:1x512x2 \
    --maxShapes=desc0:1x1024x64,desc1:1x1024x64,nkpts0:1x1024x2,nkpts1:1x1024x2
  for e in "${ENGINES[@]}"; do
    [ -s "$out/$e" ] || { echo "missing $out/$e" >&2; exit 1; }
  done
  echo "engines ready: export NVMM_TEST_ENGINE_DIR=$out"
}

case "${1:-}" in
  export) [ $# -eq 2 ] || { echo "usage: $0 export <work_dir>" >&2; exit 2; }; export_onnx "$2" ;;
  build)  [ $# -eq 3 ] || { echo "usage: $0 build <onnx_dir> <engine_dir>" >&2; exit 2; }; build_engines "$2" "$3" ;;
  *) echo "usage: $0 export <work_dir> | build <onnx_dir> <engine_dir>" >&2; exit 2 ;;
esac
