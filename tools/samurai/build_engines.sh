#!/usr/bin/env bash
# In gst-nvmm-infer:jp6 on the Jetson, whose trtexec matches the TensorRT nvmmsamurai links:
# docker run --rm --runtime nvidia --network host -v <onnx>:/onnx -v <out>:/out gst-nvmm-infer:jp6 bash /src/tools/samurai/build_engines.sh
set -eu
ONNX="${ONNX:-/onnx}"
OUT="${OUT:-/out}"
T="${T:-/usr/src/tensorrt/bin/trtexec}"
mkdir -p "$OUT"

flt() { grep -iE "error|fail|passed|engine|version" | tail -8 || true; }

echo "########## image_encoder (fixed 1x3x512x512) ##########"
"$T" --onnx="$ONNX/image_encoder.onnx" --fp16 \
  --saveEngine="$OUT/image_encoder_bplus_512.engine" 2>&1 | flt

echo "########## prompt_encoder ##########"
"$T" --onnx="$ONNX/prompt_encoder.onnx" --fp16 \
  --saveEngine="$OUT/prompt_encoder.engine" 2>&1 | flt

echo "########## mask_decoder (dynamic sparse axis Np: empty=2, box-seed=3) ##########"
"$T" --onnx="$ONNX/mask_decoder.onnx" --fp16 \
  --minShapes=sparse:1x2x256 --optShapes=sparse:1x3x256 --maxShapes=sparse:1x3x256 \
  --saveEngine="$OUT/mask_decoder.engine" 2>&1 | flt

echo "########## memory_encoder ##########"
"$T" --onnx="$ONNX/memory_encoder.onnx" --fp16 \
  --saveEngine="$OUT/memory_encoder.engine" 2>&1 | flt

echo "########## memory_attention ##########"
"$T" --onnx="$ONNX/memory_attention.onnx" --fp16 \
  --saveEngine="$OUT/memory_attention.engine" 2>&1 | flt

echo "########## copy constants ##########"
cp -v "$ONNX/samurai_consts.bin" "$OUT/samurai_consts.bin"

echo "ALL_DONE"; ls -la "$OUT"
