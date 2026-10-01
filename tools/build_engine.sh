#!/usr/bin/env bash
# On-device only: an engine is locked to the TensorRT version and GPU arch that built it,
# and deserializeCudaEngine rejects any other ("Version tag does not match").
set -eu
if [ "$#" -lt 2 ]; then
  echo "usage: [T=trtexec] [FP16=1|0] $0 <model.onnx> <out.engine> [extra trtexec args...]" >&2
  echo "  e.g. $0 mask_decoder.onnx mask_decoder.engine --minShapes=sparse:1x2x256 --optShapes=sparse:1x3x256 --maxShapes=sparse:1x3x256" >&2
  exit 2
fi
ONNX="$1"; OUT="$2"; shift 2
T="${T:-/usr/src/tensorrt/bin/trtexec}"
FP16_FLAG=""; [ "${FP16:-1}" = "1" ] && FP16_FLAG="--fp16"

[ -f "$ONNX" ] || { echo "no such onnx: $ONNX" >&2; exit 1; }
command -v "$T" >/dev/null 2>&1 || [ -x "$T" ] || { echo "no trtexec at: $T" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

echo "building $OUT from $ONNX ${FP16_FLAG} $*"
"$T" --onnx="$ONNX" $FP16_FLAG --saveEngine="$OUT" "$@" 2>&1 \
  | grep -iE "error|fail|passed|engine|tensorrt version" | tail -10 || true

[ -s "$OUT" ] && echo "OK $OUT ($(du -h "$OUT" | cut -f1))" \
  || { echo "FAILED: $OUT not produced" >&2; exit 1; }
