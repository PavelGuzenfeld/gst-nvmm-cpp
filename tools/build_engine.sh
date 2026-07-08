#!/usr/bin/env bash
# Generic on-device TensorRT engine builder: wrap trtexec to turn ANY .onnx into
# a serialized .engine, on the box (and at the TensorRT version) the consuming
# element links against. Not model-specific -- it builds a YOLO detector, a
# SAMURAI segment, or anything else you hand it.
#
# A TensorRT engine is version- and hardware-locked: build it with the same
# trtexec (same TensorRT major.minor) that the element loads it with, on the same
# GPU arch, or IRuntime::deserializeCudaEngine rejects it ("Version tag does not
# match"). That is the whole reason this is an on-device step, not a build-host
# artifact.
#
#   build_engine.sh <model.onnx> <out.engine> [extra trtexec args...]
#
# Env knobs:
#   T     trtexec path (default: /usr/src/tensorrt/bin/trtexec)
#   FP16  1 = pass --fp16 (default), 0 = full precision
#
# Examples:
#   build_engine.sh yolo.onnx yolo.engine
#   # dynamic input axis (e.g. a decoder whose sparse-prompt count varies):
#   build_engine.sh mask_decoder.onnx mask_decoder.engine \
#     --minShapes=sparse:1x2x256 --optShapes=sparse:1x3x256 --maxShapes=sparse:1x3x256
set -eu
if [ "$#" -lt 2 ]; then
  echo "usage: $0 <model.onnx> <out.engine> [extra trtexec args...]" >&2
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
