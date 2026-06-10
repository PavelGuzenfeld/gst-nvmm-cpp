#!/usr/bin/env bash
# Golden-reference validation for nvmminfer. Run ON the Jetson.
# Manual hardware gate — NOT in CI: needs the nvidia runtime, a prebuilt TRT
# engine and the ~/yolo assets, none of which exist on the GitHub runners.
#
# Produces an independent CPU reference (onnxruntime on the SAME onnx the TRT
# engine was built from) and compares it, box-by-box, to nvmminfer's TRT output
# on the same image — guarding against silent preprocess/parser regressions.
#
# Env (override as needed):
#   ONNX     reference onnx model     (default ~/yolo/yolo11n.onnx)
#   ENGINE   TensorRT engine          (default ~/yolo/yolo11n_fp16.engine)
#   IMG      test image               (default ~/yolo/bus.jpg)
#   IMGSZ    network size             (default 640)
#   CONF IOU CONF_TOL                 (compare thresholds; defaults 0.3/0.5/0.15)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="${ONNX:-$HOME/yolo/yolo11n.onnx}"
ENGINE="${ENGINE:-$HOME/yolo/yolo11n_fp16.engine}"
IMG="${IMG:-$HOME/yolo/bus.jpg}"
IMGSZ="${IMGSZ:-640}"
CONF="${CONF:-0.3}"; IOU="${IOU:-0.5}"; CONF_TOL="${CONF_TOL:-0.15}"
INFER_IMAGE="${INFER_IMAGE:-gst-nvmm-infer:jp6}"
REF_IMAGE="${REF_IMAGE:-yolo-reference:cpu}"
OUT="${OUT:-$HOME/yolo}"

fail() { echo "GOLDEN FAIL: $1" >&2; exit 1; }
for f in "$ONNX" "$ENGINE" "$IMG"; do [ -f "$f" ] || fail "missing: $f"; done

GST_PLUGIN_PATH=/src/builddir-docker/gst/nvmminfer:/src/builddir-docker/gst/nvmmdrawdet:/src/builddir-docker/gst/nvmmalloc

echo "== [1/4] build reference + infer images =="
docker build --network=host -f "$ROOT/docker/Dockerfile.yolo-reference" \
  -t "$REF_IMAGE" "$ROOT" >/dev/null || fail "reference image build failed"
docker build --network=host -f "$ROOT/docker/Dockerfile.jetson-jp6-infer" \
  -t "$INFER_IMAGE" "$ROOT" >/dev/null || fail "infer image build failed"

echo "== [2/4] independent CPU reference (onnxruntime) =="
docker run --rm --network host -v "$ROOT":/work -v "$OUT":/data "$REF_IMAGE" \
  python3 /work/tools/yolo_reference.py --onnx /data/"$(basename "$ONNX")" \
  --image /data/"$(basename "$IMG")" --imgsz "$IMGSZ" --conf "$CONF" \
  > "$OUT/golden_ref.json" || fail "reference run failed"
echo "   wrote $OUT/golden_ref.json"

echo "== [3/4] nvmminfer TRT detections =="
docker run --rm --runtime nvidia --network host -v "$ROOT":/src -v "$OUT":/data -w /src \
  -e GST_PLUGIN_PATH="$GST_PLUGIN_PATH" "$INFER_IMAGE" bash -c \
  "meson setup builddir-docker -Dbuildtype=debugoptimized -Dwerror=false 2>/dev/null; ninja -C builddir-docker >/dev/null 2>&1;
   GST_DEBUG=nvmminfer:6 gst-launch-1.0 filesrc location=/data/$(basename "$IMG") \
     ! jpegdec ! videoconvert ! 'video/x-raw,format=NV12' ! nvvidconv \
     ! 'video/x-raw(memory:NVMM),format=NV12' \
     ! nvmminfer engine-file=/data/$(basename "$ENGINE") ! fakesink" \
  2>"$OUT/nvmminfer.log" || fail "nvmminfer run failed"
echo "   detections: $(grep -cE '\[[0-9]+\] .* box=' "$OUT/nvmminfer.log") logged"

echo "== [4/4] compare within IoU>=$IOU, conf-tol=$CONF_TOL =="
docker run --rm --network host -v "$ROOT":/work -v "$OUT":/data "$REF_IMAGE" \
  python3 /work/tools/golden_compare.py --reference /data/golden_ref.json \
  --actual-log /data/nvmminfer.log --conf "$CONF" --iou "$IOU" --conf-tol "$CONF_TOL"
