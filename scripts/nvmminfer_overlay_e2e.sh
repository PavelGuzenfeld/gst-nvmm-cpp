#!/usr/bin/env bash
# End-to-end test + demo for the nvmminfer detector and the nvmmdrawdet overlay.
# Run ON the Jetson (needs /dev/nvmap, TensorRT, CUDA). Produces:
#   - $OUT/bus_annotated.jpg : a single annotated frame (visual proof)
#   - $OUT/annotated.mp4      : a short annotated H.264 clip
# and verifies the detector emits detections and the encoded clip is non-trivial.
#
# Env (override as needed):
#   ENGINE  TensorRT engine file      (default ~/yolo/yolo11n_fp16.engine)
#   IMG     test image                (default ~/yolo/bus.jpg)
#   OUT     output dir                (default ~/yolo)
#   BUILD   meson build dir           (default builddir-nvmminfer)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE="${ENGINE:-$HOME/yolo/yolo11n_fp16.engine}"
IMG="${IMG:-$HOME/yolo/bus.jpg}"
OUT="${OUT:-$HOME/yolo}"
BUILD="${BUILD:-builddir-nvmminfer}"
export PATH="$HOME/.local/bin:$PATH"
export GST_PLUGIN_PATH="$ROOT/$BUILD/gst/nvmminfer:$ROOT/$BUILD/gst/nvmmdrawdet:$ROOT/$BUILD/gst/nvmmalloc"

fail() { echo "E2E FAIL: $1"; exit 1; }
[ -f "$ENGINE" ] || fail "engine not found: $ENGINE (build with trtexec)"
[ -f "$IMG" ]    || fail "test image not found: $IMG"
mkdir -p "$OUT"

echo "== [1/3] detector emits detections =="
DETS=$(GST_DEBUG=nvmminfer:6 timeout 60 gst-launch-1.0 \
  filesrc location="$IMG" ! jpegdec ! videoconvert ! 'video/x-raw,format=NV12' \
  ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! nvmminfer engine-file="$ENGINE" ! fakesink 2>&1 \
  | grep -oE '[0-9]+ detections' | head -1 | grep -oE '^[0-9]+')
[ "${DETS:-0}" -gt 0 ] 2>/dev/null || fail "no detections produced"
echo "   detections: $DETS"

echo "== [2/3] annotated still (visual proof) =="
timeout 60 gst-launch-1.0 \
  filesrc location="$IMG" ! jpegdec ! videoconvert ! 'video/x-raw,format=NV12' \
  ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! nvmminfer engine-file="$ENGINE" ! nvmmdrawdet thickness=6 \
  ! videoconvert ! jpegenc ! filesink location="$OUT/bus_annotated.jpg" >/dev/null 2>&1
[ -s "$OUT/bus_annotated.jpg" ] || fail "annotated JPEG not produced"
echo "   wrote $OUT/bus_annotated.jpg"

echo "== [3/3] annotated H.264 clip =="
timeout 90 gst-launch-1.0 -e \
  filesrc location="$IMG" ! jpegdec ! videoconvert ! imagefreeze num-buffers=60 \
  ! 'video/x-raw,format=NV12,framerate=30/1' \
  ! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! nvmminfer engine-file="$ENGINE" ! nvmmdrawdet \
  ! videoconvert ! x264enc ! mp4mux ! filesink location="$OUT/annotated.mp4" >/dev/null 2>&1
[ -s "$OUT/annotated.mp4" ] || fail "annotated.mp4 not produced"
SZ=$(stat -c%s "$OUT/annotated.mp4")
[ "$SZ" -gt 10000 ] || fail "annotated.mp4 suspiciously small ($SZ bytes)"
echo "   wrote $OUT/annotated.mp4 ($SZ bytes)"

echo "E2E PASS  (detections=$DETS, still + mp4 produced)"
echo
echo "Watch live from a remote machine:"
echo "  1) ssh -L 6000:localhost:6000 nvidia@<jetson>   # keep open; in that shell:"
echo "     export GST_PLUGIN_PATH=$GST_PLUGIN_PATH"
echo "     gst-launch-1.0 -e filesrc location=$IMG ! jpegdec ! videoconvert ! imagefreeze \\"
echo "       ! 'video/x-raw,format=NV12,framerate=30/1' ! nvvidconv \\"
echo "       ! 'video/x-raw(memory:NVMM),format=NV12' ! nvmminfer engine-file=$ENGINE \\"
echo "       ! nvmmdrawdet ! videoconvert ! x264enc tune=zerolatency \\"
echo "       ! matroskamux streamable=true ! tcpserversink host=0.0.0.0 port=6000"
echo "  2) on the local machine:"
echo "     gst-launch-1.0 tcpclientsrc host=localhost port=6000 ! matroskademux \\"
echo "       ! decodebin ! videoconvert ! autovideosink"
echo "  (swap the filesrc/imagefreeze source for nvarguscamerasrc or a real video.)"
