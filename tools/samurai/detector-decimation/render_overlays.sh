#!/usr/bin/env bash
# Overlay videos, baseline vs gated N=3, for clip1 (forced seed) and seq-F, the GT sequence the
# gate did not protect (auto-seed, as scored). Encoding runs in the container: the host lacks
# h264parse.
. "$(dirname "$0")/lib.sh"

require_env ASSET_DIR DEPLOY_SRC EVALSET_ZIP

O=$ASSET_DIR
SRC=$DEPLOY_SRC
IMG=${DOCKER_IMAGE:-gst-nvmm-infer:jp6}
OUT=$O/overlays
mkdir -p "$OUT"

DRAW="nvmmdrawdet draw-track=true draw-det=false thickness=2 ! queue \
! nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' \
! nvv4l2h264enc bitrate=8000000 ! h264parse ! qtmux"

TRK_SEEDED="max-kf=2 seed-roi=910,490,120,120 kf-vel-noise=0.1"
TRK_AUTO="max-kf=2 seed-prefer-center=true kf-vel-noise=0.1"
FUSE="nvmmfusekf target-class=0 teardown=true teardown-border-frac=0.05 teardown-border-frames=20"

render() {
  local L=$1 SRCCHAIN=$2 TRK=$3 IV=$4 GATE=$5
  echo "=== render $L (interval=$IV gate=$GATE) ==="
  docker run --rm --runtime nvidia --network host -v "$SRC":/src -v "$O":/o "$IMG" \
    bash -lc "export GST_PLUGIN_PATH=/src/builddir-fix GST_DEBUG=0
      gst-launch-1.0 -e $SRCCHAIN \
      ! $(nvmm_detector /o "$IV" "$GATE") \
      ! $(nvmm_detgate) \
      ! nvmmsamurai engine-dir=/o/trt consts-file=/o/trt/samurai_consts.bin $TRK gmc=false \
      ! queue ! $FUSE ! queue ! $DRAW \
      ! filesink location=/o/overlays/$L.mp4" > "$OUT/$L.log" 2>&1 || true
  if [ -s "$OUT/$L.mp4" ]; then
    echo "  ok: $(du -h "$OUT/$L.mp4" | cut -f1)"
  else
    echo "  FAILED ->"; grep -E "erroneous pipeline|no property |no element |^ERROR" "$OUT/$L.log" | head -3 | sed 's/^/      /'
  fi
}

WA1=$(nvmm_source_clip /o/clip1.mp4)
render clip1_n1     "$WA1" "$TRK_SEEDED" 1 0
render clip1_n3gate "$WA1" "$TRK_SEEDED" 3 5

S=seq-F
if [ ! -f "$O/seqs/train/$S/000001.jpg" ]; then
  echo "=== extract $S ==="
  ( cd "$O" && echo "extracted $(extract_sequence "$S" "$EVALSET_ZIP" seqs) frames" )
fi
NF=$(ls "$O/seqs/train/$S"/*.jpg 2>/dev/null | wc -l)
if [ "$NF" -gt 0 ]; then
  JPG=$(nvmm_source_jpegs "/o/seqs/train/$S" "$NF")
  render seq-F_n1     "$JPG" "$TRK_AUTO" 1 0
  render seq-F_n3gate "$JPG" "$TRK_AUTO" 3 5
else
  echo "!! $S not extracted, skipped"
fi

docker run --rm -v "$O":/o "$IMG" chmod -R 777 /o/overlays 2>/dev/null || true
ls -la "$OUT"/*.mp4 2>/dev/null
echo RENDER-DONE
