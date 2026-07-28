#!/usr/bin/env bash
# Bound the convertible GPU headroom held by the per-frame detector detector.
#
# Arm A = deployed shape (detector on GPU every frame).
# Arm B = same graph, detector removed.
# Both arms force the tracker seed with seed-roi so the TRACKER WORKLOAD IS
# IDENTICAL -- without that, arm B never seeds, does no inference, and the fps
# gain is an artifact rather than a measurement.
#
# max-kf=2 is the deployed config (tracker coasts 2 of 3 frames); max-kf=0 is
# full inference every frame (worst-case GPU contention). The A->B delta bounds
# what a detector-decimation property could ever convert into throughput.
#
# Runs on the HOST (plugins load natively; host has python3-gi).
set -u
O=$ASSET_DIR
export GST_PLUGIN_PATH=$DEPLOY_SRC/builddir-fix
export GST_DEBUG=0
ITERS=${ITERS:-3}

echo "### power mode ###"; nvpmodel -q 2>/dev/null | head -4; echo

# host gst lacks h264parse (plugins-bad); decodebin picks nvv4l2decoder itself.
SRC_CHAIN="filesrc location=$O/clip1.mp4 ! decodebin ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! queue"
YOLO="nvmminfer engine-file=$O/trt/detector.engine ! queue"
SEED="seed-roi=910,490,120,120"

for MAXKF in 2 0; do
  for ARM in A B; do
    if [ "$ARM" = A ]; then CHAIN="$SRC_CHAIN ! $YOLO"; LBL="detector PRESENT"; else CHAIN="$SRC_CHAIN"; LBL="detector ABSENT"; fi
    PIPE="$CHAIN ! nvmmsamurai name=trk engine-dir=$O/trt consts-file=$O/trt/samurai_consts.bin max-kf=$MAXKF $SEED gmc=false ! queue ! nvmmfusekf target-class=0 ! fakesink sync=false"
    echo "##### max-kf=$MAXKF  ARM=$ARM  ($LBL) #####"
    python3 $O/pipeline_bench.py --probe trk --iterations $ITERS --pipeline "$PIPE" 2>&1 \
      | grep -viE "Argus|nvargus|engine plan file" | tail -12
    echo
  done
done
echo "AB-DONE"
