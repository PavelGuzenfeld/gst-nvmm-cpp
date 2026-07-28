#!/usr/bin/env bash
# infer-interval sweep, deployed max-kf=2, forced seed so tracker work is constant.
# interval=1    -> baseline (detector every frame)
# interval=1000 -> functional check: should reproduce the detector-absent arm
#                  (40.3 fps measured in the deploy build) since only frame 0 infers.
set -u
O=$ASSET_DIR
B=$REPO_SRC/builddir
export GST_PLUGIN_PATH=$B GST_DEBUG=0
ITERS=${ITERS:-2}

SRC="filesrc location=$O/clip1.mp4 ! decodebin ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! queue"
TRK="nvmmsamurai name=trk engine-dir=$O/trt consts-file=$O/trt/samurai_consts.bin max-kf=2 seed-roi=910,490,120,120 gmc=false"

for N in 1 2 3 6 1000; do
  PIPE="$SRC ! nvmminfer engine-file=$O/trt/detector.engine infer-interval=$N ! queue ! $TRK ! queue ! nvmmfusekf target-class=0 ! fakesink sync=false"
  echo "##### infer-interval=$N #####"
  python3 $O/pipeline_bench.py --probe trk --iterations $ITERS --pipeline "$PIPE" 2>&1 \
    | grep -viE "Argus|nvargus|engine plan file|BLOCKING" | tail -6
  echo
done
echo "SWEEP-DONE"
