#!/usr/bin/env bash
# Does nvmmfusekf flush-carry unlock detector decimation?
#   flush_carry_ab.sh <clip> <seed-roi> <seed-delay>
#
# Two questions:
#   1. NEUTRALITY -- at infer-interval=1 (nothing decimated), does flush-carry change
#      what is published? It fires on frames where the detector simply found nothing,
#      so it is NOT a decimation-only change and must be characterised on its own.
#   2. PARITY -- at infer-interval=3, does carry bring the emitted box back to the
#      undecimated baseline (median IoU >= 0.99, no frame < 0.9)?
# Baseline for both is (interval=1, carry=0) = today's deployed behaviour.
set -uo pipefail
C=$1; SEED=$2; DELAY=${3:-0}
O=$ASSET_DIR
B=$REPO_SRC/builddir
R=$O/results/flush_carry
mkdir -p "$R"
export GST_PLUGIN_PATH=$B GST_DEBUG=0

run() { # <interval> <carry> <tag>
  NVMMFUSEKF_CSV=$R/${C}_$3.csv gst-launch-1.0 -e \
    filesrc location=$O/$C.mp4 ! decodebin ! nvvidconv \
    ! "video/x-raw(memory:NVMM),format=NV12" ! queue \
    ! nvmminfer engine-file=$O/trt/detector.engine infer-interval=$1 ! queue \
    ! nvmmsamurai engine-dir=$O/trt consts-file=$O/trt/samurai_consts.bin \
        max-kf=2 seed-roi=$SEED seed-delay=$DELAY gmc=false ! queue \
    ! nvmmfusekf target-class=0 flush-carry=$2 ! fakesink sync=false > /dev/null 2>&1
  echo "  $3 (interval=$1 carry=$2): $(( $(wc -l < $R/${C}_$3.csv) - 1 )) rows"
}

echo "######## $C seed=$SEED delay=$DELAY ########"
run 1 0 n1c0
run 1 2 n1c2
run 3 0 n3c0
run 3 2 n3c2
run 3 5 n3c5

for T in n1c2 n3c0 n3c2 n3c5; do
  echo
  echo "==== $T vs baseline n1c0 ===="
  python3 $O/trajectory_compare.py --baseline $R/${C}_n1c0.csv --test $R/${C}_$T.csv
done
echo "CARRY-DONE-$C"
