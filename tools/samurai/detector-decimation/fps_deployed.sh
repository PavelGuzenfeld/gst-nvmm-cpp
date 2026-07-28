#!/usr/bin/env bash
# Throughput at the DEPLOYED pipeline shape -- the denominator that was missing.
#
# Every +34.6%/+57.8% figure in detector-decimation-verdict.md comes from the
# simplified chain (clip1.mp4 -> nvmminfer -> nvmmsamurai -> nvmmfusekf, gmc build,
# 20.20 fps at interval=1). This measures the deployed element list instead
# (nvmmdetgate, kf-vel-noise, fusekf teardown, deploy build) on the SAME clip and the
# same forced seed, so the two are directly comparable and the gains can be checked
# for survival against the extra elements.
set -u
O=$ASSET_DIR
B=$DEPLOY_SRC/builddir-fix
export GST_PLUGIN_PATH=$B GST_DEBUG=0
ITERS=${ITERS:-2}

SRC="filesrc location=$O/clip1.mp4 ! decodebin ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! queue"
TAIL="nvmmdetgate target-class=0 border-frac=0.02 ! queue \
! nvmmsamurai name=trk engine-dir=$O/trt consts-file=$O/trt/samurai_consts.bin max-kf=2 \
    seed-roi=910,490,120,120 kf-vel-noise=0.1 gmc=false ! queue \
! nvmmfusekf target-class=0 teardown=true teardown-border-frac=0.05 teardown-border-frames=20 \
! fakesink sync=false"

# arm label : infer-interval : infer-gate-frames
for A in "n1:1:0" "n2g5:2:5" "n3g5:3:5"; do
  L=${A%%:*}; rest=${A#*:}; IV=${rest%%:*}; GATE=${rest##*:}
  PIPE="$SRC ! nvmminfer engine-file=$O/trt/detector.engine \
        infer-interval=$IV infer-gate-frames=$GATE ! queue ! $TAIL"
  echo "##### deployed shape, $L (interval=$IV gate=$GATE) #####"
  python3 $O/pipeline_bench.py --probe trk --iterations $ITERS --pipeline "$PIPE" 2>&1 \
    | grep -viE "Argus|nvargus|engine plan file|BLOCKING" | tail -5
  echo
done
echo FPS-DEPLOYED-DONE
