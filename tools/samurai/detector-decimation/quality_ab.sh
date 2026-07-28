#!/usr/bin/env bash
# Behavioural-parity gate for nvmminfer infer-interval, at the deployed max-kf=2.
# Dumps nvmmfusekf's EMITTED per-frame box (that is what downstream receives, and
# where decimation's cost lands) for interval=1/2/3, then diffs 2 and 3 vs 1.
# Bar: per-frame IoU >= 0.99 median, no frame < 0.9, no valid-flag flips.
set -uo pipefail
O=$ASSET_DIR
B=$REPO_SRC/builddir
R=$O/results/interval_quality
mkdir -p "$R"
export GST_PLUGIN_PATH=$B GST_DEBUG=0

for N in 1 2 3; do
  echo "### run infer-interval=$N ###"
  NVMMFUSEKF_CSV=$R/n$N.csv gst-launch-1.0 -e \
    filesrc location=$O/clip1.mp4 ! decodebin ! nvvidconv \
    ! "video/x-raw(memory:NVMM),format=NV12" ! queue \
    ! nvmminfer engine-file=$O/trt/detector.engine infer-interval=$N ! queue \
    ! nvmmsamurai engine-dir=$O/trt consts-file=$O/trt/samurai_consts.bin \
        max-kf=2 seed-roi=910,490,120,120 gmc=false ! queue \
    ! nvmmfusekf target-class=0 ! fakesink sync=false \
    > /dev/null 2>&1
  echo "  rows: $(( $(wc -l < $R/n$N.csv) - 1 ))"
done

for N in 2 3; do
  echo
  echo "======== infer-interval=$N vs baseline (interval=1) ========"
  python3 $O/trajectory_compare.py --baseline $R/n1.csv --test $R/n$N.csv --verbose
  echo "exit=$?"
done
echo QUALITY-DONE
