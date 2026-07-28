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
. "$(dirname "$0")/lib.sh"

require_env ASSET_DIR REPO_SRC

C=$1; SEED=$2; DELAY=${3:-0}
O=$ASSET_DIR
R=$O/results/flush_carry
mkdir -p "$R"
export GST_PLUGIN_PATH="$REPO_SRC/builddir" GST_DEBUG=0

src=$(nvmm_source_clip "$O/$C.mp4")

run() { # <interval> <carry> <tag>
  export NVMMFUSEKF_CSV="$R/${C}_$3.csv"
  # shellcheck disable=SC2086  # the pipeline must word-split into gst-launch args
  run_pipeline "$3 (interval=$1 carry=$2)" "$NVMMFUSEKF_CSV" 0 \
    $src ! $(nvmm_detector "$O" "$1") \
    ! $(nvmm_tracker "$O" "max-kf=2 seed-roi=$SEED seed-delay=$DELAY") \
    ! $(nvmm_fusekf "flush-carry=$2") || true
  unset NVMMFUSEKF_CSV
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
