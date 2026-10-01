#!/usr/bin/env bash
# Does flush-carry restore parity at interval=3, and is it neutral at interval=1? It also fires
# where the detector found nothing, so it is not decimation-only. Baseline: interval=1, carry=0.
. "$(dirname "$0")/lib.sh"

require_env ASSET_DIR REPO_SRC
[ "$#" -ge 2 ] || { echo "usage: $0 <clip> <seed-roi> [seed-delay]" >&2; exit 2; }

C=$1; SEED=$2; DELAY=${3:-0}
O=$ASSET_DIR
R=$O/results/flush_carry
mkdir -p "$R"
export GST_PLUGIN_PATH="$REPO_SRC/builddir" GST_DEBUG=0

src=$(nvmm_source_clip "$O/$C.mp4")

run() {
  local interval=$1 carry=$2 tag=$3
  export NVMMFUSEKF_CSV="$R/${C}_$tag.csv"
  # shellcheck disable=SC2086  # the pipeline must word-split into gst-launch args
  run_pipeline "$tag (interval=$interval carry=$carry)" "$NVMMFUSEKF_CSV" 0 \
    $src ! $(nvmm_detector "$O" "$interval") \
    ! $(nvmm_tracker "$O" "max-kf=2 seed-roi=$SEED seed-delay=$DELAY") \
    ! $(nvmm_fusekf "flush-carry=$carry") || true
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
