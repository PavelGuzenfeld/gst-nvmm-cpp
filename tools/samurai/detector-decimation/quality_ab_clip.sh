#!/usr/bin/env bash
# Forced seed: auto-seed lets decimation delay the first detection and shift the trajectory.
# The seed must be visually verified (find_targets.sh); an unverified one tracked a false
# positive and produced a bogus "decimation diverges the track" result.
. "$(dirname "$0")/lib.sh"

require_env ASSET_DIR REPO_SRC
[ "$#" -ge 2 ] || { echo "usage: $0 <clip-basename> <seed-roi x,y,w,h> [seed-delay-frame]" >&2; exit 2; }

C=$1; SEED=$2; DELAY=${3:-0}
O=$ASSET_DIR
R=$O/results/interval_quality
mkdir -p "$R"
export GST_PLUGIN_PATH="$REPO_SRC/builddir" GST_DEBUG=0

src=$(nvmm_source_clip "$O/$C.mp4")
sink=$(nvmm_fusekf)

echo "######## clip=$C seed-roi=$SEED seed-delay=$DELAY ########"
for N in 1 2 3; do
  export NVMMFUSEKF_CSV="$R/${C}_v_n$N.csv"
  # shellcheck disable=SC2086  # pipeline must word-split into gst-launch args
  run_pipeline "N=$N" "$NVMMFUSEKF_CSV" 0 \
    $src ! $(nvmm_detector "$O" "$N") \
    ! $(nvmm_tracker "$O" "max-kf=2 seed-roi=$SEED seed-delay=$DELAY") ! $sink || true
done
unset NVMMFUSEKF_CSV

for N in 2 3; do
  echo
  echo "==== $C: infer-interval=$N vs baseline ===="
  python3 "$O/trajectory_compare.py" --baseline "$R/${C}_v_n1.csv" --test "$R/${C}_v_n$N.csv" || true
  python3 "$O/explain_nontoggle.py" "$R/${C}_v_n1.csv" "$R/${C}_v_n$N.csv" || true
done
echo "CLIP-DONE-$C"
