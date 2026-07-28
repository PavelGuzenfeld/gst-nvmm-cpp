#!/usr/bin/env bash
# Behavioural-parity gate for infer-interval on one clip.
#   quality_ab_clip.sh <clip-basename> <seed-roi x,y,w,h> [seed-delay-frame]
#
# Seed is forced so every arm seeds at the identical box on the identical frame. With
# auto-seed, decimation can delay the first detection by up to N-1 frames and shift
# the whole trajectory, which confounds the comparison.
#
# The seed MUST be a VISUALLY VERIFIED target. Seeding from whatever the detector
# emitted on the first frames put two clips on false positives; the tracker then
# tracked noise, fusekf distance-gated out the REAL detections, and the run produced a
# convincing but entirely bogus "decimation diverges the track" result. Use
# find_targets.sh, then extract the frame and look at it.
. "$(dirname "$0")/lib.sh"

require_env ASSET_DIR REPO_SRC

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
  # Attribute the failures: fusion-flag toggle vs genuine trajectory divergence.
  # Shared with the other harnesses rather than re-implemented inline.
  python3 "$O/explain_nontoggle.py" "$R/${C}_v_n1.csv" "$R/${C}_v_n$N.csv" || true
done
echo "CLIP-DONE-$C"
