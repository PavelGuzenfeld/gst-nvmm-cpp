#!/usr/bin/env bash
# fps per infer-interval at max-kf=2 with a forced seed, so the detector is the only variable.
# interval=1000 infers frame 0 only and must match the detector-absent arm, or frames are not skipped.
. "$(dirname "$0")/lib.sh"

require_env ASSET_DIR REPO_SRC

O=$ASSET_DIR
export GST_PLUGIN_PATH="$REPO_SRC/builddir" GST_DEBUG=0
ITERS=${ITERS:-2}
SEED=${SEED:-seed-roi=910,490,120,120}

src=$(nvmm_source_clip "$O/clip1.mp4")
trk=$(nvmm_tracker "$O" "max-kf=2 $SEED")
sink=$(nvmm_fusekf)

for N in 1 2 3 6 1000; do
  echo "##### infer-interval=$N #####"
  python3 "$O/pipeline_bench.py" --probe trk --iterations "$ITERS" \
    --pipeline "$src ! $(nvmm_detector "$O" "$N") ! $trk ! $sink" 2>&1 \
    | grep -viE "Argus|nvargus|engine plan file|BLOCKING" | tail -6
  echo
done
echo "SWEEP-DONE"
