#!/usr/bin/env bash
# Host-side upper bound on what decimation can buy: detector present (A) vs removed (B).
# The seed is forced so tracker work is identical; unseeded, arm B never infers at all.
# max-kf=2 is the deployed config; max-kf=0 is worst-case GPU contention.
. "$(dirname "$0")/lib.sh"

require_env ASSET_DIR DEPLOY_SRC

O=$ASSET_DIR
export GST_PLUGIN_PATH="$DEPLOY_SRC/builddir-fix" GST_DEBUG=0
ITERS=${ITERS:-3}
SEED=${SEED:-seed-roi=910,490,120,120}

echo "### power mode ###"; nvpmodel -q 2>/dev/null | head -4; echo

src=$(nvmm_source_clip "$O/clip1.mp4")
sink=$(nvmm_fusekf)

for MAXKF in 2 0; do
  for ARM in A B; do
    if [ "$ARM" = A ]; then
      chain="$src ! $(nvmm_detector "$O")"; lbl="detector PRESENT"
    else
      chain="$src";                          lbl="detector ABSENT"
    fi
    echo "##### max-kf=$MAXKF  ARM=$ARM  ($lbl) #####"
    python3 "$O/pipeline_bench.py" --probe "$PIPELINE_BENCH_PROBE" --iterations "$ITERS" \
      --pipeline "$chain ! $(nvmm_tracker "$O" "max-kf=$MAXKF $SEED") ! $sink" 2>&1 \
      | grep -viE "Argus|nvargus|engine plan file" | tail -12
    echo
  done
done
echo "AB-DONE"
