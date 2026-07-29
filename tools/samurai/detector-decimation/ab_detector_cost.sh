#!/usr/bin/env bash
# Bound the convertible GPU headroom held by the per-frame detector.
#
# Arm A = deployed shape (detector on GPU every frame). Arm B = detector removed.
# Both arms force the tracker seed, so the TRACKER WORKLOAD IS IDENTICAL and the
# detector is the only variable. Without the forced seed, arm B never seeds, does no
# inference at all, and its fps is an artifact rather than a measurement.
#
# max-kf=2 is the deployed config (tracker coasts 2 of 3 frames); max-kf=0 is full
# inference every frame, i.e. worst-case GPU contention. The A->B delta bounds what
# decimation could ever convert into throughput.
#
# Runs on the HOST: the plugins load natively and the host has python3-gi.
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
    python3 "$O/pipeline_bench.py" --probe trk --iterations "$ITERS" \
      --pipeline "$chain ! $(nvmm_tracker "$O" "max-kf=$MAXKF $SEED") ! $sink" 2>&1 \
      | grep -viE "Argus|nvargus|engine plan file" | tail -12
    echo
  done
done
echo "AB-DONE"
