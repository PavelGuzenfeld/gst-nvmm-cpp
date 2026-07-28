#!/usr/bin/env bash
# Throughput at the DEPLOYED pipeline shape -- the denominator the sweep was missing.
#
# interval_sweep.sh measures the simplified chain. This adds the elements deployment
# actually runs (seed gate, kf-vel-noise, fusekf teardown) on the SAME clip with the
# same forced seed, so the two are directly comparable and the gains can be checked
# for survival against the extra elements.
. "$(dirname "$0")/lib.sh"

require_env ASSET_DIR DEPLOY_SRC

O=$ASSET_DIR
export GST_PLUGIN_PATH="$DEPLOY_SRC/builddir-fix" GST_DEBUG=0
ITERS=${ITERS:-2}
SEED=${SEED:-seed-roi=910,490,120,120}
TEARDOWN=${TEARDOWN:-teardown=true teardown-border-frac=0.05 teardown-border-frames=20}

src=$(nvmm_source_clip "$O/clip1.mp4")
tail="$(nvmm_detgate) ! $(nvmm_tracker "$O" "max-kf=2 $SEED kf-vel-noise=0.1") ! $(nvmm_fusekf "$TEARDOWN")"

# label : infer-interval : infer-gate-frames
for A in "n1:1:0" "n2g5:2:5" "n3g5:3:5"; do
  L=${A%%:*}; rest=${A#*:}; IV=${rest%%:*}; GATE=${rest##*:}
  echo "##### deployed shape, $L (interval=$IV gate=$GATE) #####"
  python3 "$O/pipeline_bench.py" --probe trk --iterations "$ITERS" \
    --pipeline "$src ! $(nvmm_detector "$O" "$IV" "$GATE") ! $tail" 2>&1 \
    | grep -viE "Argus|nvargus|engine plan file|BLOCKING" | tail -5
  echo
done
echo FPS-DEPLOYED-DONE
