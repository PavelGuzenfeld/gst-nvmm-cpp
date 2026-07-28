#!/usr/bin/env bash
# Find the highest-confidence detections per clip WITH frame numbers, so a seed ROI
# can be visually verified instead of taken from whatever the detector emitted first
# (which on clip2/clip3 turned out to be false positives on cloud/terrain texture).
set -u
export GST_PLUGIN_PATH=$REPO_SRC/builddir
for C in clip2 clip3; do
  GST_DEBUG=nvmminfer:6 gst-launch-1.0 -q \
    filesrc location=$ASSET_DIR/$C.mp4 ! decodebin ! nvvidconv \
    ! "video/x-raw(memory:NVMM),format=NV12" ! queue \
    ! nvmminfer engine-file=$ASSET_DIR/trt/detector.engine \
    ! fakesink sync=false 2>&1 \
  | awk '
      /frame [0-9]+: [0-9]+ detections/ { match($0, /frame [0-9]+/); f = substr($0, RSTART+6, RLENGTH-6) }
      /box=\(/ {
        match($0, /\] [a-z]+ [0-9.]+/); s = substr($0, RSTART, RLENGTH); split(s, a, " ")
        match($0, /box=\([0-9.]+,[0-9.]+ [0-9.]+x[0-9.]+\)/); b = substr($0, RSTART+5, RLENGTH-5)
        print a[3], f, b
      }' > /tmp/${C}_fdets.txt
  n=$(wc -l < /tmp/${C}_fdets.txt)
  echo "=== $C: $n detections total ==="
  echo "  conf>=0.80: $(awk '$1>=0.80' /tmp/${C}_fdets.txt | wc -l)  >=0.70: $(awk '$1>=0.70' /tmp/${C}_fdets.txt | wc -l)  >=0.50: $(awk '$1>=0.50' /tmp/${C}_fdets.txt | wc -l)"
  echo "  top 8 (conf frame box):"
  sort -k1 -rn /tmp/${C}_fdets.txt | head -8 | sed 's/^/    /'
done
echo TARGETS-DONE
