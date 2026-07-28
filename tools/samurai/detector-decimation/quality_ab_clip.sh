#!/usr/bin/env bash
# Behavioural-parity gate for infer-interval on one clip.
#   quality_ab_clip.sh <clip-basename> <seed-roi x,y,w,h> [seed-delay-frame]
# Seed is forced so N=1/2/3 seed at the identical box on the identical frame --
# with auto-seed, decimation can delay the first detection by up to N-1 frames and
# shift the whole trajectory, which would confound the comparison.
#
# The seed MUST be a VISUALLY VERIFIED target. Seeding from whatever the detector
# emitted on the first frames put clip2/clip3 on false positives (cloud and terrain
# texture); the tracker then tracked noise, fusekf distance-gated out the REAL
# detections (clip3: 74% of frames have conf>=0.80 dets but only 3.9% were fused),
# and the run produced a bogus "decimation diverges the track" result.
# seed-delay pins the seed to a frame where the target is known to be present.
set -uo pipefail
C=$1; SEED=$2; DELAY=${3:-0}
O=$ASSET_DIR
B=$REPO_SRC/builddir
R=$O/results/interval_quality
mkdir -p "$R"
export GST_PLUGIN_PATH=$B GST_DEBUG=0

echo "######## clip=$C seed-roi=$SEED seed-delay=$DELAY ########"
for N in 1 2 3; do
  NVMMFUSEKF_CSV=$R/${C}_v_n$N.csv gst-launch-1.0 -e \
    filesrc location=$O/$C.mp4 ! decodebin ! nvvidconv \
    ! "video/x-raw(memory:NVMM),format=NV12" ! queue \
    ! nvmminfer engine-file=$O/trt/detector.engine infer-interval=$N ! queue \
    ! nvmmsamurai engine-dir=$O/trt consts-file=$O/trt/samurai_consts.bin \
        max-kf=2 seed-roi=$SEED seed-delay=$DELAY gmc=false ! queue \
    ! nvmmfusekf target-class=0 ! fakesink sync=false > /dev/null 2>&1
  echo "  N=$N rows: $(( $(wc -l < $R/${C}_v_n$N.csv) - 1 ))"
done

for N in 2 3; do
  echo
  echo "==== $C: infer-interval=$N vs baseline ===="
  python3 $O/trajectory_compare.py --baseline $R/${C}_v_n1.csv --test $R/${C}_v_n$N.csv
  # Attribute the failures: flush-BB toggle vs genuine trajectory divergence.
  python3 - "$R/${C}_v_n1.csv" "$R/${C}_v_n$N.csv" <<'PY'
import csv, statistics as st, sys
def rd(p): return {int(r["frame"]): r for r in csv.DictReader(open(p)) if r.get("frame")}
def iou(a, c):
    ax2, ay2 = float(a["left"])+float(a["width"]), float(a["top"])+float(a["height"])
    cx2, cy2 = float(c["left"])+float(c["width"]), float(c["top"])+float(c["height"])
    ix = max(0.0, min(ax2, cx2) - max(float(a["left"]), float(c["left"])))
    iy = max(0.0, min(ay2, cy2) - max(float(a["top"]), float(c["top"])))
    inter = ix*iy
    u = float(a["width"])*float(a["height"]) + float(c["width"])*float(c["height"]) - inter
    return inter/u if u > 0 else 0.0
b, t = rd(sys.argv[1]), rd(sys.argv[2])
same = [iou(b[f], t[f]) for f in b if f in t and b[f]["yolo_fused"] == t[f]["yolo_fused"]]
diff = [iou(b[f], t[f]) for f in b if f in t and b[f]["yolo_fused"] != t[f]["yolo_fused"]]
for lbl, v in (("yolo_fused AGREES", same), ("yolo_fused DIFFERS", diff)):
    if v:
        print(f"  {lbl}: n={len(v)} median={st.median(v):.4f} below0.9={sum(1 for x in v if x < 0.9)}")
    else:
        print(f"  {lbl}: n=0")
areas = sorted(float(v["width"])*float(v["height"]) for v in b.values())
print(f"  baseline box area px^2: median={st.median(areas):.0f}")
PY
done
echo "CLIP-DONE-$C"
