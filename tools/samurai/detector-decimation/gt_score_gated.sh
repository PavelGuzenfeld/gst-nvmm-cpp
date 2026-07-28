#!/usr/bin/env bash
# GT A/B of ACQUISITION-GATED decimation at the deployed operating point.
#   arms: (interval, gate) pairs; baseline is results/gtd_n1 from gt_score_ab_deployed.sh
#
# Ungated N=3 cost ~23% GT success with 3 of 12 sequences never acquiring. The gate
# holds decimation off until `gate` consecutive inferred frames have produced a
# detection, and re-arms the hold the moment one produces none -- so acquisition and
# reacquisition run at full detector rate, which is where the damage was.
set -uo pipefail
cd $ASSET_DIR
O=$ASSET_DIR
SRC=$DEPLOY_SRC
IMG=gst-nvmm-infer:jp6
SEQFILE=${SEQFILE:-$SEQ_LIST}
ARMS=${ARMS:-"2:5 3:5"}

for A in $ARMS; do mkdir -p "results/gtg_n${A%%:*}g${A##*:}"; done

while read -r S; do
  [ -z "$S" ] && continue
  echo "######## $S ########"
  # Archive path via argv: this heredoc is quoted, so "$VAR" would stay literal.
  N=$(python3 - "$S" "$EVALSET_ZIP" <<'PY'
import sys, zipfile
s, zip_path = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(zip_path); names = z.namelist()
mem = [n for n in names if n.startswith("train/%s/" % s)]
if not mem:
    print(0); sys.exit(0)
z.extractall("seqs", members=mem)
print(sum(1 for n in mem if n.endswith(".jpg")))
PY
)
  if [ "$N" -le 0 ]; then echo "  MISSING in zip, skipped"; continue; fi

  for A in $ARMS; do
    IV=${A%%:*}; GATE=${A##*:}
    OUT=results/gtg_n${IV}g${GATE}
    docker run --rm --runtime nvidia --network host -v "$SRC":/src -v "$O":/o "$IMG" \
      bash -lc "GST_PLUGIN_PATH=/src/builddir-fix NVMMFUSEKF_CSV=/o/$OUT/$S.csv \
        gst-launch-1.0 -e \
        multifilesrc location=/o/seqs/train/$S/%06d.jpg index=1 stop-index=$N \
          caps=image/jpeg,framerate=25/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! queue ! \
        nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! queue ! \
        nvmminfer engine-file=/o/trt/detector.engine \
                  infer-interval=$IV infer-gate-frames=$GATE ! queue ! \
        nvmmdetgate target-class=0 border-frac=0.02 ! queue ! \
        nvmmsamurai engine-dir=/o/trt consts-file=/o/trt/samurai_consts.bin max-kf=2 \
                    seed-prefer-center=true kf-vel-noise=0.1 gmc=false ! queue ! \
        nvmmfusekf target-class=0 teardown=true teardown-border-frac=0.05 \
                   teardown-border-frames=20 ! fakesink sync=false" \
      > "$OUT/$S.gst.log" 2>&1 || true
    if grep -qE "erroneous pipeline|no property |no element |^ERROR" "$OUT/$S.gst.log"; then
      echo "  n${IV}g${GATE} PIPELINE ERROR ->"
      grep -E "erroneous pipeline|no property |no element |^ERROR" "$OUT/$S.gst.log" | head -2 | sed 's/^/      /'
    else
      echo "  n${IV}g${GATE} rows: $(( $(wc -l < "$OUT/$S.csv" 2>/dev/null || echo 1) - 1 ))"
    fi
  done
  find seqs/train/"$S" -name '*.jpg' -delete 2>/dev/null || true
done < "$SEQFILE"

echo "=== score ==="
DIRS=""
for A in $ARMS; do DIRS="$DIRS gtg_n${A%%:*}g${A##*:}"; done
for d in $DIRS; do
  docker run --rm --network host -v "$O":/o -w /o "$IMG" \
    python3 /o/$SCORER --pred-dir "/o/results/$d" --gt /o/seqs/train \
      --gt-name $GT_LABEL --out "/o/results/scorecard_$d.md"
done
docker run --rm -v "$O":/o "$IMG" chmod -R 777 /o/results 2>/dev/null || true
echo "GATED-AB-DONE"
