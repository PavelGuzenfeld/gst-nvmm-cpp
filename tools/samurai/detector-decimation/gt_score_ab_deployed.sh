#!/usr/bin/env bash
# GT A/B of detector decimation at the DEPLOYED operating point, all 12 sequences.
#
# Fixes measurement 6's headline caveat: that run used the repo-checkout build,
# which lacks nvmmdetgate and kf-vel-noise, giving fp_absent_rate=1.000 on 4/5
# sequences and absolute success far below the deployed scorecard. This mirrors
# $DEPLOY_HARNESS element-for-element (detector -> nvmmdetgate -> nvmmsamurai ->
# nvmmfusekf with teardown) against the deploy checkout, which is now patched with
# infer-interval, so results are comparable to results/$DEPLOY_SCORECARD.
#
# Per-sequence extract -> run both arms -> delete frames: the box sits at 94% disk,
# so all 12 sequences must never be on disk at once.
set -uo pipefail
cd $ASSET_DIR
O=$ASSET_DIR
SRC=$DEPLOY_SRC
IMG=gst-nvmm-infer:jp6
SEQFILE=${SEQFILE:-$SEQ_LIST}
mkdir -p results/gtd_n1 results/gtd_n3

while read -r S; do
  [ -z "$S" ] && continue
  echo "######## $S ########"
  # Archive path goes through argv, NOT interpolation: this heredoc is quoted
  # (<<'PY'), so a "$VAR" inside it stays a literal and silently opens a file that
  # does not exist.
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
  echo "  extracted $N frames"

  for IV in 1 3; do
    OUT=results/gtd_n$IV
    docker run --rm --runtime nvidia --network host -v "$SRC":/src -v "$O":/o "$IMG" \
      bash -lc "GST_PLUGIN_PATH=/src/builddir-fix NVMMFUSEKF_CSV=/o/$OUT/$S.csv \
        gst-launch-1.0 -e \
        multifilesrc location=/o/seqs/train/$S/%06d.jpg index=1 stop-index=$N \
          caps=image/jpeg,framerate=25/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! queue ! \
        nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! queue ! \
        nvmminfer engine-file=/o/trt/detector.engine infer-interval=$IV ! queue ! \
        nvmmdetgate target-class=0 border-frac=0.02 ! queue ! \
        nvmmsamurai engine-dir=/o/trt consts-file=/o/trt/samurai_consts.bin max-kf=2 \
                    seed-prefer-center=true kf-vel-noise=0.1 gmc=false ! queue ! \
        nvmmfusekf target-class=0 teardown=true teardown-border-frac=0.05 \
                   teardown-border-frames=20 ! fakesink sync=false" \
      > "$OUT/$S.gst.log" 2>&1 || true
    # gst's own failure forms only -- a bare case-insensitive "error" matches
    # nvmminfer's benign "...may even cause errors" TRT notice.
    if grep -qE "erroneous pipeline|no property |no element |^ERROR" "$OUT/$S.gst.log"; then
      echo "  N=$IV PIPELINE ERROR ->"
      grep -E "erroneous pipeline|no property |no element |^ERROR" "$OUT/$S.gst.log" | head -2 | sed 's/^/      /'
    else
      echo "  N=$IV rows: $(( $(wc -l < "$OUT/$S.csv" 2>/dev/null || echo 1) - 1 ))"
    fi
  done

  # Container wrote the CSVs as root; frames are ours. Delete frames only.
  find seqs/train/"$S" -name '*.jpg' -delete 2>/dev/null || true
done < "$SEQFILE"

echo "=== score both arms ==="
docker run --rm --network host -v "$O":/o -w /o "$IMG" bash -lc '
  for a in 1 3; do
    python3 /o/$SCORER --pred-dir /o/results/gtd_n$a --gt /o/seqs/train \
        --gt-name $GT_LABEL --out /o/results/scorecard_gtd_n$a.md
  done
  chmod -R 777 /o/results 2>/dev/null || true'
echo "GTD-AB-DONE"
