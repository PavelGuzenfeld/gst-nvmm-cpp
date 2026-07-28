#!/usr/bin/env bash
# Score detector decimation against the labelled evaluation set ground truth: infer-interval=1 vs 3.
#
# WHY GT and not parity: decimation removes YOLO as the second measurement to the
# master KF, so the fused box necessarily differs from an undecimated run. Parity
# ("is it identical?") is unachievable by construction; the real question is "is it
# worse?", which only GT answers. Metrics that matter here are success (IoU>0.5 on
# exist==1 frames), state_acc (official the labelled evaluation set), miss and fp_absent_rate.
#
# Runs on the host with the repo-checkout build (the one carrying infer-interval).
# NOTE: the deployed GT harness ($DEPLOY_HARNESS) inserts nvmmdetgate, which does not
# exist in this checkout -- so absolute scores here are NOT comparable to
# results/$DEPLOY_SCORECARD. Both arms are identical apart from infer-interval, so the
# A/B is internally valid, which is all this needs to be.
. "$(dirname "$0")/lib.sh"

require_env ASSET_DIR REPO_SRC EVALSET_ZIP SCORER GT_LABEL

cd "$ASSET_DIR"
B=$REPO_SRC/builddir
SEQS="${SEQS:-seq-C seq-K seq-L seq-B seq-G}"

echo "=== Phase 1: extract $(echo $SEQS | wc -w) sequences ==="
python3 - <<PY
import zipfile
z = zipfile.ZipFile("$EVALSET_ZIP"); names = z.namelist()
with open("manifest_gt_ab.txt", "w") as man:
    for s in "$SEQS".split():
        mem = [n for n in names if n.startswith("train/%s/" % s)]
        if not mem:
            print("MISSING in zip:", s); continue
        z.extractall("seqs", members=mem)
        nj = sum(1 for n in mem if n.endswith(".jpg"))
        man.write("%s\t%d\n" % (s, nj)); print("extracted %s : %d frames" % (s, nj))
PY

export GST_PLUGIN_PATH=$B GST_DEBUG=0
for N in 1 3; do
  OUT=results/gt_n$N; mkdir -p $OUT
  echo "=== Phase 2: run infer-interval=$N ==="
  while IFS=$'\t' read -r seq stop; do
    [ -z "$seq" ] && continue
    # NOTE: no kf-vel-noise= here -- that property exists only in the divergent deploy
    # checkout, and gst-launch hard-rejects an unknown property, so copying the
    # deployed harness's arg list verbatim silently produces zero output.
    export NVMMFUSEKF_CSV="$OUT/$seq.csv"
    # shellcheck disable=SC2086  # the pipeline must word-split into gst-launch args
    run_pipeline "N=$N $seq" "$NVMMFUSEKF_CSV" "$stop" \
      $(nvmm_source_jpegs "seqs/train/$seq" "$stop") \
      ! $(nvmm_detector . "$N") \
      ! $(nvmm_tracker . "max-kf=2 seed-prefer-center=true") \
      ! $(nvmm_fusekf) || true
    unset NVMMFUSEKF_CSV
  done < manifest_gt_ab.txt
done

echo "=== Phase 3: score both arms ==="
for N in 1 3; do
  python3 $SCORER --pred-dir results/gt_n$N --gt seqs/train \
      --gt-name $GT_LABEL --out results/scorecard_gt_n$N.md
done

echo "=== Phase 4: cleanup extracted frames (disk is tight) ==="
find seqs/train -name '*.jpg' -delete 2>/dev/null || true
echo "GT-AB-DONE"
