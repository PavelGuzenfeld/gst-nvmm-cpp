#!/usr/bin/env bash
# Ground-truth decimation A/B on the deployed chain element for element: the simplified chain
# understated the cost about threefold. Parity is unreachable once a Kalman measurement goes.
# Sequences are extracted and deleted one at a time; all of them never fit on disk at once.
. "$(dirname "$0")/lib.sh"

require_env ASSET_DIR DEPLOY_SRC EVALSET_ZIP SEQ_LIST SCORER GT_LABEL

cd "$ASSET_DIR"
O=$ASSET_DIR
SEQFILE=${SEQFILE:-$SEQ_LIST}
ARMS=${ARMS:-"1 3"}
TEARDOWN=${TEARDOWN:-teardown=true teardown-border-frac=0.05 teardown-border-frames=20}

for IV in $ARMS; do mkdir -p "results/gtd_n$IV"; done

while read -r S; do
  [ -z "$S" ] && continue
  echo "######## $S ########"
  N=$(extract_sequence "$S" "$EVALSET_ZIP" seqs)
  if [ "$N" -le 0 ]; then echo "  MISSING in archive, skipped"; continue; fi
  echo "  extracted $N frames"

  for IV in $ARMS; do
    out=results/gtd_n$IV
    pipe="$(nvmm_source_jpegs "/o/seqs/train/$S" "$N") ! $(nvmm_detector /o "$IV") \
! $(nvmm_detgate) ! $(nvmm_tracker /o "max-kf=2 seed-prefer-center=true kf-vel-noise=0.1") \
! $(nvmm_fusekf "$TEARDOWN")"
    docker_gst "n$IV" "$O/$out/$S.csv" "$N" "$pipe" || true
  done

  find "seqs/train/$S" -name '*.jpg' -delete 2>/dev/null || true
done < "$SEQFILE"

echo "=== score each arm ==="
for IV in $ARMS; do
  docker_score "/o/results/gtd_n$IV" "/o/results/scorecard_gtd_n$IV.md"
done
docker run --rm -v "$O":/o "${DOCKER_IMAGE:-gst-nvmm-infer:jp6}" chmod -R 777 /o/results 2>/dev/null || true
echo "GTD-AB-DONE"
