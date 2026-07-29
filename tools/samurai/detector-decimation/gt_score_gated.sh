#!/usr/bin/env bash
# Ground-truth A/B of ACQUISITION-GATED decimation at the deployed operating point.
#   arms: "interval:gate" pairs. Baseline is results/gtd_n1 from gt_score_ab_deployed.sh.
#
# Ungated decimation cost ~23% GT success with 3 of 12 sequences never acquiring at
# all. The gate holds decimation off until `gate` consecutive inferred frames have
# produced a detection and re-arms the hold the moment one produces none, so
# acquisition and reacquisition run at full detector rate -- which is where the damage
# was. Note the gate keys on ANY detection, not target class: nvmminfer cannot see the
# target class, so it is a proxy for track state, not the real signal.
. "$(dirname "$0")/lib.sh"

require_env ASSET_DIR DEPLOY_SRC EVALSET_ZIP SEQ_LIST SCORER GT_LABEL

cd "$ASSET_DIR"
O=$ASSET_DIR
SEQFILE=${SEQFILE:-$SEQ_LIST}
ARMS=${ARMS:-"2:5 3:5"}
TEARDOWN=${TEARDOWN:-teardown=true teardown-border-frac=0.05 teardown-border-frames=20}

arm_dir() { echo "results/gtg_n${1%%:*}g${1##*:}"; }

for A in $ARMS; do mkdir -p "$(arm_dir "$A")"; done

while read -r S; do
  [ -z "$S" ] && continue
  echo "######## $S ########"
  N=$(extract_sequence "$S" "$EVALSET_ZIP" seqs)
  if [ "$N" -le 0 ]; then echo "  MISSING in archive, skipped"; continue; fi

  for A in $ARMS; do
    IV=${A%%:*}; GATE=${A##*:}; out=$(arm_dir "$A")
    pipe="$(nvmm_source_jpegs "/o/seqs/train/$S" "$N") ! $(nvmm_detector /o "$IV" "$GATE") \
! $(nvmm_detgate) ! $(nvmm_tracker /o "max-kf=2 seed-prefer-center=true kf-vel-noise=0.1") \
! $(nvmm_fusekf "$TEARDOWN")"
    docker_gst "n${IV}g${GATE}" "$O/$out/$S.csv" "$N" "$pipe" || true
  done
  find "seqs/train/$S" -name '*.jpg' -delete 2>/dev/null || true
done < "$SEQFILE"

echo "=== score each arm ==="
for A in $ARMS; do
  d=$(arm_dir "$A")
  docker_score "/o/$d" "/o/results/scorecard_$(basename "$d").md"
done
docker run --rm -v "$O":/o "${DOCKER_IMAGE:-gst-nvmm-infer:jp6}" chmod -R 777 /o/results 2>/dev/null || true
echo "GATED-AB-DONE"
