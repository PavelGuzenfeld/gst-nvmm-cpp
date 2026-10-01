# Sourced, not executed. Success is exit status plus CSV row count, never a log grep: an unknown
# property gave silent empty CSVs, and /error/ matched a benign TensorRT notice.
# No -e: a sweep must survive one failed arm, and grep -v exits 1 when it filters everything.
set -uo pipefail

require_env() {
  local missing=() v
  for v in "$@"; do
    if [ -z "${!v:-}" ]; then missing+=("$v"); fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: required environment not set: ${missing[*]}" >&2
    echo "  see tools/samurai/detector-decimation/README.md" >&2
    return 1
  fi
}

# run_pipeline <label> <csv-path> <expected-rows|0 to skip> <gst-launch args...>
run_pipeline() {
  local label=$1 csv=$2 want=$3; shift 3
  local log="${csv%.csv}.gst.log" rc=0
  rm -f "$csv"
  gst-launch-1.0 -e "$@" > "$log" 2>&1 || rc=$?

  if [ "$rc" -ne 0 ]; then
    echo "  $label: FAILED (gst-launch exit $rc)"
    grep -E "erroneous pipeline|no property |no element |^ERROR" "$log" | head -3 | sed 's/^/      /'
    return 1
  fi
  if [ ! -s "$csv" ]; then
    echo "  $label: FAILED (no output at $csv -- pipeline ran but produced nothing)"
    tail -3 "$log" | sed 's/^/      /'
    return 1
  fi
  local rows; rows=$(( $(wc -l < "$csv") - 1 ))
  if [ "$want" -gt 0 ] && [ "$rows" -ne "$want" ]; then
    echo "  $label: FAILED ($rows rows, expected $want -- frames were dropped)"
    return 1
  fi
  echo "  $label: ok ($rows rows)"
}

# docker_gst <label> <host-csv> <expected-rows|0> <pipeline>: run_pipeline in a container, env passed via -e only
docker_gst() {
  local label=$1 csv=$2 want=$3 pipeline=$4
  local log="${csv%.csv}.gst.log" rc=0
  : "${DEPLOY_SRC:?}" "${ASSET_DIR:?}"
  local img=${DOCKER_IMAGE:-gst-nvmm-infer:jp6}
  rm -f "$csv"
  docker run --rm --runtime nvidia --network host \
    -v "$DEPLOY_SRC":/src -v "$ASSET_DIR":/o \
    -e GST_PLUGIN_PATH=/src/builddir-fix \
    -e GST_DEBUG=0 \
    -e NVMMFUSEKF_CSV="/o/${csv#"$ASSET_DIR"/}" \
    "$img" gst-launch-1.0 -e $pipeline > "$log" 2>&1 || rc=$?

  if [ "$rc" -ne 0 ]; then
    echo "  $label: FAILED (gst-launch exit $rc)"
    grep -E "erroneous pipeline|no property |no element |^ERROR" "$log" | head -3 | sed 's/^/      /'
    return 1
  fi
  if [ ! -s "$csv" ]; then
    echo "  $label: FAILED (no output -- pipeline ran but produced nothing)"; return 1
  fi
  local rows; rows=$(( $(wc -l < "$csv") - 1 ))
  if [ "$want" -gt 0 ] && [ "$rows" -ne "$want" ]; then
    echo "  $label: FAILED ($rows rows, expected $want)"; return 1
  fi
  echo "  $label: ok ($rows rows)"
}

# docker_score <pred-dir> <out-md>
docker_score() {
  : "${SCORER:?}" "${GT_LABEL:?}" "${ASSET_DIR:?}"
  docker run --rm --network host -v "$ASSET_DIR":/o -w /o \
    -e SCORER -e GT_LABEL "${DOCKER_IMAGE:-gst-nvmm-infer:jp6}" \
    bash -lc 'python3 "/o/$SCORER" --pred-dir "'"$1"'" --gt /o/seqs/train --gt-name "$GT_LABEL" --out "'"$2"'"'
}

# extract_sequence <name> <archive> <destdir>: echoes the jpg count, 0 if absent; paths via argv, not the quoted heredoc
extract_sequence() {
  python3 - "$1" "$2" "$3" <<'PY'
import sys, zipfile
seq, zip_path, dest = sys.argv[1], sys.argv[2], sys.argv[3]
z = zipfile.ZipFile(zip_path)
mem = [n for n in z.namelist() if n.startswith("train/%s/" % seq)]
if not mem:
    print(0); sys.exit(0)
z.extractall(dest, members=mem)
print(sum(1 for n in mem if n.endswith(".jpg")))
PY
}

nvmm_source_clip()  { echo "filesrc location=$1 ! decodebin ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! queue"; }
nvmm_source_jpegs() { echo "multifilesrc location=$1/%06d.jpg index=1 stop-index=$2 caps=image/jpeg,framerate=25/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! queue ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! queue"; }

PIPELINE_BENCH_PROBE=trk
nvmm_tracker() { echo "nvmmsamurai name=$PIPELINE_BENCH_PROBE engine-dir=$1/trt consts-file=$1/trt/samurai_consts.bin ${2:-} gmc=false ! queue"; }
nvmm_fusekf()  { echo "nvmmfusekf target-class=0 ${1:-} ! fakesink sync=false"; }
nvmm_detgate() { echo "nvmmdetgate target-class=0 border-frac=${1:-0.02} ! queue"; }
# nvmm_detector <asset-dir> [interval=1] [gate=0, omitted so builds predating the property still parse]
nvmm_detector() {
  local asset_dir=$1 interval=${2:-1} gate=${3:-0}
  local args="infer-interval=$interval"
  [ "$gate" != "0" ] && args="$args infer-gate-frames=$gate"
  echo "nvmminfer engine-file=$asset_dir/trt/detector.engine $args ! queue"
}
