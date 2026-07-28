# Shared harness helpers. Source, do not execute:  . "$(dirname "$0")/lib.sh"
#
# Exists because the same docker invocation, pipeline string and log-grep were
# copy-pasted across six scripts; adding the deployed element list once meant editing
# three of them.
#
# The important part is run_pipeline(): success is decided by EXIT STATUS and OUTPUT
# ROW COUNT, not by grepping logs for the word "error". Log-grepping cost three
# separate debug cycles here -- an unknown property produced silent empty CSVs that
# read as success, and a case-insensitive /error/ matched a benign TensorRT notice
# that says "may even cause errors". Outcome checks cannot make either mistake.

# pipefail matters: every harness pipes gst-launch into grep/tail, and without it the
# pipeline's status is grep's, so a crashed gst-launch reports success.
#
# Deliberately NOT -e. These harnesses sweep arms and must survive one arm failing to
# report the rest, and `grep -v` legitimately exits 1 when it filters everything out --
# with -e plus pipefail that ends the sweep after the first arm. run_pipeline captures
# its own exit status explicitly, so it does not need -e to detect failure.
set -uo pipefail

require_env() {  # require_env VAR...  -- fail loudly and early, naming what is missing
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

# run_pipeline <label> <csv-path> <expected-rows|0> <gst-launch args...>
#
# Returns non-zero and explains itself on: non-zero exit, missing CSV, empty CSV, or a
# row count that disagrees with the frame count. Pass 0 for expected-rows to skip the
# count check when the frame total is not known up front.
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

# docker_gst <label> <host-csv> <expected-rows|0> <pipeline>
#
# Same outcome contract as run_pipeline, for the containerised path. Two traps this
# closes, both of which shipped as silent no-ops:
#   - a single-quoted `bash -lc '...$VAR...'` does NOT expand on the host, and the
#     container has no such variable, so the command ran with empty arguments;
#   - `|| true` plus log-grepping reported success for a pipeline that produced
#     nothing.
# Everything the container needs is passed explicitly with -e.
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

# docker_score <pred-dir> <out-md>  -- scorer name and label file come from the
# environment and are passed with -e, not interpolated into a quoted string.
docker_score() {
  : "${SCORER:?}" "${GT_LABEL:?}" "${ASSET_DIR:?}"
  docker run --rm --network host -v "$ASSET_DIR":/o -w /o \
    -e SCORER -e GT_LABEL "${DOCKER_IMAGE:-gst-nvmm-infer:jp6}" \
    bash -lc 'python3 "/o/$SCORER" --pred-dir "'"$1"'" --gt /o/seqs/train --gt-name "$GT_LABEL" --out "'"$2"'"'
}

# extract_sequence <name> <archive> <destdir>  -- echoes the jpg count, 0 if absent.
# Archive path goes through argv because the heredoc is quoted; interpolating "$VAR"
# into a quoted heredoc silently yields a literal filename, which broke three
# harnesses at once.
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

# Composable element chains -- one stage each, so a change to the deployed shape
# happens in one place while callers stay free to vary max-kf, insert the gate
# element, or swap the sink. A single monolithic "tail" builder could not express
# the arms these harnesses actually sweep.
#
# host gst lacks h264parse (plugins-bad), so clips go through decodebin, which picks
# nvv4l2decoder itself.
nvmm_source_clip()  { echo "filesrc location=$1 ! decodebin ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! queue"; }
nvmm_source_jpegs() { echo "multifilesrc location=$1/%06d.jpg index=1 stop-index=$2 caps=image/jpeg,framerate=25/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! queue ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! queue"; }

# $1 asset dir, $2 extra nvmmsamurai args (max-kf, seed, kf-vel-noise, ...).
# Named 'trk' because pipeline_bench.py counts buffers at that pad.
nvmm_tracker() { echo "nvmmsamurai name=trk engine-dir=$1/trt consts-file=$1/trt/samurai_consts.bin ${2:-} gmc=false ! queue"; }
# $1 extra nvmmfusekf args (teardown, flush-carry, ...).
nvmm_fusekf()  { echo "nvmmfusekf target-class=0 ${1:-} ! fakesink sync=false"; }
# Seed-gate element, present only in the deployed shape.
nvmm_detgate() { echo "nvmmdetgate target-class=0 border-frac=${1:-0.02} ! queue"; }
# $1 asset dir, $2 infer-interval, $3 infer-gate-frames.
# Emit a property ONLY when it is not the default: gst-launch hard-rejects an unknown
# property, so passing "infer-gate-frames=0" to a build that predates the gate fails
# the whole pipeline for a setting that means "off". Omitting the default keeps the
# harnesses runnable against older builds.
nvmm_detector() {
  local args="infer-interval=${2:-1}"
  [ "${3:-0}" != "0" ] && args="$args infer-gate-frames=$3"
  echo "nvmminfer engine-file=$1/trt/detector.engine $args ! queue"
}
# $1 asset dir, $2 extra nvmmsamurai args, $3 extra nvmmfusekf args (optional).
# Element is named 'trk' because pipeline_bench.py counts buffers at that pad.
nvmm_tracker_tail() {
  echo "nvmmsamurai name=trk engine-dir=$1/trt consts-file=$1/trt/samurai_consts.bin max-kf=2 ${2:-} gmc=false ! queue ! nvmmfusekf target-class=0 ${3:-} ! fakesink sync=false"
}
