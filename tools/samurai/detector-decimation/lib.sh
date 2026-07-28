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
set -euo pipefail

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

# Standard element chains, so a change to the deployed shape happens in one place.
nvmm_source_clip()  { echo "filesrc location=$1 ! decodebin ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! queue"; }
nvmm_source_jpegs() { echo "multifilesrc location=$1/%06d.jpg index=1 stop-index=$2 caps=image/jpeg,framerate=25/1 ! jpegparse ! nvv4l2decoder mjpeg=1 ! queue ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! queue"; }
nvmm_detector()     { echo "nvmminfer engine-file=$1/trt/detector.engine infer-interval=${2:-1} infer-gate-frames=${3:-0} ! queue"; }
nvmm_tracker_tail() { echo "nvmmsamurai engine-dir=$1/trt consts-file=$1/trt/samurai_consts.bin max-kf=2 $2 gmc=false ! queue ! nvmmfusekf target-class=0 ! fakesink sync=false"; }
