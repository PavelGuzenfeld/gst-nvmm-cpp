#!/bin/bash
# Jetson hardware validation script.
# Run on a Jetson device with JetPack 5.1.1+ (L4T R35.3.1+) or JetPack 6 and GStreamer installed.
#
# Usage:
#   ./scripts/jetson-test.sh           # full test suite
#   ./scripts/jetson-test.sh --quick   # unit tests only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD="$ROOT/builddir"
OUT="$ROOT/test_output"
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:${BUILD}/gst/nvmmalloc${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GST_PLUGIN_PATH="${BUILD}/gst/nvmmconvert:${BUILD}/gst/nvmmsink:${BUILD}/gst/nvmmappsrc:${BUILD}/gst/nvmmalloc"

PASS=0
FAIL=0
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

pass() { echo "  PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

run_pipeline() {
    local name="$1"; shift
    timeout 10 gst-launch-1.0 -e "$@" 2>/dev/null
    if [ $? -eq 0 ]; then pass "$name"; else fail "$name"; fi
}

echo "=== gst-nvmm-cpp Jetson Validation ==="
echo "Device: $(cat /proc/device-tree/model 2>/dev/null || echo unknown)"
echo "L4T:    $(head -1 /etc/nv_tegra_release 2>/dev/null | sed 's/.*R\([0-9]*\).*/R\1/' || echo unknown)"
echo ""

# --- Build ---
echo "--- Build ---"
if [ ! -d "$BUILD" ]; then
    ~/.local/bin/meson setup "$BUILD" -Dcpp_std=c++14 -Dbuildtype=debugoptimized -Dwerror=false
fi
ninja -C "$BUILD"
echo ""

# --- Clear GStreamer cache ---
rm -f ~/.cache/gstreamer-1.0/registry.*.bin

# --- Unit tests ---
echo "--- Unit Tests ---"
~/.local/bin/meson test -C "$BUILD" --print-errorlogs
echo ""

[ $QUICK -eq 1 ] && { echo "Quick mode: $PASS passed, $FAIL failed"; exit $FAIL; }

# --- Pipeline tests ---
echo "--- Pipeline Tests ---"
mkdir -p "$OUT"

run_pipeline "passthrough" \
    videotestsrc num-buffers=1 pattern=smpte ! \
    'video/x-raw,width=640,height=480,format=I420' ! \
    nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! nvmmconvert ! \
    nvvidconv ! 'video/x-raw,format=I420' ! nvjpegenc ! \
    filesink location="$OUT/ci_passthrough.jpg"

run_pipeline "flip-180" \
    videotestsrc num-buffers=1 pattern=smpte ! \
    'video/x-raw,width=640,height=480,format=I420' ! \
    nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
    nvmmconvert flip-method=rotate-180 ! \
    'video/x-raw(memory:NVMM),width=640,height=480' ! \
    nvvidconv ! 'video/x-raw,format=I420' ! nvjpegenc ! \
    filesink location="$OUT/ci_flip180.jpg"

run_pipeline "scale-1080p-480p" \
    videotestsrc num-buffers=1 pattern=smpte ! \
    'video/x-raw,width=1920,height=1080,format=I420' ! \
    nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! nvmmconvert ! \
    'video/x-raw(memory:NVMM),width=640,height=480' ! \
    nvvidconv ! 'video/x-raw,format=I420' ! nvjpegenc ! \
    filesink location="$OUT/ci_scale.jpg"

run_pipeline "crop" \
    videotestsrc num-buffers=1 pattern=smpte ! \
    'video/x-raw,width=1920,height=1080,format=I420' ! \
    nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
    nvmmconvert crop-x=100 crop-y=50 crop-w=800 crop-h=600 ! \
    'video/x-raw(memory:NVMM),width=800,height=600' ! \
    nvvidconv ! 'video/x-raw,format=I420' ! nvjpegenc ! \
    filesink location="$OUT/ci_crop.jpg"

run_pipeline "format-convert-NV12-RGBA" \
    videotestsrc num-buffers=1 pattern=smpte ! \
    'video/x-raw,width=640,height=480,format=I420' ! \
    nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! nvmmconvert ! \
    'video/x-raw(memory:NVMM),format=RGBA,width=640,height=480' ! \
    nvvidconv ! 'video/x-raw,format=RGBA' ! videoconvert ! jpegenc ! \
    filesink location="$OUT/ci_nv12_rgba.jpg"

run_pipeline "decoder-nvmmconvert" \
    videotestsrc num-buffers=5 ! \
    'video/x-raw,width=640,height=480' ! x264enc tune=zerolatency ! \
    'video/x-h264,stream-format=byte-stream' ! \
    nvv4l2decoder ! 'video/x-raw(memory:NVMM)' ! \
    nvmmconvert flip-method=rotate-180 ! \
    'video/x-raw(memory:NVMM),width=640,height=480' ! \
    nvvidconv ! 'video/x-raw,format=I420' ! nvjpegenc ! \
    filesink location="$OUT/ci_decoder.jpg"

run_pipeline "tee-2way" \
    videotestsrc num-buffers=5 pattern=smpte ! \
    'video/x-raw,width=640,height=480,format=I420,framerate=10/1' ! \
    nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
    tee name=t \
    t. ! queue ! nvmmconvert flip-method=rotate-180 ! \
         'video/x-raw(memory:NVMM),width=640,height=480' ! \
         nvvidconv ! 'video/x-raw,format=I420' ! nvjpegenc ! \
         filesink location="$OUT/ci_tee_a.jpg" \
    t. ! queue ! nvmmconvert ! nvvidconv ! 'video/x-raw,format=I420' ! \
         nvjpegenc ! filesink location="$OUT/ci_tee_b.jpg"

run_pipeline "30f-throughput" \
    videotestsrc num-buffers=30 ! \
    'video/x-raw,width=1920,height=1080,format=I420,framerate=30/1' ! \
    nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
    queue max-size-buffers=5 ! \
    nvmmconvert flip-method=rotate-180 ! \
    'video/x-raw(memory:NVMM),width=640,height=480' ! \
    nvvidconv ! 'video/x-raw,format=I420' ! fakesink sync=false

# --- IPC pipeline test (nvmmsink → nvmmappsrc, two processes) ---
echo ""
echo "--- IPC Pipeline Test (two-process nvmmsink → nvmmappsrc) ---"
SHM_NAME="/nvmm_test_e2e_$$"

gst-launch-1.0 -e \
    videotestsrc num-buffers=30 pattern=ball ! \
    'video/x-raw,width=640,height=480,format=I420,framerate=30/1' ! \
    nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
    nvmmsink shm-name="$SHM_NAME" pool-size=8 \
    2>/dev/null &
IPC_PROD_PID=$!

# Wait for SHM to appear (nvmmsink creates it on READY→PAUSED)
for i in $(seq 1 50); do
    ls /dev/shm 2>/dev/null | grep -qF "${SHM_NAME#/}" && break
    sleep 0.1
done

IPC_LOG=$(mktemp)
export GST_DEBUG="nvmmipc.pool:6"
export GST_DEBUG_NO_COLOR=1
timeout 15 gst-launch-1.0 -e \
    nvmmappsrc shm-name="$SHM_NAME" is-live=true ! \
    fakesink sync=false \
    >"$IPC_LOG" 2>&1 || true
unset GST_DEBUG

kill "$IPC_PROD_PID" 2>/dev/null || true
wait "$IPC_PROD_PID" 2>/dev/null || true

FRAMES_RX=$(grep -c "fetched frame" "$IPC_LOG" 2>/dev/null | grep -oE "[0-9]+$" || echo 0)
EOS=$(grep -c "EOS received" "$IPC_LOG" 2>/dev/null | grep -oE "[0-9]+$" || echo 0)
rm -f "$IPC_LOG"

if [ "$FRAMES_RX" -gt 0 ] && [ "$EOS" -gt 0 ]; then
    pass "ipc-pipeline (${FRAMES_RX} frames RX cross-process)"
else
    fail "ipc-pipeline (frames_rx=${FRAMES_RX} eos=${EOS})"
fi

echo ""

# --- Benchmarks ---
echo "--- Benchmarks ---"
"$BUILD/benchmarks/bench_nvmm" 2>/dev/null | grep -E '^(benchmark|alloc|map|transform)'
echo ""

# --- Summary ---
echo "=== Results: $PASS passed, $FAIL failed ==="
exit $FAIL
