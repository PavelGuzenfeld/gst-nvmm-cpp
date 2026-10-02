#!/usr/bin/env bash
# Usage: run-sanitizers.sh [asan|tsan|both], in the dev container; tsan needs --privileged for setarch -R (ASLR vs TSan shadow).
# asan: LD_PRELOAD for the unsanitized plugin scanner, GLib shutdown leaks ignored; pure_cpp is the unpreloaded leak lane.
# tsan skips plugin (the scanner cannot load TSan .so files) and nvidia_hwlib (closed NVIDIA libs double-lock and OOM).
set -eu

MODE="${1:-both}"
TSAN_TIMEOUT_MULTIPLIER=3
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_asan_ubsan() {
    echo "=== ASan + UBSan (detect_leaks=0, GLib-shutdown-noise suites) ==="
    rm -rf builddir-asan
    meson setup builddir-asan \
        -Dcpp_std=c++14 \
        -Dbuildtype=debug \
        -Dwerror=false \
        -Db_sanitize=address,undefined \
        -Db_lundef=false
    meson compile -C builddir-asan

    local libasan
    libasan="$(gcc -print-file-name=libasan.so)"
    [ -z "$libasan" ] && { echo "libasan.so not found"; exit 1; }

    LD_PRELOAD="$libasan" \
    ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:abort_on_error=1:verify_asan_link_order=0" \
    UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1:abort_on_error=1" \
    G_SLICE=always-malloc \
    G_DEBUG=gc-friendly \
    meson test -C builddir-asan --print-errorlogs --no-suite pure_cpp

    ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:abort_on_error=1" \
    UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1:abort_on_error=1" \
    meson test -C builddir-asan --print-errorlogs --suite pure_cpp
}

run_tsan() {
    echo "=== TSan ==="
    rm -rf builddir-tsan
    meson setup builddir-tsan \
        -Dcpp_std=c++14 \
        -Dbuildtype=debug \
        -Dwerror=false \
        -Db_sanitize=thread \
        -Db_lundef=false
    meson compile -C builddir-tsan

    TSAN_OPTIONS="halt_on_error=1:abort_on_error=1:second_deadlock_stack=1:suppressions=$ROOT/scripts/tsan.supp" \
    G_SLICE=always-malloc \
    meson test -C builddir-tsan --print-errorlogs \
        --timeout-multiplier "$TSAN_TIMEOUT_MULTIPLIER" \
        --no-suite plugin \
        --no-suite nvidia_hwlib \
        --wrapper "setarch $(uname -m) -R"
}

case "$MODE" in
    asan)  run_asan_ubsan ;;
    tsan)  run_tsan ;;
    both)  run_asan_ubsan; run_tsan ;;
    *)     echo "unknown mode: $MODE"; exit 2 ;;
esac
