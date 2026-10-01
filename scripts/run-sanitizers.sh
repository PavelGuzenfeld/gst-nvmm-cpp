#!/usr/bin/env bash
# Usage: run-sanitizers.sh [asan|tsan|both]   in the dev container, whose mock NvBufSurface
# API lets the suite run host-side.
set -eu

MODE="${1:-both}"
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

    # LD_PRELOAD: an ASan .so dlopen'd by the unsanitized plugin scanner aborts with
    # "runtime does not come first in initial library list".
    local libasan
    libasan="$(gcc -print-file-name=libasan.so)"
    [ -z "$libasan" ] && { echo "libasan.so not found"; exit 1; }

    # detect_leaks=0: LeakSanitizer flags GLib/GStreamer shutdown leaks we do not own.
    LD_PRELOAD="$libasan" \
    ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:abort_on_error=1:verify_asan_link_order=0" \
    UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1:abort_on_error=1" \
    G_SLICE=always-malloc \
    G_DEBUG=gc-friendly \
    meson test -C builddir-asan --print-errorlogs --no-suite pure_cpp

    # pure_cpp is the repo's only real leak-detection lane. No LD_PRELOAD: it would also
    # instrument meson's own ninja/python children, which then report their allocations as leaks.
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

    # setarch -R stops ASLR mapping into TSan's shadow (needs --privileged in a container).
    # plugin: the unsanitized scanner cannot load TSan .so files. nvidia_hwlib: closed NVIDIA
    # libs double-lock and OOM under TSan; that suite is empty on the mock build.
    TSAN_OPTIONS="halt_on_error=1:abort_on_error=1:second_deadlock_stack=1:suppressions=$ROOT/scripts/tsan.supp" \
    G_SLICE=always-malloc \
    meson test -C builddir-tsan --print-errorlogs \
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
