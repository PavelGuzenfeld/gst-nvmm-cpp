#!/usr/bin/env bash
# Build + run the test suite under ASan+UBSan and (separately) under TSan.
# Usage: ./scripts/run-sanitizers.sh            # both
#        ./scripts/run-sanitizers.sh asan       # ASan + UBSan only
#        ./scripts/run-sanitizers.sh tsan       # TSan only
set -eu

MODE="${1:-both}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

run_asan_ubsan() {
    echo "=== ASan + UBSan ==="
    rm -rf builddir-asan
    meson setup builddir-asan \
        -Dcpp_std=c++14 \
        -Dbuildtype=debug \
        -Dwerror=false \
        -Db_sanitize=address,undefined \
        -Db_lundef=false
    meson compile -C builddir-asan

    # When a sanitizer is linked into a .so that's dlopen'd by an
    # unsanitized host (GStreamer plugin scanner), ASan aborts with
    # "runtime does not come first in initial library list". LD_PRELOAD
    # the runtime into every test to get around it.
    local libasan
    libasan="$(gcc -print-file-name=libasan.so)"
    [ -z "$libasan" ] && { echo "libasan.so not found"; exit 1; }

    # LeakSanitizer flags GLib / GStreamer shutdown leaks we don't own;
    # detect_leaks=0 keeps the signal-to-noise high. UBSan halts on any
    # finding so our own UB surfaces loudly.
    LD_PRELOAD="$libasan" \
    ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:abort_on_error=1:verify_asan_link_order=0" \
    UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1:abort_on_error=1" \
    G_SLICE=always-malloc \
    G_DEBUG=gc-friendly \
    meson test -C builddir-asan --print-errorlogs
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

    # TSan does NOT need LD_PRELOAD like ASan does — the runtime is
    # linked into each test binary and can handle dlopen'd .so files
    # that were also built with -fsanitize=thread.
    #
    # Modern kernels randomize the mmap base into regions TSan reserves
    # as its shadow space, giving "unexpected memory mapping" aborts on
    # startup. `setarch -R` disables ASLR for the wrapped process.
    # Skip the plugin-factory suite: those tests dlopen our TSan-instrumented
    # .so files, which GStreamer's plugin scanner rejects (unsanitized scanner
    # can't load sanitized libs, and TSan can't be LD_PRELOADed the way ASan
    # can). The backend_concurrency test covers the same atomic-heavy paths
    # without going through the factory.
    TSAN_OPTIONS="halt_on_error=1:abort_on_error=1:second_deadlock_stack=1:suppressions=/src/scripts/tsan.supp" \
    G_SLICE=always-malloc \
    meson test -C builddir-tsan --print-errorlogs \
        --no-suite plugin-factory \
        --wrapper "setarch $(uname -m) -R"
}

case "$MODE" in
    asan)  run_asan_ubsan ;;
    tsan)  run_tsan ;;
    both)  run_asan_ubsan; run_tsan ;;
    *)     echo "unknown mode: $MODE"; exit 2 ;;
esac
