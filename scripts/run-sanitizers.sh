#!/usr/bin/env bash
# Build + run the test suite under ASan+UBSan and (separately) under TSan.
# Usage: ./scripts/run-sanitizers.sh            # both
#        ./scripts/run-sanitizers.sh asan       # ASan + UBSan only
#        ./scripts/run-sanitizers.sh tsan       # TSan only
#
# Run inside the dev container (docker/Dockerfile.dev), which builds the mock
# NvBufSurface API so the suite runs host-side.
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

    # When a sanitizer is linked into a .so that's dlopen'd by an unsanitized
    # host (GStreamer's plugin scanner), ASan aborts with "runtime does not
    # come first in initial library list". LD_PRELOAD the runtime into every
    # test to get around it.
    local libasan
    libasan="$(gcc -print-file-name=libasan.so)"
    [ -z "$libasan" ] && { echo "libasan.so not found"; exit 1; }

    # LeakSanitizer flags GLib / GStreamer shutdown leaks we don't own;
    # detect_leaks=0 keeps the signal high there. UBSan halts on any finding so
    # our own UB surfaces loudly.
    LD_PRELOAD="$libasan" \
    ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:abort_on_error=1:verify_asan_link_order=0" \
    UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1:abort_on_error=1" \
    G_SLICE=always-malloc \
    G_DEBUG=gc-friendly \
    meson test -C builddir-asan --print-errorlogs --no-suite pure_cpp

    # pure_cpp suite (header-only, no GLib/GStreamer/CUDA — see tests/meson.build):
    # plain executables, so ASan is already linked in via -Db_sanitize (no dlopen'd
    # .so, so no LD_PRELOAD workaround needed — and none wanted: LD_PRELOAD-ing
    # libasan into the `meson test` env var also instruments meson's OWN ninja/
    # python child processes, which then report THEIR allocations as "leaked".
    # This is the only lane in the repo that enforces real leak detection — keep
    # it separate rather than weakening the detect_leaks=0 pass above, which stays
    # permissive for GLib's own leaks.
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

    # Modern kernels randomize the mmap base into regions TSan reserves as its
    # shadow space, giving "unexpected memory mapping" aborts on startup.
    # `setarch -R` disables ASLR for the wrapped process — which needs the
    # capability to set the process personality. In a container run with
    # `--privileged` (or `--security-opt seccomp=unconfined --cap-add SYS_ADMIN`)
    # or on a bare host; an unprivileged container fails with
    # "setarch: failed to set personality: Operation not permitted".
    #
    # Skip the `plugin` suite: those tests dlopen our TSan-instrumented .so
    # files, which GStreamer's plugin scanner rejects (an unsanitized scanner
    # can't load sanitized libs, and TSan can't be LD_PRELOADed the way ASan
    # can). The remaining core + concurrency tests cover the atomic-heavy IPC
    # paths without going through the plugin factory.
    #
    # Skip `nvidia_hwlib` too: on a real-API (Jetson) build, nvmm_buffer and
    # nvmm_transform delegate to closed NVIDIA libs that TSan flags but we can't
    # fix (libnvbufsurftransform double-locks its own global mutex; the CUDA
    # allocator OOMs under TSan's shadow space). That suite is empty on the mock
    # build, so this flag is a harmless no-op there and those tests still run.
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
