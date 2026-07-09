#!/usr/bin/env bash
# Coverage-guided fuzzing of the GMC estimator + target-masking core
# (tests/fuzz_gmc.cpp) with clang's libFuzzer, under ASan+UBSan.
#
# The standalone build of tests/fuzz_gmc.cpp (meson test `fuzz_gmc`, run under
# scripts/run-sanitizers.sh) is a deterministic smoke sweep — it always passes once
# the code is correct and never gets deeper. THIS script is the actual fuzzer:
# coverage-guided, runs until the time budget expires or it finds a crash.
#
# Usage: ./scripts/run-fuzz.sh [seconds]     # default 60s
#
# Run inside the dev container (docker/Dockerfile.dev) or any host with clang +
# libFuzzer (clang ships libFuzzer since ~6.0). Requires clang, not gcc.
set -eu

SECONDS_BUDGET="${1:-60}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v clang++ >/dev/null 2>&1 || { echo "clang++ not found (libFuzzer needs clang)"; exit 1; }

OUT=/tmp/gst-nvmm-fuzz-gmc
CORPUS="$OUT/corpus"
mkdir -p "$CORPUS"

echo "=== building fuzz_gmc (libFuzzer + ASan + UBSan) ==="
clang++ -std=c++14 -O1 -g -DGMC_LIBFUZZER \
    -fsanitize=fuzzer,address,undefined \
    -Igst/common -Igst/nvmmsamurai \
    tests/fuzz_gmc.cpp -o "$OUT/fuzz_gmc"

echo "=== fuzzing for ${SECONDS_BUDGET}s (corpus: $CORPUS) ==="
UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1:abort_on_error=1" \
"$OUT/fuzz_gmc" -max_total_time="$SECONDS_BUDGET" "$CORPUS"

echo "=== OK: no crash in ${SECONDS_BUDGET}s; corpus has $(ls "$CORPUS" | wc -l) inputs ==="
