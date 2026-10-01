#!/usr/bin/env bash
# Usage: run-fuzz.sh [seconds=60]   libFuzzer + ASan/UBSan over tests/fuzz_gmc.cpp; needs clang.
# The meson fuzz_gmc test is only a deterministic smoke sweep; this is the coverage-guided run.
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
