// Fuzz + sanitizer harness for the GMC estimator core (host, no CUDA/VPI):
//   nvmm::PhaseCorrelator::correlate, nvmm::estimate_shift, refine_correlation_peak,
//   nvmm::gmc_map_box_to_patch / nvmm::gmc_mask_box_to_mean (target-aware masking).
// Feeds arbitrary bytes as two grayscale patches (and, separately, reinterpreted as
// raw box doubles/ints, so NaN/Inf/subnormal/huge values arise organically rather
// than only via hand-picked cases) to surface OOB / UB / non-finite output.
//
// Two entry points from one core (`fuzz_gmc_once`):
//   - libFuzzer (compile with -DGMC_LIBFUZZER): coverage-guided. Build with clang:
//       clang++ -std=c++14 -O1 -g -DGMC_LIBFUZZER -fsanitize=fuzzer,address,undefined
//       -I gst/common -I gst/nvmmsamurai tests/fuzz_gmc.cpp -o fuzz_gmc  (then ./fuzz_gmc)
//   - standalone main(): deterministic pseudo-random sweep, buildable with g++ and
//     registered as a meson test, so it runs under scripts/run-sanitizers.sh (ASan+UBSan)
//     — that is the leak/UB coverage for this code path in CI (no clang needed).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gmc_mask.hpp"
#include "phase_correlation.hpp"
#include "samurai_gmc.hpp"

namespace {
// One fuzz iteration: derive a power-of-two patch size + two patches from `data`, run
// every estimator, and require finite outputs. Returns false on a non-finite result.
bool fuzz_gmc_once(const uint8_t *data, size_t size) {
    if (size < 4) return true;
    const int sizes[] = {16, 32, 64};   // pow2 (PhaseCorrelator requires it)
    const int n = sizes[data[0] % 3];
    const size_t n2 = (size_t)n * n;
    std::vector<uint8_t> a(n2), b(n2);
    std::vector<float> af(n2), bf(n2);
    for (size_t i = 0; i < n2; i++) {
        a[i] = data[(1 + i) % size];
        b[i] = data[(1 + i + (size_t)n) % size];
        af[i] = (float)a[i];
        bf[i] = (float)b[i];
    }
    nvmm::PhaseCorrelator pc(n, n);
    const nvmm::PhaseCorrelator::Shift s = pc.correlate(af.data(), bf.data());
    const nvmm::GmcShift g = nvmm::estimate_shift(a.data(), b.data(), n, n / 4);
    return std::isfinite(s.x) && std::isfinite(s.y) && std::isfinite(s.response) &&
           std::isfinite(g.dx) && std::isfinite(g.dy) && std::isfinite(g.conf);
}

// Fuzz gmc_map_box_to_patch + gmc_mask_box_to_mean: reinterpret raw bytes as doubles
// and int32s (organically producing NaN/Inf/subnormal/huge/negative values, the
// pathological cases a real diverged-Kalman-state or adversarial input could hit)
// and feed them through the box mapper and the masker. Success = no crash / no OOB
// write (ASan/UBSan catch those); the mapper's own isfinite guard is what's under
// test, so no output is asserted here.
bool fuzz_gmc_mask_once(const uint8_t *data, size_t size) {
    if (size < 48) return true;
    double box[4];
    std::memcpy(box, data, sizeof(box));
    const nvmm::GmcMaskBox mb = nvmm::gmc_map_box_to_patch(box[0], box[1], box[2], box[3],
                                                           1920, 1080, 512, 128);
    constexpr int N = 32;
    uint8_t patch[N * N];
    for (int i = 0; i < N * N; i++) patch[i] = (uint8_t)i;
    nvmm::gmc_mask_box_to_mean(patch, N, mb.x0, mb.y0, mb.x1, mb.y1);

    // Independently: raw fuzzed ints straight into the masker, bypassing the mapper's
    // own clamping — proves gmc_mask_box_to_mean's clamp alone is OOB-safe for any
    // (possibly huge/negative/inverted) box, not just ones the mapper could produce.
    int32_t ivals[4];
    std::memcpy(ivals, data + 32, sizeof(ivals));
    nvmm::gmc_mask_box_to_mean(patch, N, ivals[0], ivals[1], ivals[2], ivals[3]);
    return true;
}
}  // namespace

#ifdef GMC_LIBFUZZER
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    fuzz_gmc_once(data, size);        // libFuzzer catches crashes/ASan/UBSan; results unused
    fuzz_gmc_mask_once(data, size);
    return 0;
}
#else
#include <cstdio>
int main() {
    // Deterministic LCG sweep — no external corpus; deterministic so a failure repros.
    // 3000 iters keeps the ASan+UBSan run (scripts/run-sanitizers.sh) well under its
    // timeout; deep coverage-guided fuzzing is the libFuzzer variant (see header).
    uint32_t s = 0x12345678u;
    std::vector<uint8_t> buf;
    for (int iter = 0; iter < 3000; iter++) {
        s = s * 1664525u + 1013904223u;
        const size_t sz = 4 + (s % 4096);
        buf.resize(sz);
        for (size_t i = 0; i < sz; i++) { s = s * 1664525u + 1013904223u; buf[i] = (uint8_t)(s >> 24); }
        if (!fuzz_gmc_once(buf.data(), sz)) {
            std::printf("FAIL: non-finite estimator output at iter %d (seed-derived)\n", iter);
            return 1;
        }
        if (!fuzz_gmc_mask_once(buf.data(), sz)) {
            std::printf("FAIL: gmc_mask crash/OOB at iter %d (seed-derived)\n", iter);
            return 1;
        }
    }
    std::printf("OK (3000 iters, estimator + mask)\n");
    return 0;
}
#endif
