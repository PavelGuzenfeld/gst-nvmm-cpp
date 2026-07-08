// Unit test for the GMC backend layer:
//   1) resolve_gmc_backend() auto-order + graceful degradation (gmc_backend.hpp),
//   2) gmc_patch_size() policy,
//   3) SIGN-AGREEMENT parity between the NCC baseline (samurai_gmc.hpp) and the
//      FFT phase-correlation backend (phase_correlation.hpp) on the SAME known
//      shift. This is the critical guard: the two estimators have different native
//      conventions, and a flipped sign feeding kf.shift() would *double* camera
//      motion instead of cancelling it. Both must return the shift with the same
//      sign and magnitude as the injected (sx,sy).
//
// exitcode protocol: non-zero on any failure. Self-contained, no OpenCV/CUDA/GST.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "gmc_backend.hpp"
#include "phase_correlation.hpp"
#include "samurai_gmc.hpp"

using nvmm::GmcBackend;

namespace {
int g_fails = 0;
void check(bool ok, const char *what) {
    std::printf("  %-52s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) g_fails++;
}

constexpr int N = 128;  // pow2 so the same patch feeds NCC and the FFT correlator

std::vector<uint8_t> g_base;
uint8_t base_at(int x, int y) {
    x = ((x % N) + N) % N; y = ((y % N) + N) % N;  // circular
    return g_base[(size_t)y * N + x];
}
}  // namespace

int main() {
    // ---- 1) resolve_gmc_backend: auto order fft-cuda > pva > fft-cpu ----
    check(resolve_gmc_backend(GmcBackend::Auto, true, true) == GmcBackend::FftCuda,
          "auto -> fft-cuda when cuda available");
    check(resolve_gmc_backend(GmcBackend::Auto, false, true) == GmcBackend::Pva,
          "auto -> pva when only pva available");
    check(resolve_gmc_backend(GmcBackend::Auto, false, false) == GmcBackend::FftCpu,
          "auto -> fft-cpu when no accelerator");
    // explicit requests degrade DOWN, never silently up
    check(resolve_gmc_backend(GmcBackend::FftCuda, false, true) == GmcBackend::Pva,
          "fft-cuda unavailable -> pva");
    check(resolve_gmc_backend(GmcBackend::FftCuda, false, false) == GmcBackend::FftCpu,
          "fft-cuda unavailable, no pva -> fft-cpu");
    check(resolve_gmc_backend(GmcBackend::Pva, false, false) == GmcBackend::FftCpu,
          "pva unavailable -> fft-cpu");
    check(resolve_gmc_backend(GmcBackend::Ncc, true, true) == GmcBackend::Ncc,
          "explicit ncc always honored");
    check(resolve_gmc_backend(GmcBackend::FftCpu, true, true) == GmcBackend::FftCpu,
          "explicit fft-cpu always honored");

    // ---- 2) patch-size policy: PVA needs >=160, others 128 ----
    check(nvmm::gmc_patch_size(GmcBackend::Ncc) == 128, "ncc patch 128");
    check(nvmm::gmc_patch_size(GmcBackend::FftCpu) == 128, "fft-cpu patch 128");
    check(nvmm::gmc_patch_size(GmcBackend::Pva) == 256, "pva patch 256 (>=160)");

    // ---- 3) sign-agreement parity: NCC vs FFT on the same injected shift ----
    g_base.resize((size_t)N * N);
    uint32_t s = 0x9e3779b9u;
    for (auto &v : g_base) { s = s * 1664525u + 1013904223u; v = (uint8_t)((s >> 24) & 0xFF); }

    nvmm::PhaseCorrelator pc(N, N);
    const int shifts[][2] = {{0, 0}, {2, 0}, {0, 3}, {3, -2}, {-4, 5}, {6, -3}};
    for (auto &sh : shifts) {
        const int sx = sh[0], sy = sh[1];
        std::vector<uint8_t> prev((size_t)N * N), curr((size_t)N * N);
        std::vector<float>   prevf((size_t)N * N), currf((size_t)N * N);
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                const uint8_t p = base_at(x, y);
                const uint8_t c = base_at(x - sx, y - sy);  // content moved by (sx,sy)
                const size_t i = (size_t)y * N + x;
                prev[i] = p; curr[i] = c;
                prevf[i] = (float)p; currf[i] = (float)c;
            }
        const nvmm::GmcShift ncc = nvmm::estimate_shift(prev.data(), curr.data(), N, 24);
        const nvmm::PhaseCorrelator::Shift fft = pc.correlate(prevf.data(), currf.data());

        char buf[96];
        // NCC recovers the exact integer shift with the injected sign.
        std::snprintf(buf, sizeof buf, "ncc recovers (%d,%d)", sx, sy);
        check(std::fabs(ncc.dx - sx) < 0.5f && std::fabs(ncc.dy - sy) < 0.5f, buf);
        // FFT recovers the same shift, same sign, sub-pixel.
        std::snprintf(buf, sizeof buf, "fft recovers (%d,%d)", sx, sy);
        check(std::fabs(fft.x - sx) < 0.15 && std::fabs(fft.y - sy) < 0.15, buf);
        // The two AGREE in sign+value (the real guard against a doubled-motion bug).
        std::snprintf(buf, sizeof buf, "ncc/fft agree (%d,%d)", sx, sy);
        check(std::fabs(ncc.dx - fft.x) < 0.5 && std::fabs(ncc.dy - fft.y) < 0.5, buf);
    }

    std::printf(g_fails ? "FAIL (%d)\n" : "OK\n", g_fails);
    return g_fails ? 1 : 0;
}
