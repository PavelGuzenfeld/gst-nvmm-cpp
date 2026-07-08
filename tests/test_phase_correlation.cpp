// Golden test for PhaseCorrelator (gst/common/phase_correlation.hpp) — recovers
// KNOWN circular shifts of a broadband noise image, the standard phase-correlation
// regression check. Self-contained, no OpenCV: the sub-pixel precision was certified
// separately against cv::phaseCorrelate (0.009 px), so here the tolerance guards
// STRUCTURAL regressions (sign flip -> off by 2*shift, missing fft-shift -> off by
// W/2, bad peak -> whole pixels), which are all whole-pixel errors.
//
// exitcode protocol: returns non-zero if any shift is mis-recovered.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "phase_correlation.hpp"

namespace {
constexpr int W = 128, H = 64;   // small pow2 — fast; same code path as any pow2 size

std::vector<float> g_base;
float at(int x, int y) {
    x = ((x % W) + W) % W; y = ((y % H) + H) % H;
    return g_base[(size_t)y * W + x];
}
}  // namespace

int main() {
    g_base.resize((size_t)W * H);
    uint32_t s = 0x9e3779b9u;
    for (auto &v : g_base) { s = s * 1664525u + 1013904223u; v = (float)((s >> 24) & 0xFF); }

    nvmm::PhaseCorrelator pc(W, H);
    const int shifts[][2] = {{0, 0}, {1, 0}, {0, 1}, {3, -2}, {-5, 4}, {7, -6}};
    int fails = 0;
    for (auto &sh : shifts) {
        const int sx = sh[0], sy = sh[1];
        std::vector<float> a((size_t)W * H), b((size_t)W * H);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                a[(size_t)y * W + x] = at(x, y);
                b[(size_t)y * W + x] = at(x - sx, y - sy);   // content moved by (sx,sy)
            }
        const nvmm::PhaseCorrelator::Shift r = pc.correlate(a.data(), b.data());
        const bool ok = std::fabs(r.x - sx) < 0.15 && std::fabs(r.y - sy) < 0.15 &&
                        r.response > 0.80;
        std::printf("  shift(%d,%d) -> (%.3f,%.3f) resp=%.3f  %s\n",
                    sx, sy, r.x, r.y, r.response, ok ? "ok" : "FAIL");
        if (!ok) fails++;
    }
    std::printf(fails ? "FAIL (%d)\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
