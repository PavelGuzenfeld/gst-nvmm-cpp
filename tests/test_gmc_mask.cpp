#include <cstdio>
#include <cstdint>
#include <cmath>
#include <limits>
#include <vector>

#include "gmc_mask.hpp"

using nvmm::GmcMaskBox;
using nvmm::gmc_map_box_to_patch;
using nvmm::gmc_mask_box_to_mean;

namespace {
int g_fails = 0;
void check(bool ok, const char *what) {
    std::printf("  %-56s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) g_fails++;
}
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
}

int main() {
    check(!gmc_map_box_to_patch(kNaN, 0, 10, 10, 1920, 1080, 512, 128).overlaps,
          "NaN left -> no overlap");
    check(!gmc_map_box_to_patch(0, kInf, 10, 10, 1920, 1080, 512, 128).overlaps,
          "Inf top -> no overlap");
    check(!gmc_map_box_to_patch(0, 0, kNaN, 10, 1920, 1080, 512, 128).overlaps,
          "NaN width -> no overlap");
    check(!gmc_map_box_to_patch(0, 0, 10, kInf, 1920, 1080, 512, 128).overlaps,
          "Inf height -> no overlap");
    check(!gmc_map_box_to_patch(0, 0, 0, 10, 1920, 1080, 512, 128).overlaps,
          "zero width -> no overlap (degenerate)");
    check(!gmc_map_box_to_patch(0, 0, 10, -5, 1920, 1080, 512, 128).overlaps,
          "negative height -> no overlap (degenerate)");
    check(!gmc_map_box_to_patch(0, 0, 10, 10, 1920, 1080, 0, 128).overlaps,
          "sq=0 -> no overlap (degenerate)");

    {
        const GmcMaskBox mb = gmc_map_box_to_patch(960 - 5, 540 - 5, 10, 10, 1920, 1080, 1080, 128, 1.0);
        check(mb.overlaps, "centered box overlaps the patch");
        const int cx = (mb.x0 + mb.x1) / 2, cy = (mb.y0 + mb.y1) / 2;
        check(std::abs(cx - 64) <= 1 && std::abs(cy - 64) <= 1,
              "centered box maps near patch center (64,64)");
    }

    check(!gmc_map_box_to_patch(-10000, -10000, 10, 10, 1920, 1080, 1080, 128).overlaps,
          "far off-frame box -> no overlap");

    {
        constexpr int N = 16;
        std::vector<uint8_t> patch(N * N);
        for (int i = 0; i < N * N; i++) patch[i] = (uint8_t)((i * 37) % 256);
        uint64_t sum = 0;
        for (int i = 0; i < N * N; i++) sum += patch[i];
        const uint8_t mean = (uint8_t)(sum / (N * N));

        std::vector<uint8_t> before = patch;
        gmc_mask_box_to_mean(patch.data(), N, 4, 4, 10, 10);

        bool region_ok = true, outside_ok = true;
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                const bool inside = x >= 4 && x < 10 && y >= 4 && y < 10;
                if (inside && patch[y * N + x] != mean) region_ok = false;
                if (!inside && patch[y * N + x] != before[y * N + x]) outside_ok = false;
            }
        check(region_ok, "masked region filled with patch mean");
        check(outside_ok, "outside the masked region is untouched");

        std::vector<uint8_t> unchanged = patch;
        gmc_mask_box_to_mean(patch.data(), N, 10, 10, 4, 4);
        check(patch == unchanged, "inverted box (x0>=x1) is a no-op");
        gmc_mask_box_to_mean(patch.data(), N, 1000, 1000, 2000, 2000);
        check(patch == unchanged, "fully out-of-range box clamps to no-op");
    }

    std::printf(g_fails ? "FAIL (%d)\n" : "OK\n", g_fails);
    return g_fails ? 1 : 0;
}
