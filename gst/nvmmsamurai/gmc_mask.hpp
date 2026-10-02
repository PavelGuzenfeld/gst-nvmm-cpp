#pragma once
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <limits>

namespace nvmm {

struct GmcMaskBox {
    int  x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    bool overlaps = false;
};

/// Box in frame px; the patch is an n x n downscale of a sq x sq crop centered in
/// the frame, and margin 1.25 inflates the box 25%. A box that is non-finite or past
/// int range returns overlaps=false: it would otherwise reach a double->int cast, which is UB.
inline GmcMaskBox gmc_map_box_to_patch(double left, double top, double width, double height,
                                       int frame_w, int frame_h, int sq, int patch_n,
                                       double margin = 1.25)
{
    GmcMaskBox out;
    if (width <= 0.0 || height <= 0.0 || sq <= 0 || patch_n <= 0)
        return out;
    const double s2p = (double)patch_n / sq;
    const double cx = (left + width  * 0.5 - (frame_w - sq) / 2.0) * s2p;
    const double cy = (top  + height * 0.5 - (frame_h - sq) / 2.0) * s2p;
    const double hw = width  * 0.5 * margin * s2p;
    const double hh = height * 0.5 * margin * s2p;
    const double int_max = (double)std::numeric_limits<int>::max();
    if (!(std::fabs(cx) + std::fabs(hw) <= int_max) || !(std::fabs(cy) + std::fabs(hh) <= int_max))
        return out;
    out.x0 = (int)(cx - hw); out.y0 = (int)(cy - hh);
    out.x1 = (int)(cx + hw); out.y1 = (int)(cy + hh);
    out.overlaps = out.x1 > 0 && out.y1 > 0 && out.x0 < patch_n && out.y0 < patch_n;
    return out;
}

/// Filling the target with the patch mean in both frames makes its edges correlate
/// at zero shift instead of adding a spurious peak.
inline void gmc_mask_box_to_mean(uint8_t *patch, int n, int x0, int y0, int x1, int y1)
{
    x0 = x0 < 0 ? 0 : x0; y0 = y0 < 0 ? 0 : y0;
    x1 = x1 > n ? n : x1; y1 = y1 > n ? n : y1;
    if (x0 >= x1 || y0 >= y1) return;
    uint64_t sum = 0;
    for (int i = 0; i < n * n; i++) sum += patch[i];
    const uint8_t mean = (uint8_t)(sum / ((uint64_t)n * n));
    for (int y = y0; y < y1; y++)
        std::memset(patch + (size_t)y * n + x0, mean, (size_t)(x1 - x0));
}

}
