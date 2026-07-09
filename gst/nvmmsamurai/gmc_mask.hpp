/// gmc_mask.hpp — target-aware GMC masking (host, dependency-free, unit-testable).
///
/// The tracked target's own motion inside the GMC center patch can contaminate the
/// camera-motion estimate (the FFT backends report a spurious shift for a moving
/// target on an otherwise-static camera — see samurai_tracker.cpp's apply_gmc).
/// The fix: map the tracked box into the patch's coordinate space and fill it with
/// the patch mean in BOTH frames before estimating, so its edges correlate at zero
/// shift (reinforcing the background estimate) instead of adding a spurious peak.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace nvmm {

struct GmcMaskBox {
    int  x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    bool overlaps = false;   // false: box is non-finite, degenerate, or off-patch
};

/// Map a tracked box (frame pixel coords: left, top, width, height) into the
/// coordinate space of an n x n GMC patch, where the patch is a bilinear
/// downscale of a `sq x sq` square crop centered in a `frame_w x frame_h` frame.
/// `margin` inflates the mapped box (1.25 = +25%) to absorb sub-pixel/tracking
/// jitter at the target's edge. Returns `overlaps=false` — the caller's cue to
/// skip masking — when the box is non-finite (e.g. a diverged Kalman state),
/// degenerate (w/h <= 0), or maps entirely outside the patch; a non-finite input
/// would otherwise reach a double->int cast, which is undefined behavior.
inline GmcMaskBox gmc_map_box_to_patch(double left, double top, double width, double height,
                                       int frame_w, int frame_h, int sq, int patch_n,
                                       double margin = 1.25)
{
    GmcMaskBox out;
    const double sum = left + top + width + height;
    if (!std::isfinite(sum) || width <= 0.0 || height <= 0.0 || sq <= 0 || patch_n <= 0)
        return out;
    const double s2p = (double)patch_n / sq;   // patch px per frame px
    const double cx = (left + width  * 0.5 - (frame_w - sq) / 2.0) * s2p;
    const double cy = (top  + height * 0.5 - (frame_h - sq) / 2.0) * s2p;
    const double hw = width  * 0.5 * margin * s2p;
    const double hh = height * 0.5 * margin * s2p;
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(hw) || !std::isfinite(hh))
        return out;
    out.x0 = (int)(cx - hw); out.y0 = (int)(cy - hh);
    out.x1 = (int)(cx + hw); out.y1 = (int)(cy + hh);
    out.overlaps = out.x1 > 0 && out.y1 > 0 && out.x0 < patch_n && out.y0 < patch_n;
    return out;
}

/// Fill the (clamped) box [x0,y0)-(x1,y1) in an n x n patch with the patch mean.
/// No-op if the clamped box is empty.
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

}  // namespace nvmm
