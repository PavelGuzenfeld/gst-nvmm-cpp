/// Active-content bounds of a frame — excludes uniform letterbox / pillarbox bars.
///
/// Many capture/transcode pipelines pad non-16:9 sources with black (or constant
/// gray) bars. Anything keyed off the frame extent (ROI gating, edge-of-frame
/// rejection, normalisation) wants the *content* rectangle, not the padded one.
///
/// Detection is by per-row / per-column intensity RANGE (max−min), not mean: a bar
/// is a band that spans almost no gray levels. Using the range (not a mean/brightness
/// threshold) keeps it correct on a DARK-but-textured frame — e.g. a thermal/IR sky,
/// which is dim everywhere yet has real gradient, so a mean threshold would wrongly
/// classify it as a bar and crop the content.
///
/// All four row/column min/max profiles come out of ONE sweep over the frame
/// (the OpenCV version needed four full cv::reduce passes).
///
/// Pure C++14, header-only, no dependencies.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "image.hpp"

namespace nvmm {
namespace video {

struct ActiveRegionParams {
    int bar_range = 15;  // a row/col spanning fewer than this many gray levels is a uniform bar
};

/// Content rectangle of `gray`, trimming uniform border bars. Returns the full
/// frame when nothing looks like a bar (or the frame is entirely uniform), and
/// an empty rect for an empty view.
inline img::Rect active_region(img::View<const uint8_t> gray, const ActiveRegionParams &p = {})
{
    if (gray.empty()) return img::Rect();
    const int w = gray.width, h = gray.height;
    std::vector<uint8_t> cmin((size_t)w, 255), cmax((size_t)w, 0);
    std::vector<int> rrange((size_t)h);
    for (int y = 0; y < h; y++) {
        const uint8_t *row = gray.row(y);
        uint8_t lo = 255, hi = 0;
        for (int x = 0; x < w; x++) {
            const uint8_t v = row[x];
            lo = std::min(lo, v);
            hi = std::max(hi, v);
            cmin[(size_t)x] = std::min(cmin[(size_t)x], v);
            cmax[(size_t)x] = std::max(cmax[(size_t)x], v);
        }
        rrange[(size_t)y] = (int)hi - (int)lo;
    }
    auto crange = [&](int i) { return (int)cmax[(size_t)i] - (int)cmin[(size_t)i]; };
    int x0 = 0;      while (x0 < w - 1 && crange(x0) < p.bar_range) x0++;
    int x1 = w - 1;  while (x1 > x0    && crange(x1) < p.bar_range) x1--;
    int y0 = 0;      while (y0 < h - 1 && rrange[(size_t)y0] < p.bar_range) y0++;
    int y1 = h - 1;  while (y1 > y0    && rrange[(size_t)y1] < p.bar_range) y1--;
    // Entirely-uniform frame (every row/col is a "bar") -> no content to trim, return full.
    if (x1 <= x0 || y1 <= y0) return img::Rect{0, 0, w, h};
    return img::Rect{x0, y0, x1 - x0 + 1, y1 - y0 + 1};
}

}  // namespace video
}  // namespace nvmm
