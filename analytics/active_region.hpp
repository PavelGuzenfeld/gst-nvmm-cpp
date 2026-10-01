#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "image.hpp"

namespace nvmm {
namespace video {

struct ActiveRegionParams {
    int bar_range = 15;
};

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
    if (x1 <= x0 || y1 <= y0) return img::Rect{0, 0, w, h};
    return img::Rect{x0, y0, x1 - x0 + 1, y1 - y0 + 1};
}

}
}
