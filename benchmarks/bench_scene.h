/// Synthetic frame generator shared by the analytics benchmarks
/// (bench_analytics.cpp, bench_analytics_cuda.cpp) — deterministic, no
/// dependencies, so it's equally usable from the OpenCV-backed golden-lane
/// benchmark and the OpenCV-free CUDA-lane benchmark.
#pragma once
#include <cstdint>

#include "image.hpp"

namespace bench_scene {

struct Lcg {
    unsigned s;
    explicit Lcg(unsigned seed) : s(seed) {}
    unsigned next() { s = s * 1664525u + 1013904223u; return s >> 8; }
    int uniform(int lo, int hi) { return lo + (int)(next() % (unsigned)(hi - lo)); }
};

inline nvmm::img::Image<uint8_t> textured(int w, int h, unsigned seed)
{
    Lcg rng(seed);
    nvmm::img::Image<uint8_t> f(w, h, 100);
    for (int i = 0; i < w * h / 400; i++) {
        const int cx = rng.uniform(6, w - 6), cy = rng.uniform(6, h - 6);
        const int r = rng.uniform(2, 5);
        const uint8_t v = (uint8_t)rng.uniform(60, 240);
        for (int y = cy - r; y <= cy + r; y++)
            for (int x = cx - r; x <= cx + r; x++)
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) f.at(y, x) = v;
    }
    return f;
}

}  // namespace bench
