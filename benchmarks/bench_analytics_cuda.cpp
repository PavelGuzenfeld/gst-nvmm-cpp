/// Benchmark: fused host low_texture_motion vs the CUDA implementation
/// (including upload/download — the honest end-to-end cost for a host-resident
/// frame). CSV to stdout, same convention as bench_nvmm. Lives behind
/// -Danalytics_cuda (needs a CUDA device; no OpenCV).
#include <chrono>
#include <cstdio>

#include "analytics_kernels.hpp"
#include "low_texture_motion.hpp"

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::micro>;

namespace {

struct Lcg {
    unsigned s;
    explicit Lcg(unsigned seed) : s(seed) {}
    unsigned next() { s = s * 1664525u + 1013904223u; return s >> 8; }
    int uniform(int lo, int hi) { return lo + (int)(next() % (unsigned)(hi - lo)); }
};

nvmm::img::Image<uint8_t> textured(int w, int h, unsigned seed)
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

template <typename Fn>
void bench(const char *impl, int w, int h, int iters, Fn &&fn)
{
    fn();  // warm-up (first CUDA call also builds the context + buffers)
    double total = 0, mn = 1e12, mx = 0;
    for (int i = 0; i < iters; i++) {
        const auto t0 = Clock::now();
        fn();
        const double us = Duration(Clock::now() - t0).count();
        total += us;
        if (us < mn) mn = us;
        if (us > mx) mx = us;
    }
    printf("low_texture_motion_%s_%dx%d,%d,%.2f,%.2f,%.2f,%.2f\n", impl, w, h, iters,
           total, total / iters, mn, mx);
}

}  // namespace

int main()
{
    printf("benchmark,iterations,total_us,avg_us,min_us,max_us\n");
    const int sizes[][2] = {{640, 360}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    for (const auto &sz : sizes) {
        const int w = sz[0], h = sz[1];
        const int iters = w >= 1920 ? 30 : 60;
        nvmm::img::Image<uint8_t> frame = textured(w, h, 5);
        nvmm::img::Image<uint8_t> ra = textured(w, h, 6);
        nvmm::img::Image<uint8_t> rb = textured(w, h, 7);

        nvmm::motion::LowTextureMotionParams p;
        bench("cpu", w, h, iters,
              [&] { (void)nvmm::motion::low_texture_motion(frame, ra, rb, p); });

        nvmm::motion::LowTextureMotionCuda gpu;
        nvmm::img::Image<float> out;
        bench("cuda", w, h, iters, [&] {
            if (!gpu.run(frame, ra, rb, p, out)) { fprintf(stderr, "cuda failed\n"); }
        });
    }
    return 0;
}
