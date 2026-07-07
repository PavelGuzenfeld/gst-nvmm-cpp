/// Synthetic-scene builders for the analytics tests — the OpenCV-free
/// replacements for the cv::RNG / cv::circle / cv::warpAffine plumbing the
/// original tests used. Deterministic (fixed-seed LCG); numeric sequences
/// differ from cv::RNG, so tests assert on behavioural ranges, not exact
/// pixel values (as they already did).
#pragma once
#include <cmath>
#include <cstdint>

#include "image.hpp"
#include "image_ops.hpp"

namespace scene {

class Rng {
public:
    explicit Rng(unsigned seed) : s_(seed ? seed : 1u) {}
    unsigned next() { s_ = s_ * 1664525u + 1013904223u; return s_ >> 8; }
    int uniform(int lo, int hi) { return lo + (int)(next() % (unsigned)(hi - lo)); }
    float gauss(float sigma) {
        const float u1 = (next() % 100000 + 1) / 100001.f;
        const float u2 = (next() % 100000) / 100000.f;
        return sigma * std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2);
    }

private:
    unsigned s_;
};

inline uint8_t clamp_u8(float v)
{
    return (uint8_t)(v < 0.f ? 0.f : v > 255.f ? 255.f : v + 0.5f);
}

inline void fill_circle(nvmm::img::Image<uint8_t> &im, int cx, int cy, int r, uint8_t val)
{
    for (int y = cy - r; y <= cy + r; y++) {
        if (y < 0 || y >= im.height()) continue;
        for (int x = cx - r; x <= cx + r; x++) {
            if (x < 0 || x >= im.width()) continue;
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) im.at(y, x) = val;
        }
    }
}

inline void gaussian_blur_u8(nvmm::img::Image<uint8_t> &im, int k)
{
    nvmm::img::Image<float> f(im.width(), im.height()), tmp;
    for (int y = 0; y < im.height(); y++)
        for (int x = 0; x < im.width(); x++) f.at(y, x) = (float)im.at(y, x);
    nvmm::img::gaussian_blur<float>(f.view(), tmp, f.view(), k);
    for (int y = 0; y < im.height(); y++)
        for (int x = 0; x < im.width(); x++) im.at(y, x) = clamp_u8(f.at(y, x));
}

/// BORDER_REFLECT index fold (fedcba|abcdefgh|hgfedcb) — what the original
/// tests' cv::warpAffine(..., BORDER_REFLECT) used.
inline int reflect(int i, int n)
{
    while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - 1 - i;
    return i;
}

/// Translate by (dx, dy) with bilinear sampling and reflected border —
/// replaces the tests' cv::warpAffine translation.
inline nvmm::img::Image<uint8_t> translate(const nvmm::img::Image<uint8_t> &src,
                                           double dx, double dy)
{
    const int w = src.width(), h = src.height();
    nvmm::img::Image<uint8_t> out(w, h);
    for (int y = 0; y < h; y++) {
        const double sy = y - dy;
        const int y0 = (int)std::floor(sy);
        const float fy = (float)(sy - y0);
        const uint8_t *r0 = src.row(reflect(y0, h)), *r1 = src.row(reflect(y0 + 1, h));
        for (int x = 0; x < w; x++) {
            const double sx = x - dx;
            const int x0 = (int)std::floor(sx);
            const float fx = (float)(sx - x0);
            const int xa = reflect(x0, w), xb = reflect(x0 + 1, w);
            const float v = (1.f - fy) * ((1.f - fx) * r0[xa] + fx * r0[xb]) +
                            fy * ((1.f - fx) * r1[xa] + fx * r1[xb]);
            out.at(y, x) = clamp_u8(v);
        }
    }
    return out;
}

/// Textured but band-limited background: soft bright blobs on gray + light blur
/// (the dual_homography / gate tests' make_bg()).
inline nvmm::img::Image<uint8_t> textured_bg(int size, unsigned seed)
{
    nvmm::img::Image<uint8_t> bg(size, size, 100);
    Rng rng(seed);
    for (int i = 0; i < 160; i++) {
        const int x = rng.uniform(8, size - 8), y = rng.uniform(8, size - 8);
        fill_circle(bg, x, y, rng.uniform(2, 5), (uint8_t)rng.uniform(150, 240));
    }
    gaussian_blur_u8(bg, 5);
    return bg;
}

}  // namespace scene
