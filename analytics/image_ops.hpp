/// Shared building blocks for the fused analytics passes.
///
/// The separable Gaussian blur is written so a component can FUSE its next step
/// into the final (vertical) pass via an emit functor — thresholding, IIR
/// updates, masking — instead of materialising an intermediate and sweeping the
/// frame again. Kernel weights and borders reproduce OpenCV's defaults
/// (getGaussianKernel's fixed small-size tables / sigma formula, and
/// BORDER_REFLECT_101) so the golden-comparison tests can hold tight
/// tolerances; the tables are the only OpenCV-derived numerics here.
///
/// Pure C++14, header-only, no dependencies.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "image.hpp"

// Lets analytics_kernels.cu call this file's border/kernel helpers directly
// from CUDA device code instead of keeping a second copy in sync by hand.
// A no-op qualifier outside nvcc, so plain C++14 TUs are unaffected.
#if defined(__CUDACC__)
#define NVMM_ANALYTICS_HD __host__ __device__
#else
#define NVMM_ANALYTICS_HD
#endif

namespace nvmm {
namespace img {

/// BORDER_REFLECT_101 index fold: gfedcb|abcdefgh|gfedcba. Valid for n > 1 and
/// any index one kernel-radius out of range (the analytics kernels are far
/// smaller than the frames).
NVMM_ANALYTICS_HD inline int reflect101(int i, int n)
{
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}

/// Weights of cv::getGaussianKernel(k, sigma = 0): fixed bit-exact tables for
/// odd k <= 9 (OpenCV 4.x, n/256 values), otherwise sigma = 0.3*((k-1)*0.5-1) + 0.8.
inline std::vector<float> gaussian_kernel(int k)
{
    std::vector<float> w((size_t)k);
    static const float k1[] = {1.f};
    static const float k3[] = {0.25f, 0.5f, 0.25f};
    static const float k5[] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
    static const float k7[] = {0.03125f, 0.109375f, 0.21875f, 0.28125f,
                               0.21875f, 0.109375f, 0.03125f};
    static const float k9[] = {0.015625f, 0.05078125f, 0.1171875f, 0.19921875f,
                               0.234375f, 0.19921875f, 0.1171875f, 0.05078125f,
                               0.015625f};
    const float *fixed = (k == 1) ? k1 : (k == 3) ? k3 : (k == 5) ? k5
                       : (k == 7) ? k7 : (k == 9) ? k9 : nullptr;
    if (fixed) { std::copy(fixed, fixed + k, w.begin()); return w; }
    const double sigma = 0.3 * ((k - 1) * 0.5 - 1.0) + 0.8;
    double sum = 0.0;
    for (int i = 0; i < k; i++) {
        const double d = i - (k - 1) * 0.5;
        const double v = std::exp(-d * d / (2.0 * sigma * sigma));
        w[(size_t)i] = (float)v;
        sum += v;
    }
    for (int i = 0; i < k; i++) w[(size_t)i] = (float)(w[(size_t)i] / sum);
    return w;
}

/// Horizontal pass of a separable convolution: src (u8 or float) -> float dst.
/// dst may not alias src.
template <typename SrcT>
inline void convolve_rows(View<const SrcT> src, View<float> dst, const std::vector<float> &k)
{
    const int r = (int)k.size() / 2, w = src.width, h = src.height;
    const float *kp = k.data();
    for (int y = 0; y < h; y++) {
        const SrcT *s = src.row(y);
        float *d = dst.row(y);
        auto edge = [&](int x) {
            float acc = 0.f;
            for (int i = 0; i < (int)k.size(); i++)
                acc += kp[i] * (float)s[reflect101(x - r + i, w)];
            d[x] = acc;
        };
        int x = 0;
        const int interior_end = w - r;
        for (; x < r && x < w; x++) edge(x);
        for (; x < interior_end; x++) {   // no border folding on the hot path
            float acc = 0.f;
            const SrcT *sp = s + x - r;
            for (int i = 0; i < (int)k.size(); i++) acc += kp[i] * (float)sp[i];
            d[x] = acc;
        }
        for (; x < w; x++) edge(x);
    }
}

/// Vertical pass of a separable convolution, fused with the caller's next step:
/// emit(y, row_ptr, width) receives each finished output row (a scratch buffer,
/// valid only during the call). Row-major accumulation keeps this cache-friendly.
template <typename Emit>
inline void convolve_cols(View<const float> src, const std::vector<float> &k, Emit &&emit)
{
    const int r = (int)k.size() / 2, w = src.width, h = src.height;
    std::vector<float> acc((size_t)w);
    for (int y = 0; y < h; y++) {
        std::fill(acc.begin(), acc.end(), 0.f);
        for (int i = 0; i < (int)k.size(); i++) {
            const float *s = src.row(reflect101(y - r + i, h));
            const float kv = k[(size_t)i];
            for (int x = 0; x < w; x++) acc[(size_t)x] += kv * s[x];
        }
        emit(y, acc.data(), w);
    }
}

/// Gaussian blur matching cv::GaussianBlur(src, dst, {k,k}, 0) on float data.
/// `tmp` is resized as needed; dst may alias src.
template <typename SrcT>
inline void gaussian_blur(View<const SrcT> src, Image<float> &tmp, View<float> dst, int ksize)
{
    const std::vector<float> k = gaussian_kernel(ksize);
    if (tmp.width() != src.width || tmp.height() != src.height)
        tmp = Image<float>(src.width, src.height);
    convolve_rows(src, tmp.view(), k);
    convolve_cols(tmp.view(), k, [&](int y, const float *row, int w) {
        std::copy(row, row + w, dst.row(y));
    });
}

/// Zero `mb` pixels around the border (writes only the border, not a full pass).
inline void zero_border(View<float> m, int mb)
{
    if (m.empty() || mb <= 0) return;
    const int w = m.width, h = m.height;
    const int top = std::min(mb, h), bot = std::max(0, h - mb);
    for (int y = 0; y < top; y++) std::fill(m.row(y), m.row(y) + w, 0.f);
    for (int y = bot; y < h; y++) std::fill(m.row(y), m.row(y) + w, 0.f);
    const int lw = std::min(mb, w);
    for (int y = top; y < bot; y++) {
        float *r = m.row(y);
        std::fill(r, r + lw, 0.f);
        std::fill(r + std::max(0, w - mb), r + w, 0.f);
    }
}

/// Max over the window [cx-r, cx+r) x [cy-r, cy+r) clipped to the image
/// (replaces cv::minMaxLoc on a sub-rect). 0 on an empty view/window.
inline float window_max(View<const float> m, float cx, float cy, int r)
{
    if (m.empty()) return 0.f;
    const int x0 = std::max(0, (int)cx - r), y0 = std::max(0, (int)cy - r);
    const int x1 = std::min(m.width, (int)cx + r), y1 = std::min(m.height, (int)cy + r);
    if (x1 <= x0 || y1 <= y0) return 0.f;
    float mx = m.at(y0, x0);
    for (int y = y0; y < y1; y++) {
        const float *row = m.row(y);
        for (int x = x0; x < x1; x++) mx = std::max(mx, row[x]);
    }
    return mx;
}

}  // namespace img
}  // namespace nvmm
