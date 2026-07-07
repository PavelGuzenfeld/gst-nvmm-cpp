/// Frame-difference motion restricted to LOW-TEXTURE regions.
///
/// A plain frame difference on a moving camera is swamped by edge/parallax clutter
/// wherever the background is textured. But in a LOW-gradient region — open sky,
/// water, a smooth wall — a small object moving across it produces a clean diff with
/// nothing else to confuse it. This computes exactly that: build a mask of the
/// low-gradient area and frame-difference only inside it.
///
/// Two reference frames (e.g. two different past frames) are min-combined so a
/// transient that appears against only one reference is suppressed. Pairs naturally
/// with dual_homography.hpp (textured-background mover) — this covers the
/// low-texture-background case the homography residual can't (no features there).
///
/// The OpenCV op chain (Sobel×2, magnitude, blur, compare, morphologyEx, absdiff×2,
/// min, convert, masked copy) is fused into four sweeps:
///   1. Sobel-x + Sobel-y + gradient magnitude in one pass;
///   2. separable Gaussian blur whose final pass fuses the threshold and feeds the
///      mask's integral image directly (no intermediate blurred frame or raw mask);
///   3. morphological close via box sums on integral images — exact for a binary
///      mask with a square SE, O(1) per pixel regardless of kernel size (the
///      border behaves like OpenCV's ±inf morphology border: outside pixels never
///      dilate and never block erosion);
///   4. min(|cur−ref_a|, |cur−ref_b|) · mask -> float output in one pass.
///
/// Pure C++14, header-only, no dependencies.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "image.hpp"
#include "image_ops.hpp"

namespace nvmm {
namespace motion {

struct LowTextureMotionParams {
    float grad_thresh = 14.f;  // gradient magnitude below this (after blur) = "low texture"
    int   grad_blur = 7;       // blur applied to the gradient before thresholding (odd)
    int   close_k = 15;        // morphological close to fill the hole the object cuts in the mask
    int   diff_blur = 3;       // blur applied to the output motion (odd; 0 = none)
    int   border = 12;         // zero out this many px around the frame border
};

namespace detail {

/// Sobel-x, Sobel-y (3x3, REFLECT_101 border) and gradient magnitude, one sweep.
inline void sobel_magnitude(img::View<const uint8_t> src, img::View<float> mag)
{
    const int w = src.width, h = src.height;
    for (int y = 0; y < h; y++) {
        const uint8_t *ym = src.row(img::reflect101(y - 1, h));
        const uint8_t *y0 = src.row(y);
        const uint8_t *yp = src.row(img::reflect101(y + 1, h));
        float *d = mag.row(y);
        auto at = [&](int x) {
            const int xm = img::reflect101(x - 1, w), xp = img::reflect101(x + 1, w);
            const int gx = ((int)ym[xp] + 2 * y0[xp] + yp[xp]) - ((int)ym[xm] + 2 * y0[xm] + yp[xm]);
            const int gy = ((int)yp[xm] + 2 * yp[x] + yp[xp]) - ((int)ym[xm] + 2 * ym[x] + ym[xp]);
            d[x] = std::sqrt((float)(gx * gx + gy * gy));
        };
        at(0);
        for (int x = 1; x < w - 1; x++) {   // no border folding on the hot path
            const int gx = ((int)ym[x + 1] + 2 * y0[x + 1] + yp[x + 1]) -
                           ((int)ym[x - 1] + 2 * y0[x - 1] + yp[x - 1]);
            const int gy = ((int)yp[x - 1] + 2 * yp[x] + yp[x + 1]) -
                           ((int)ym[x - 1] + 2 * ym[x] + ym[x + 1]);
            d[x] = std::sqrt((float)(gx * gx + gy * gy));
        }
        if (w > 1) at(w - 1);
    }
}

/// Integral image builder: I(y+1, x+1) = sum of m over [0,y]x[0,x]. `integ` has
/// (w+1)*(h+1) elements, row-major, first row/col zero.
class BoxSum {
public:
    BoxSum(int w, int h) : w_(w), h_(h), integ_((size_t)(w + 1) * (h + 1), 0u) {}

    /// Accumulate row y from per-pixel 0/1 values produced by `bit(x)`.
    template <typename BitFn>
    void add_row(int y, BitFn &&bit) {
        const uint32_t *prev = &integ_[(size_t)y * (w_ + 1)];
        uint32_t *out = &integ_[(size_t)(y + 1) * (w_ + 1)];
        uint32_t rowsum = 0;
        out[0] = 0;
        for (int x = 0; x < w_; x++) {
            rowsum += bit(x);
            out[x + 1] = prev[x + 1] + rowsum;
        }
    }

    /// Sum over the window [x-r, x+r] x [y-r, y+r] clipped to the image, plus the
    /// clipped window's area (for the erode-side "all ones" test).
    uint32_t window(int x, int y, int r, int &area) const {
        const int x0 = std::max(0, x - r), y0 = std::max(0, y - r);
        const int x1 = std::min(w_ - 1, x + r), y1 = std::min(h_ - 1, y + r);
        area = (x1 - x0 + 1) * (y1 - y0 + 1);
        const size_t stride = (size_t)w_ + 1;
        const uint32_t *a = &integ_[(size_t)y0 * stride];
        const uint32_t *b = &integ_[(size_t)(y1 + 1) * stride];
        return b[x1 + 1] - b[x0] - a[x1 + 1] + a[x0];
    }

private:
    int w_, h_;
    std::vector<uint32_t> integ_;
};

/// Low-texture mask of `cur`: blurred gradient magnitude below `grad_thresh`,
/// then morphologically closed with a close_k square SE. Returns 0/1 per pixel.
inline img::Image<uint8_t> low_texture_mask(img::View<const uint8_t> cur,
                                            const LowTextureMotionParams &p)
{
    const int w = cur.width, h = cur.height;
    img::Image<float> grad(w, h), tmp(w, h);
    sobel_magnitude(cur, grad.view());

    // blur -> threshold -> integral, with no blurred frame or raw mask buffer
    const std::vector<float> k = img::gaussian_kernel(p.grad_blur);
    img::convolve_rows<float>(grad.view(), tmp.view(), k);
    BoxSum low_sum(w, h);
    img::convolve_cols(tmp.view(), k, [&](int y, const float *row, int) {
        low_sum.add_row(y, [&](int x) { return row[x] < p.grad_thresh ? 1u : 0u; });
    });

    // close = erode(dilate): box sums make both O(1)/px and exact on a binary mask.
    // The dilated value at each x is needed only to feed dil_sum's running row
    // sum, so compute it inline rather than materialising a whole dilated frame.
    const int r = p.close_k / 2;
    BoxSum dil_sum(w, h);
    for (int y = 0; y < h; y++) {
        int area;
        dil_sum.add_row(y, [&](int x) {
            return low_sum.window(x, y, r, area) > 0 ? 1u : 0u;
        });
    }
    img::Image<uint8_t> closed(w, h);
    for (int y = 0; y < h; y++) {
        uint8_t *cr = closed.row(y);
        for (int x = 0; x < w; x++) {
            int area;
            const uint32_t sum = dil_sum.window(x, y, r, area);
            cr[x] = sum == (uint32_t)area ? 1 : 0;
        }
    }
    return closed;
}

}  // namespace detail

/// Motion (float) of `cur` vs two references, kept only in low-texture regions.
/// All inputs single-channel u8, same size.
inline img::Image<float> low_texture_motion(img::View<const uint8_t> cur,
                                            img::View<const uint8_t> ref_a,
                                            img::View<const uint8_t> ref_b,
                                            const LowTextureMotionParams &p = {})
{
    const int w = cur.width, h = cur.height;
    img::Image<uint8_t> mask = detail::low_texture_mask(cur, p);

    img::Image<float> out(w, h);
    for (int y = 0; y < h; y++) {
        const uint8_t *c = cur.row(y), *ra = ref_a.row(y), *rb = ref_b.row(y);
        const uint8_t *m = mask.row(y);
        float *o = out.row(y);
        for (int x = 0; x < w; x++) {
            const int da = c[x] > ra[x] ? c[x] - ra[x] : ra[x] - c[x];
            const int db = c[x] > rb[x] ? c[x] - rb[x] : rb[x] - c[x];
            o[x] = m[x] ? (float)std::min(da, db) : 0.f;
        }
    }
    if (p.diff_blur > 0) {
        img::Image<float> tmp;
        img::gaussian_blur<float>(out.view(), tmp, out.view(), p.diff_blur);
    }
    img::zero_border(out.view(), p.border);
    return out;
}

}  // namespace motion
}  // namespace nvmm
