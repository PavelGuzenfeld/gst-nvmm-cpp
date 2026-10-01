#pragma once
#include <cmath>
#include <cstdint>

#include "image.hpp"
#include "image_ops.hpp"

namespace nvmm {
namespace motion {

struct MagnifyParams {
    /// Frames per second; sets the cutoffs in Hz.
    float fps = 30.f;
    float low_hz = 0.5f;
    /// Must exceed low_hz.
    float high_hz = 3.f;
    float alpha = 10.f;
    /// Odd kernel, 0 disables; restricts magnification to coarse motion.
    int   blur = 0;
};

/// Linear Eulerian video magnification (Wu et al., SIGGRAPH 2012) with the streaming
/// IIR bandpass: two first-order low-passes, O(1) state per pixel. Global camera
/// motion passes as signal, so it does not survive panning.
class MotionMagnifier {
public:
    explicit MotionMagnifier(const MagnifyParams &p = {}) : p_(p) {
        const float two_pi = 6.28318530717958647692f;
        r_low_  = 1.f - std::exp(-two_pi * p_.low_hz  / p_.fps);
        r_high_ = 1.f - std::exp(-two_pi * p_.high_hz / p_.fps);
    }

    /// The first frame initialises the filters and is returned as is (blurred if enabled).
    img::Image<float> process(img::View<const uint8_t> frame) { return run(frame); }
    img::Image<float> process(img::View<const float> frame) { return run(frame); }

    void reset() { init_ = false; }

private:
    template <typename SrcT>
    img::Image<float> run(img::View<const SrcT> frame) {
        img::Image<float> out(frame.width, frame.height);
        if (p_.blur > 0) {
            const std::vector<float> k = img::gaussian_kernel(p_.blur);
            if (tmp_.width() != frame.width || tmp_.height() != frame.height)
                tmp_ = img::Image<float>(frame.width, frame.height);
            img::convolve_rows(frame, tmp_.view(), k);
            img::convolve_cols(tmp_.view(), k, [&](int y, const float *row, int w) {
                step_row(y, row, w, out);
            });
        } else {
            for (int y = 0; y < frame.height; y++)
                step_row(y, as_float_row(frame.row(y), frame.width), frame.width, out);
        }
        init_ = true;
        return out;
    }

    const float *as_float_row(const float *s, int) { return s; }
    const float *as_float_row(const uint8_t *s, int w) {
        xrow_.assign(s, s + w);
        return xrow_.data();
    }

    void step_row(int y, const float *xv, int w, img::Image<float> &out) {
        if (!init_) {
            if (y == 0) {
                lp_low_  = img::Image<float>(out.width(), out.height());
                lp_high_ = img::Image<float>(out.width(), out.height());
            }
            std::copy(xv, xv + w, lp_low_.row(y));
            std::copy(xv, xv + w, lp_high_.row(y));
            std::copy(xv, xv + w, out.row(y));
            return;
        }
        float *ll = lp_low_.row(y), *lh = lp_high_.row(y), *o = out.row(y);
        for (int x = 0; x < w; x++) {
            ll[x] += r_low_  * (xv[x] - ll[x]);
            lh[x] += r_high_ * (xv[x] - lh[x]);
            o[x] = xv[x] + p_.alpha * (lh[x] - ll[x]);
        }
    }

    MagnifyParams p_;
    float r_low_ = 0.f, r_high_ = 0.f;
    img::Image<float> lp_low_, lp_high_, tmp_;
    std::vector<float> xrow_;
    bool init_ = false;
};

}
}
