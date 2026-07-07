/// Eulerian motion / intensity magnification (linear method, streaming IIR).
///
/// Reference implementation of Wu et al., "Eulerian Video Magnification for
/// Revealing Subtle Changes in the World" (SIGGRAPH 2012), linear variant: amplify
/// the temporally band-passed intensity at each pixel to make small periodic motion
/// (or colour change) visible. Streaming/online — a per-pixel temporal bandpass
/// built from two first-order IIR low-passes (the paper's real-time "IIR" filter),
/// so it runs frame-by-frame with O(1) state per pixel and no frame buffer.
///
///   lp_low  += r_low  * (x - lp_low)     // slow low-pass  (low cutoff)
///   lp_high += r_high * (x - lp_high)    // fast low-pass  (high cutoff)
///   band     = lp_high - lp_low          // temporal band-pass [low_hz, high_hz]
///   out      = x + alpha * band          // magnified frame
///
/// Convert + both IIR updates + band + output happen in ONE fused sweep (the
/// OpenCV version materialised ~6 whole-frame temporaries per frame); the
/// optional spatial blur adds a separable pre-pass whose vertical pass feeds
/// the IIR step directly.
///
/// HONEST NOTE: this is a generic reference tool. It magnifies SMALL periodic
/// signals against a static or slowly-varying background; it does NOT survive large
/// camera motion / parallax (the band-pass treats the global motion as signal). For
/// independent-motion detection on a panning camera, see dual_homography.hpp.
///
/// Pure C++14, header-only, no dependencies. Single-channel input (u8 or float);
/// process planes independently for multi-channel data.
#pragma once
#include <cmath>
#include <cstdint>

#include "image.hpp"
#include "image_ops.hpp"

namespace nvmm {
namespace motion {

struct MagnifyParams {
    float fps = 30.f;     // sample rate (frames/second) — sets the cutoffs in Hz
    float low_hz = 0.5f;  // temporal pass-band low edge
    float high_hz = 3.f;  // temporal pass-band high edge (> low_hz)
    float alpha = 10.f;   // magnification factor for the band-passed signal
    int   blur = 0;       // optional spatial Gaussian blur kernel (odd; 0 = none) to
                          // restrict magnification to coarse motion and curb noise
};

class MotionMagnifier {
public:
    explicit MotionMagnifier(const MagnifyParams &p = {}) : p_(p) {
        // first-order discrete low-pass coefficient for a cutoff fc: r = 1 - e^{-2pi fc/fps}
        const float two_pi = 6.28318530717958647692f;
        r_low_  = 1.f - std::exp(-two_pi * p_.low_hz  / p_.fps);
        r_high_ = 1.f - std::exp(-two_pi * p_.high_hz / p_.fps);
    }

    /// Push one frame and get the magnified frame back as float. The first
    /// frame initialises the filters and is returned as-is (blurred if enabled).
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

}  // namespace motion
}  // namespace nvmm
