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
/// HONEST NOTE: this is a generic reference tool. It magnifies SMALL periodic
/// signals against a static or slowly-varying background; it does NOT survive large
/// camera motion / parallax (the band-pass treats the global motion as signal). For
/// independent-motion detection on a panning camera, see dual_homography.hpp.
///
/// OpenCV-only, header-only, no GStreamer/CUDA.
#pragma once
#include <opencv2/opencv.hpp>
#include <cmath>

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
        r_low_  = 1.f - std::exp(-2.f * (float)CV_PI * p_.low_hz  / p_.fps);
        r_high_ = 1.f - std::exp(-2.f * (float)CV_PI * p_.high_hz / p_.fps);
    }

    /// Push one frame (CV_8U or CV_32F, 1- or 3-channel) and get the magnified frame
    /// back as CV_32F. The first frame initialises the filters and is returned as-is.
    cv::Mat process(const cv::Mat &frame) {
        cv::Mat x;
        frame.convertTo(x, CV_32F);
        if (p_.blur > 0) cv::GaussianBlur(x, x, cv::Size(p_.blur, p_.blur), 0);
        if (!init_) { lp_low_ = x.clone(); lp_high_ = x.clone(); init_ = true; return x; }
        lp_low_  += r_low_  * (x - lp_low_);
        lp_high_ += r_high_ * (x - lp_high_);
        cv::Mat band = lp_high_ - lp_low_;     // temporal band-pass
        return x + p_.alpha * band;            // magnified
    }

    void reset() { init_ = false; }

private:
    MagnifyParams p_;
    float r_low_ = 0.f, r_high_ = 0.f;
    cv::Mat lp_low_, lp_high_;
    bool init_ = false;
};

}  // namespace motion
}  // namespace nvmm
