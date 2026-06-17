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
/// OpenCV-only, header-only, no GStreamer/CUDA.
#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>

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
inline void zero_border_f(cv::Mat &m, int mb) {
    if (m.empty() || mb <= 0) return;
    m.rowRange(0, std::min(mb, m.rows)) = 0;
    m.rowRange(std::max(0, m.rows - mb), m.rows) = 0;
    m.colRange(0, std::min(mb, m.cols)) = 0;
    m.colRange(std::max(0, m.cols - mb), m.cols) = 0;
}
}  // namespace detail

/// Motion (CV_32F) of `cur` vs two references, kept only in low-texture regions.
/// All inputs single-channel CV_8U, same size.
inline cv::Mat low_texture_motion(const cv::Mat &cur, const cv::Mat &ref_a,
                                  const cv::Mat &ref_b,
                                  const LowTextureMotionParams &p = {})
{
    CV_Assert(cur.type() == CV_8U && ref_a.size() == cur.size() && ref_b.size() == cur.size());
    cv::Mat gx, gy, grad, gblur, low;
    cv::Sobel(cur, gx, CV_32F, 1, 0);
    cv::Sobel(cur, gy, CV_32F, 0, 1);
    cv::magnitude(gx, gy, grad);
    cv::GaussianBlur(grad, gblur, cv::Size(p.grad_blur, p.grad_blur), 0);
    low = (gblur < p.grad_thresh);                         // low-texture mask
    cv::morphologyEx(low, low, cv::MORPH_CLOSE,
                     cv::Mat::ones(p.close_k, p.close_k, CV_8U));  // fill the object's own hole
    cv::Mat da, db, dmin, out;
    cv::absdiff(cur, ref_a, da);
    cv::absdiff(cur, ref_b, db);
    dmin = cv::min(da, db);
    dmin.convertTo(dmin, CV_32F);
    out = cv::Mat::zeros(cur.size(), CV_32F);
    dmin.copyTo(out, low);                                  // keep diff only in low-texture area
    if (p.diff_blur > 0) cv::GaussianBlur(out, out, cv::Size(p.diff_blur, p.diff_blur), 0);
    detail::zero_border_f(out, p.border);
    return out;
}

}  // namespace motion
}  // namespace nvmm
