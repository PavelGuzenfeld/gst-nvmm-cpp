/// Active-content bounds of a frame — excludes uniform letterbox / pillarbox bars.
///
/// Many capture/transcode pipelines pad non-16:9 sources with black (or constant
/// gray) bars. Anything keyed off the frame extent (ROI gating, edge-of-frame
/// rejection, normalisation) wants the *content* rectangle, not the padded one.
///
/// Detection is by per-row / per-column intensity RANGE (max−min), not mean: a bar
/// is a band that spans almost no gray levels. Using the range (not a mean/brightness
/// threshold) keeps it correct on a DARK-but-textured frame — e.g. a thermal/IR sky,
/// which is dim everywhere yet has real gradient, so a mean threshold would wrongly
/// classify it as a bar and crop the content.
///
/// OpenCV-only, header-only, no GStreamer/CUDA.
#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>

namespace nvmm {
namespace video {

struct ActiveRegionParams {
    int bar_range = 15;  // a row/col spanning fewer than this many gray levels is a uniform bar
};

/// Content rectangle of `gray` (CV_8U), trimming uniform border bars. Returns the
/// full frame when nothing looks like a bar (or the frame is entirely uniform).
inline cv::Rect active_region(const cv::Mat &gray, const ActiveRegionParams &p = {})
{
    CV_Assert(gray.type() == CV_8U && !gray.empty());
    cv::Mat cmin, cmax, rmin, rmax;   // REDUCE_MAX/MIN keep input depth (CV_8U) — can't promote
    cv::reduce(gray, cmax, 0, cv::REDUCE_MAX);
    cv::reduce(gray, cmin, 0, cv::REDUCE_MIN);
    cv::reduce(gray, rmax, 1, cv::REDUCE_MAX);
    cv::reduce(gray, rmin, 1, cv::REDUCE_MIN);
    auto crange = [&](int i){ return (int)cmax.at<uchar>(i) - (int)cmin.at<uchar>(i); };
    auto rrange = [&](int i){ return (int)rmax.at<uchar>(i) - (int)rmin.at<uchar>(i); };
    int x0 = 0;               while (x0 < gray.cols - 1 && crange(x0) < p.bar_range) x0++;
    int x1 = gray.cols - 1;   while (x1 > x0           && crange(x1) < p.bar_range) x1--;
    int y0 = 0;               while (y0 < gray.rows - 1 && rrange(y0) < p.bar_range) y0++;
    int y1 = gray.rows - 1;   while (y1 > y0           && rrange(y1) < p.bar_range) y1--;
    // Entirely-uniform frame (every row/col is a "bar") -> no content to trim, return full.
    if (x1 <= x0 || y1 <= y0) return cv::Rect(0, 0, gray.cols, gray.rows);
    return cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
}

}  // namespace video
}  // namespace nvmm
