/// Shared plumbing for the golden-comparison tests: nvmm::img <-> cv::Mat
/// converters and aggregate difference metrics. Test-only; the production
/// headers never see OpenCV.
#pragma once
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "image.hpp"

namespace golden {

inline cv::Mat to_cv(const nvmm::img::Image<uint8_t> &im)
{
    cv::Mat m(im.height(), im.width(), CV_8U);
    for (int y = 0; y < im.height(); y++)
        for (int x = 0; x < im.width(); x++) m.at<uchar>(y, x) = im.at(y, x);
    return m;
}

inline cv::Mat to_cv(const nvmm::img::Image<float> &im)
{
    cv::Mat m(im.height(), im.width(), CV_32F);
    for (int y = 0; y < im.height(); y++)
        for (int x = 0; x < im.width(); x++) m.at<float>(y, x) = im.at(y, x);
    return m;
}

inline nvmm::img::Image<uint8_t> from_cv_u8(const cv::Mat &m)
{
    nvmm::img::Image<uint8_t> im(m.cols, m.rows);
    for (int y = 0; y < m.rows; y++)
        for (int x = 0; x < m.cols; x++) im.at(y, x) = m.at<uchar>(y, x);
    return im;
}

/// max |ours - reference| over the whole frame (reference is CV_32F).
inline double max_abs_diff(const nvmm::img::Image<float> &ours, const cv::Mat &ref)
{
    double mx = 0;
    for (int y = 0; y < ours.height(); y++)
        for (int x = 0; x < ours.width(); x++)
            mx = std::max(mx, std::fabs((double)ours.at(y, x) - ref.at<float>(y, x)));
    return mx;
}

/// Fraction of pixels where two binary masks disagree (ours: 0/1, ref: 0/255).
inline double mask_disagree_frac(const nvmm::img::Image<uint8_t> &ours, const cv::Mat &ref)
{
    long bad = 0;
    for (int y = 0; y < ours.height(); y++)
        for (int x = 0; x < ours.width(); x++)
            bad += (ours.at(y, x) != 0) != (ref.at<uchar>(y, x) != 0) ? 1 : 0;
    return (double)bad / ((double)ours.width() * ours.height());
}

}  // namespace golden
