/// Plane+parallax independent-motion residual for a moving (panning) camera.
///
/// Frame differencing and single-homography background subtraction both fail on a
/// translating camera: a single dominant-plane homography leaves strong residual
/// wherever the scene has depth (near-field parallax), which then masquerades as
/// motion. This computes a DUAL-homography residual instead:
///
///   1. ORB-match the current frame to a reference frame and fit H1 (RANSAC) — the
///      dominant background plane.
///   2. Re-fit H2 on the RANSAC OUTLIERS — a second plane that captures the
///      parallax structure the first plane missed.
///   3. Warp the reference onto the current frame by each plane and keep, per pixel,
///      the residual that survives BOTH planes (element-wise min).
///
/// A pixel that is explained by either plane (flat background OR near-field parallax)
/// is suppressed; only genuinely independent motion — an object moving relative to
/// the whole scene — keeps a high residual. Two reference frames (e.g. two different
/// past frames) are combined the same way so a static edge that aligns under one
/// reference but not the other is also rejected.
///
/// OpenCV-only, header-only, no GStreamer/CUDA — unit-testable on the host.
/// (Distinct from common/nvmm_motion.hpp, which scores boxes from a precomputed
///  optical-flow field; this works directly on the raw grayscale frames.)
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace nvmm {
namespace motion {

struct DualHomographyParams {
    int    orb_features = 4000;  // ORB keypoints per frame
    double ransac_thresh = 3.0;  // findHomography RANSAC reprojection threshold (px)
    int    min_matches = 20;     // need at least this many good matches to fit a plane
    int    valid_erode = 15;     // erode the warp's valid region by this (drop warp edges)
    int    blur = 3;             // Gaussian blur applied to the residual (odd; 0 = none)
    int    border = 12;          // zero out this many px around the frame border
    float  ratio = 0.75f;        // Lowe ratio for the ORB match
};

namespace detail {

inline bool orb_match(cv::ORB &orb, cv::BFMatcher &bf, float ratio,
                      const cv::Mat &a, const cv::Mat &b,
                      std::vector<cv::Point2f> &p1, std::vector<cv::Point2f> &p2)
{
    std::vector<cv::KeyPoint> k1, k2; cv::Mat d1, d2;
    orb.detectAndCompute(a, cv::noArray(), k1, d1);
    orb.detectAndCompute(b, cv::noArray(), k2, d2);
    if (d1.empty() || d2.empty()) return false;
    std::vector<std::vector<cv::DMatch>> knn;
    bf.knnMatch(d1, d2, knn, 2);
    p1.clear(); p2.clear();
    for (auto &m : knn)
        if (m.size() == 2 && m[0].distance < ratio * m[1].distance) {
            p1.push_back(k1[m[0].queryIdx].pt);
            p2.push_back(k2[m[0].trainIdx].pt);
        }
    return (int)p1.size() >= 20;
}

// dual homography a->b: H1 = dominant plane, H2 = parallax plane (RANSAC outliers).
inline void two_planes(cv::ORB &orb, cv::BFMatcher &bf, const DualHomographyParams &p,
                       const cv::Mat &a, const cv::Mat &b, cv::Mat &H1, cv::Mat &H2)
{
    H1.release(); H2.release();
    std::vector<cv::Point2f> p1, p2;
    if (!orb_match(orb, bf, p.ratio, a, b, p1, p2)) return;
    cv::Mat mask;
    H1 = cv::findHomography(p1, p2, cv::RANSAC, p.ransac_thresh, mask);
    std::vector<cv::Point2f> o1, o2;
    for (int i = 0; i < mask.rows; i++)
        if (!mask.at<uchar>(i)) { o1.push_back(p1[i]); o2.push_back(p2[i]); }
    if ((int)o1.size() >= p.min_matches) H2 = cv::findHomography(o1, o2, cv::RANSAC, p.ransac_thresh);
}

// residual of `a` vs `nb` warped by inv(H): |a - warp(nb)| inside the valid region.
inline cv::Mat residual(const cv::Mat &a, const cv::Mat &nb, const cv::Mat &H, int erode_k)
{
    if (H.empty()) return cv::Mat();
    cv::Mat Hinv = H.inv(), wp, dd;
    cv::warpPerspective(nb, wp, Hinv, a.size());
    cv::absdiff(a, wp, dd); dd.convertTo(dd, CV_32F);
    cv::Mat ones = cv::Mat::ones(a.size(), CV_32F), vw, vm;
    cv::warpPerspective(ones, vw, Hinv, a.size());
    cv::erode((vw > 0.5f), vm, cv::Mat::ones(erode_k, erode_k, CV_8U));
    cv::Mat out = cv::Mat::zeros(a.size(), CV_32F);
    dd.copyTo(out, vm);
    return out;
}

// combine the residuals to both references through one plane (min => survive both).
inline cv::Mat combine(const cv::Mat &cur, const cv::Mat &ra, const cv::Mat &rb,
                       const cv::Mat &Ha, const cv::Mat &Hb, int erode_k)
{
    cv::Mat r1 = residual(cur, ra, Ha, erode_k), r2 = residual(cur, rb, Hb, erode_k);
    if (r1.empty() && r2.empty()) return cv::Mat();
    if (r1.empty()) return r2;
    if (r2.empty()) return r1;
    return cv::min(r1, r2);
}

inline void zero_border(cv::Mat &m, int mb)
{
    if (m.empty() || mb <= 0) return;
    m.rowRange(0, std::min(mb, m.rows)) = 0;
    m.rowRange(std::max(0, m.rows - mb), m.rows) = 0;
    m.colRange(0, std::min(mb, m.cols)) = 0;
    m.colRange(std::max(0, m.cols - mb), m.cols) = 0;
}

}  // namespace detail

/// Independent-motion residual of `cur` against two reference frames.
/// `cur`, `ref_a`, `ref_b` are single-channel CV_8U (grayscale), same size.
/// Returns CV_32F (same size): high where motion is unexplained by either the
/// dominant plane or the parallax plane. Empty if no homography could be fit
/// (e.g. textureless input) — callers should treat empty as "no estimate".
inline cv::Mat independent_motion_residual(const cv::Mat &cur, const cv::Mat &ref_a,
                                           const cv::Mat &ref_b,
                                           const DualHomographyParams &p = {})
{
    CV_Assert(cur.type() == CV_8U && ref_a.size() == cur.size() && ref_b.size() == cur.size());
    cv::Ptr<cv::ORB> orb = cv::ORB::create(p.orb_features);
    cv::BFMatcher bf(cv::NORM_HAMMING);
    cv::Mat Ha1, Ha2, Hb1, Hb2;
    detail::two_planes(*orb, bf, p, cur, ref_a, Ha1, Ha2);
    detail::two_planes(*orb, bf, p, cur, ref_b, Hb1, Hb2);
    cv::Mat r1 = detail::combine(cur, ref_a, ref_b, Ha1, Hb1, p.valid_erode);
    if (r1.empty()) return cv::Mat();
    cv::Mat r2 = detail::combine(cur, ref_a, ref_b, Ha2, Hb2, p.valid_erode);
    if (p.blur > 0) cv::GaussianBlur(r1, r1, cv::Size(p.blur, p.blur), 0);
    cv::Mat sc;
    if (!r2.empty()) { cv::GaussianBlur(r2, r2, cv::Size(p.blur, p.blur), 0); sc = cv::min(r1, r2); }
    else sc = r1;
    detail::zero_border(sc, p.border);
    return sc;
}

}  // namespace motion
}  // namespace nvmm
