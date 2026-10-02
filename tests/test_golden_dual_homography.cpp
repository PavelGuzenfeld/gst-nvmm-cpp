#include "dual_homography.hpp"
#include "analytics_scene.h"
#include "golden_util.h"
#include "test_harness.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using namespace nvmm;

namespace reference {

struct Params {
    int orb_features = 4000;
    double ransac_thresh = 3.0;
    int min_matches = 20;
    int valid_erode = 15;
    int blur = 3;
    int border = 12;
    float ratio = 0.75f;
};

bool orb_match(cv::ORB &orb, cv::BFMatcher &bf, float ratio,
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

void two_planes(cv::ORB &orb, cv::BFMatcher &bf, const Params &p,
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

cv::Mat residual(const cv::Mat &a, const cv::Mat &nb, const cv::Mat &H, int erode_k)
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

cv::Mat combine(const cv::Mat &cur, const cv::Mat &ra, const cv::Mat &rb,
                const cv::Mat &Ha, const cv::Mat &Hb, int erode_k)
{
    cv::Mat r1 = residual(cur, ra, Ha, erode_k), r2 = residual(cur, rb, Hb, erode_k);
    if (r1.empty() && r2.empty()) return cv::Mat();
    if (r1.empty()) return r2;
    if (r2.empty()) return r1;
    return cv::min(r1, r2);
}

void zero_border(cv::Mat &m, int mb)
{
    if (m.empty() || mb <= 0) return;
    m.rowRange(0, std::min(mb, m.rows)) = 0;
    m.rowRange(std::max(0, m.rows - mb), m.rows) = 0;
    m.colRange(0, std::min(mb, m.cols)) = 0;
    m.colRange(std::max(0, m.cols - mb), m.cols) = 0;
}

cv::Mat independent_motion_residual(const cv::Mat &cur, const cv::Mat &ref_a,
                                    const cv::Mat &ref_b, const Params &p = {})
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(p.orb_features);
    cv::BFMatcher bf(cv::NORM_HAMMING);
    cv::Mat Ha1, Ha2, Hb1, Hb2;
    two_planes(*orb, bf, p, cur, ref_a, Ha1, Ha2);
    two_planes(*orb, bf, p, cur, ref_b, Hb1, Hb2);
    cv::Mat r1 = combine(cur, ref_a, ref_b, Ha1, Hb1, p.valid_erode);
    if (r1.empty()) return cv::Mat();
    cv::Mat r2 = combine(cur, ref_a, ref_b, Ha2, Hb2, p.valid_erode);
    if (p.blur > 0) cv::GaussianBlur(r1, r1, cv::Size(p.blur, p.blur), 0);
    cv::Mat sc;
    if (!r2.empty()) { cv::GaussianBlur(r2, r2, cv::Size(p.blur, p.blur), 0); sc = cv::min(r1, r2); }
    else sc = r1;
    zero_border(sc, p.border);
    return sc;
}

}

/// Valid regions differ by construction, so up to 3% of pixels may differ by > 1.0. In
/// both interiors the bound is 2.0: OpenCV warps through 5-bit fixed-point bilinear into U8.
TEST(fused_residual_matches_warp_chain_for_fixed_H) {
    img::Image<uint8_t> cur_i = scene::textured_bg(256, 21);
    img::Image<uint8_t> ref_i = scene::translate(cur_i, 5.3, -3.7);
    motion::detail::Mat3 H;
    H.m[0] = 1.002; H.m[1] = 0.001;  H.m[2] = 5.3;
    H.m[3] = -0.001; H.m[4] = 0.998; H.m[5] = -3.7;
    H.m[6] = 4e-6;  H.m[7] = -3e-6;  H.m[8] = 1.0;

    motion::detail::Plane pa; pa.H = H; pa.ok = true;
    motion::detail::Plane none;
    img::Image<float> ours;
    const int erode_k = 15;
    ASSERT_TRUE(motion::detail::plane_pair_residual(cur_i, ref_i, pa, ref_i, none,
                                                    erode_k, ours));

    cv::Mat Hcv = (cv::Mat_<double>(3, 3) << H.m[0], H.m[1], H.m[2], H.m[3], H.m[4],
                   H.m[5], H.m[6], H.m[7], H.m[8]);
    cv::Mat ref = reference::residual(golden::to_cv(cur_i), golden::to_cv(ref_i), Hcv, erode_k);

    long over = 0;
    double worst_interior = 0;
    const int inset = erode_k + 8;
    for (int y = 0; y < 256; y++)
        for (int x = 0; x < 256; x++) {
            const double d = std::fabs((double)ours.at(y, x) - ref.at<float>(y, x));
            if (d > 1.0) over++;
            if (y >= inset && y < 256 - inset && x >= inset && x < 256 - inset)
                worst_interior = std::max(worst_interior, d);
        }
    const double frac = (double)over / (256.0 * 256.0);
    printf("[edge>1.0: %.3f%%, interior worst %.3f] ", 100.0 * frac, worst_interior);
    ASSERT_TRUE(frac <= 0.03);
    ASSERT_TRUE(worst_interior <= 2.0);
}

/// OpenCV's RANSAC is randomized, so both fits are scored by reprojection error against
/// the planted H: ours <= max(0.35 px, 2x OpenCV's).
TEST(ransac_quality_comparable_to_findHomography) {
    motion::detail::Mat3 Ht;
    Ht.m[0] = 1.01; Ht.m[1] = 0.02;  Ht.m[2] = 6.0;
    Ht.m[3] = -0.015; Ht.m[4] = 0.99; Ht.m[5] = 4.0;
    Ht.m[6] = 1e-5; Ht.m[7] = -2e-5; Ht.m[8] = 1.0;

    motion::detail::Lcg rng(42);
    std::vector<motion::detail::Pt> p1, p2;
    std::vector<cv::Point2f> c1, c2;
    for (int i = 0; i < 300; i++) {
        const float x = (float)rng.below(256), y = (float)rng.below(256);
        double u = 0, v = 0;
        ASSERT_TRUE(Ht.project(x, y, u, v));
        u += (rng.below(200) - 100) / 400.0;
        v += (rng.below(200) - 100) / 400.0;
        if (i % 3 == 0) { u = rng.below(256); v = rng.below(256); }
        p1.push_back(motion::detail::Pt{x, y});
        p2.push_back(motion::detail::Pt{(float)u, (float)v});
        c1.push_back(cv::Point2f(x, y));
        c2.push_back(cv::Point2f((float)u, (float)v));
    }

    motion::detail::Mat3 Ho;
    std::vector<uint8_t> mask;
    ASSERT_TRUE(motion::detail::find_homography_ransac(p1, p2, 3.0, Ho, mask));
    cv::Mat Hcv = cv::findHomography(c1, c2, cv::RANSAC, 3.0);
    ASSERT_TRUE(!Hcv.empty());

    auto grid_err = [&](auto project) {
        double mx = 0;
        for (int y = 0; y < 256; y += 16)
            for (int x = 0; x < 256; x += 16) {
                double ut = 0, vt = 0, u = 0, v = 0;
                if (!Ht.project(x, y, ut, vt)) continue;
                project(x, y, u, v);
                mx = std::max(mx, std::hypot(u - ut, v - vt));
            }
        return mx;
    };
    const double ours = grid_err([&](int x, int y, double &u, double &v) {
        Ho.project(x, y, u, v);
    });
    const double cvs = grid_err([&](int x, int y, double &u, double &v) {
        const double w = Hcv.at<double>(2, 0) * x + Hcv.at<double>(2, 1) * y + Hcv.at<double>(2, 2);
        u = (Hcv.at<double>(0, 0) * x + Hcv.at<double>(0, 1) * y + Hcv.at<double>(0, 2)) / w;
        v = (Hcv.at<double>(1, 0) * x + Hcv.at<double>(1, 1) * y + Hcv.at<double>(1, 2)) / w;
    });
    printf("[grid err ours=%.3f cv=%.3f px] ", ours, cvs);
    ASSERT_TRUE(ours <= std::max(0.35, 2.0 * cvs));
}

constexpr unsigned kLegacySeed = 12345;
constexpr unsigned kFirstHeldOutSeed = 21;
constexpr unsigned kLastHeldOutSeed = 60;
constexpr double kMinRatioSmallMotionSweepMin0p986 = 0.9;
constexpr double kMinRatioOrbSweepMin0p777 = 0.7;

struct SeedOutcome {
    float mover[2];
    double ratio_vs_reference[2];
};

SeedOutcome separation_vs_reference(unsigned seed)
{
    img::Image<uint8_t> bg = scene::textured_bg(256, seed);
    img::Image<uint8_t> cur = bg;
    img::Image<uint8_t> ref_a = scene::translate(bg, 6, 4);
    img::Image<uint8_t> ref_b = scene::translate(bg, 12, 8);
    scene::fill_circle(cur, 180, 180, 6, 255);
    scene::fill_circle(ref_a, 150, 180, 6, 255);
    scene::fill_circle(ref_b, 120, 180, 6, 255);

    auto bg_median = [](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.f : v[v.size() / 2];
    };

    cv::Mat ref_res = reference::independent_motion_residual(
        golden::to_cv(cur), golden::to_cv(ref_a), golden::to_cv(ref_b));
    ASSERT_TRUE(!ref_res.empty());
    double ref_mover;
    cv::minMaxLoc(ref_res(cv::Rect(170, 170, 20, 20)), nullptr, &ref_mover);
    std::vector<float> rv;
    for (int y = 30; y < 226; y += 12)
        for (int x = 30; x < 226; x += 12)
            if (std::abs(y - 180) > 25) {
                double mx;
                cv::minMaxLoc(ref_res(cv::Rect(x - 3, y - 3, 6, 6)), nullptr, &mx);
                rv.push_back((float)mx);
            }
    const double ref_ratio = ref_mover / (bg_median(rv) + 1.0);

    SeedOutcome out{};
    for (int pl = 0; pl < 2; pl++) {
        motion::DualHomographyParams p;
        p.pipeline = pl == 0 ? motion::FeaturePipeline::small_motion
                             : motion::FeaturePipeline::orb;
        img::Image<float> res = motion::independent_motion_residual(cur, ref_a, ref_b, p);
        ASSERT_TRUE(!res.empty());
        out.mover[pl] = img::window_max(res.view(), 180, 180, 10);
        std::vector<float> ov;
        for (int y = 30; y < 226; y += 12)
            for (int x = 30; x < 226; x += 12)
                if (std::abs(y - 180) > 25)
                    ov.push_back(img::window_max(res.view(), (float)x, (float)y, 3));
        out.ratio_vs_reference[pl] = (out.mover[pl] / (bg_median(ov) + 1.0)) / ref_ratio;
    }
    return out;
}

/// Bounds sit below a 1-60 sweep after the #104 stray fix (small_motion min 0.986x, orb min
/// 0.777x of the reference ratio); before it 17 of seeds 21-60 fell under 0.9x on small_motion.
/// The mover must also clear twice the downstream gate threshold (12).
TEST(component_separation_comparable_to_reference_on_held_out_seeds) {
    std::vector<unsigned> seeds{kLegacySeed};
    for (unsigned s = kFirstHeldOutSeed; s <= kLastHeldOutSeed; s++) seeds.push_back(s);

    const double bounds[2] = {kMinRatioSmallMotionSweepMin0p986, kMinRatioOrbSweepMin0p777};
    int failing = 0;
    for (unsigned seed : seeds) {
        const SeedOutcome o = separation_vs_reference(seed);
        for (int pl = 0; pl < 2; pl++) {
            const bool ok = o.mover[pl] > 24.0f && o.ratio_vs_reference[pl] >= bounds[pl];
            if (!ok) {
                printf("\n    seed %u p%d mover=%.1f ratio_vs_ref=%.3f (need >= %.3f) ",
                       seed, pl, o.mover[pl], o.ratio_vs_reference[pl], bounds[pl]);
                failing++;
            }
        }
    }
    ASSERT_EQ(failing, 0);
}

}

int main() {
    printf("== golden: dual_homography vs OpenCV ==\n");
    return tests_failed > 0 ? 1 : 0;
}
