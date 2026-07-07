/// Benchmark: hand-rolled (fused) analytics vs the OpenCV reference chains.
/// Output: CSV to stdout (same convention as bench_nvmm), one row per
/// (component, implementation, frame size). avg over `iters` runs after one
/// warm-up. Lives behind -Danalytics_golden (needs OpenCV as the baseline).
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

#include "active_region.hpp"
#include "bench_scene.h"
#include "dual_homography.hpp"
#include "golden_util.h"
#include "low_texture_motion.hpp"
#include "motion_magnify.hpp"

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::micro>;

namespace {

template <typename Fn>
void bench(const char *component, const char *impl, int w, int h, int iters, Fn &&fn)
{
    fn();  // warm-up
    double total = 0, mn = 1e12, mx = 0;
    for (int i = 0; i < iters; i++) {
        const auto t0 = Clock::now();
        fn();
        const double us = Duration(Clock::now() - t0).count();
        total += us;
        if (us < mn) mn = us;
        if (us > mx) mx = us;
    }
    printf("%s_%s_%dx%d,%d,%.2f,%.2f,%.2f,%.2f\n", component, impl, w, h, iters,
           total, total / iters, mn, mx);
}

// original OpenCV chains (the pre-port implementations), as the baselines
cv::Rect cv_active_region(const cv::Mat &gray)
{
    cv::Mat cmin, cmax, rmin, rmax;
    cv::reduce(gray, cmax, 0, cv::REDUCE_MAX);
    cv::reduce(gray, cmin, 0, cv::REDUCE_MIN);
    cv::reduce(gray, rmax, 1, cv::REDUCE_MAX);
    cv::reduce(gray, rmin, 1, cv::REDUCE_MIN);
    auto crange = [&](int i){ return (int)cmax.at<uchar>(i) - (int)cmin.at<uchar>(i); };
    auto rrange = [&](int i){ return (int)rmax.at<uchar>(i) - (int)rmin.at<uchar>(i); };
    int x0 = 0;               while (x0 < gray.cols - 1 && crange(x0) < 15) x0++;
    int x1 = gray.cols - 1;   while (x1 > x0            && crange(x1) < 15) x1--;
    int y0 = 0;               while (y0 < gray.rows - 1 && rrange(y0) < 15) y0++;
    int y1 = gray.rows - 1;   while (y1 > y0            && rrange(y1) < 15) y1--;
    if (x1 <= x0 || y1 <= y0) return cv::Rect(0, 0, gray.cols, gray.rows);
    return cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
}

cv::Mat cv_low_texture_motion(const cv::Mat &cur, const cv::Mat &ref_a, const cv::Mat &ref_b,
                              const nvmm::motion::LowTextureMotionParams &p)
{
    cv::Mat gx, gy, grad, gblur, low;
    cv::Sobel(cur, gx, CV_32F, 1, 0);
    cv::Sobel(cur, gy, CV_32F, 0, 1);
    cv::magnitude(gx, gy, grad);
    cv::GaussianBlur(grad, gblur, cv::Size(p.grad_blur, p.grad_blur), 0);
    low = (gblur < p.grad_thresh);
    cv::morphologyEx(low, low, cv::MORPH_CLOSE, cv::Mat::ones(p.close_k, p.close_k, CV_8U));
    cv::Mat da, db, dmin, out;
    cv::absdiff(cur, ref_a, da);
    cv::absdiff(cur, ref_b, db);
    dmin = cv::min(da, db);
    dmin.convertTo(dmin, CV_32F);
    out = cv::Mat::zeros(cur.size(), CV_32F);
    dmin.copyTo(out, low);
    if (p.diff_blur > 0) cv::GaussianBlur(out, out, cv::Size(p.diff_blur, p.diff_blur), 0);
    return out;
}

class CvMagnifier {
public:
    explicit CvMagnifier(const nvmm::motion::MagnifyParams &p) : p_(p) {
        r_low_  = 1.f - std::exp(-2.f * (float)CV_PI * p_.low_hz  / p_.fps);
        r_high_ = 1.f - std::exp(-2.f * (float)CV_PI * p_.high_hz / p_.fps);
    }
    cv::Mat process(const cv::Mat &frame) {
        cv::Mat x;
        frame.convertTo(x, CV_32F);
        if (p_.blur > 0) cv::GaussianBlur(x, x, cv::Size(p_.blur, p_.blur), 0);
        if (!init_) { lp_low_ = x.clone(); lp_high_ = x.clone(); init_ = true; return x; }
        lp_low_  += r_low_  * (x - lp_low_);
        lp_high_ += r_high_ * (x - lp_high_);
        cv::Mat band = lp_high_ - lp_low_;
        return x + p_.alpha * band;
    }

private:
    nvmm::motion::MagnifyParams p_;
    float r_low_ = 0.f, r_high_ = 0.f;
    cv::Mat lp_low_, lp_high_;
    bool init_ = false;
};

cv::Mat cv_orb_residual(const cv::Mat &cur, const cv::Mat &ref_a, const cv::Mat &ref_b)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(4000);
    cv::BFMatcher bf(cv::NORM_HAMMING);
    auto two_planes = [&](const cv::Mat &a, const cv::Mat &b, cv::Mat &H1, cv::Mat &H2) {
        H1.release(); H2.release();
        std::vector<cv::KeyPoint> k1, k2; cv::Mat d1, d2;
        orb->detectAndCompute(a, cv::noArray(), k1, d1);
        orb->detectAndCompute(b, cv::noArray(), k2, d2);
        if (d1.empty() || d2.empty()) return;
        std::vector<std::vector<cv::DMatch>> knn;
        bf.knnMatch(d1, d2, knn, 2);
        std::vector<cv::Point2f> p1, p2;
        for (auto &m : knn)
            if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance) {
                p1.push_back(k1[m[0].queryIdx].pt);
                p2.push_back(k2[m[0].trainIdx].pt);
            }
        if ((int)p1.size() < 20) return;
        cv::Mat mask;
        H1 = cv::findHomography(p1, p2, cv::RANSAC, 3.0, mask);
        std::vector<cv::Point2f> o1, o2;
        for (int i = 0; i < mask.rows; i++)
            if (!mask.at<uchar>(i)) { o1.push_back(p1[i]); o2.push_back(p2[i]); }
        if ((int)o1.size() >= 20) H2 = cv::findHomography(o1, o2, cv::RANSAC, 3.0);
    };
    auto residual = [&](const cv::Mat &a, const cv::Mat &nb, const cv::Mat &H) {
        if (H.empty()) return cv::Mat();
        cv::Mat Hinv = H.inv(), wp, dd;
        cv::warpPerspective(nb, wp, Hinv, a.size());
        cv::absdiff(a, wp, dd); dd.convertTo(dd, CV_32F);
        cv::Mat ones = cv::Mat::ones(a.size(), CV_32F), vw, vm;
        cv::warpPerspective(ones, vw, Hinv, a.size());
        cv::erode((vw > 0.5f), vm, cv::Mat::ones(15, 15, CV_8U));
        cv::Mat out = cv::Mat::zeros(a.size(), CV_32F);
        dd.copyTo(out, vm);
        return out;
    };
    cv::Mat Ha1, Ha2, Hb1, Hb2;
    two_planes(cur, ref_a, Ha1, Ha2);
    two_planes(cur, ref_b, Hb1, Hb2);
    cv::Mat r1 = residual(cur, ref_a, Ha1), r2 = residual(cur, ref_b, Hb1);
    cv::Mat sc = r1.empty() ? r2 : r2.empty() ? r1 : cv::min(r1, r2);
    if (!sc.empty()) cv::GaussianBlur(sc, sc, cv::Size(3, 3), 0);
    return sc;
}

nvmm::img::Image<uint8_t> shift(const nvmm::img::Image<uint8_t> &src, int dx, int dy)
{
    nvmm::img::Image<uint8_t> out(src.width(), src.height(), 100);
    for (int y = 0; y < src.height(); y++) {
        const int sy = y - dy;
        if (sy < 0 || sy >= src.height()) continue;
        for (int x = 0; x < src.width(); x++) {
            const int sx = x - dx;
            if (sx >= 0 && sx < src.width()) out.at(y, x) = src.at(sy, sx);
        }
    }
    return out;
}

}  // namespace

int main()
{
    printf("benchmark,iterations,total_us,avg_us,min_us,max_us\n");

    const int sizes[][2] = {{640, 360}, {1280, 720}, {1920, 1080}};
    for (const auto &sz : sizes) {
        const int w = sz[0], h = sz[1];
        const int iters = w >= 1920 ? 20 : 50;

        nvmm::img::Image<uint8_t> frame = bench_scene::textured(w, h, 5);
        nvmm::img::Image<uint8_t> ref_a = shift(frame, 6, 4);
        nvmm::img::Image<uint8_t> ref_b = shift(frame, 12, 8);
        cv::Mat cv_frame = golden::to_cv(frame), cv_ra = golden::to_cv(ref_a),
                cv_rb = golden::to_cv(ref_b);

        // active_region
        bench("active_region", "fused", w, h, iters,
              [&] { (void)nvmm::video::active_region(frame); });
        bench("active_region", "opencv", w, h, iters,
              [&] { (void)cv_active_region(cv_frame); });

        // low_texture_motion
        nvmm::motion::LowTextureMotionParams ltp;
        bench("low_texture_motion", "fused", w, h, iters,
              [&] { (void)nvmm::motion::low_texture_motion(frame, ref_a, ref_b, ltp); });
        bench("low_texture_motion", "opencv", w, h, iters,
              [&] { (void)cv_low_texture_motion(cv_frame, cv_ra, cv_rb, ltp); });

        // motion_magnify (blurless + blurred)
        for (int blur : {0, 5}) {
            nvmm::motion::MagnifyParams mp;
            mp.blur = blur;
            nvmm::motion::MotionMagnifier ours(mp);
            CvMagnifier cvm(mp);
            bench("motion_magnify", blur ? "fused_blur5" : "fused", w, h, iters,
                  [&] { (void)ours.process(frame.view()); });
            bench("motion_magnify", blur ? "opencv_blur5" : "opencv", w, h, iters,
                  [&] { (void)cvm.process(cv_frame); });
        }

        // dual_homography (fewer iters — heavy)
        const int hiters = w >= 1920 ? 5 : 10;
        for (int pl = 0; pl < 2; pl++) {
            nvmm::motion::DualHomographyParams dp;
            dp.pipeline = pl == 0 ? nvmm::motion::FeaturePipeline::small_motion
                                  : nvmm::motion::FeaturePipeline::orb;
            bench("dual_homography", pl == 0 ? "fused_small_motion" : "fused_orb", w, h,
                  hiters,
                  [&] { (void)nvmm::motion::independent_motion_residual(frame, ref_a, ref_b, dp); });
        }
        bench("dual_homography", "opencv_orb", w, h, hiters,
              [&] { (void)cv_orb_residual(cv_frame, cv_ra, cv_rb); });
    }
    return 0;
}
