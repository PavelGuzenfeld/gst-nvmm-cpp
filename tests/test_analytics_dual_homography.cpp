/// Synthetic unit test for analytics/dual_homography.hpp.
///
/// Build a textured background, simulate a camera pan (a global translation) into
/// two reference frames, and add a blob that moves INDEPENDENTLY of that pan. The
/// dual-homography residual must peak on the independent mover and stay low on the
/// (pan-explained) static background.
#include "dual_homography.hpp"
#include "test_harness.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// Textured but BAND-LIMITED background: soft blobs + a light blur, so ORB still
// finds features but there are no 1px-sharp edges (which would leave warp
// re-interpolation residual on the static background — real footage is naturally
// band-limited the same way).
cv::Mat make_bg() {
    cv::Mat bg(256, 256, CV_8U, cv::Scalar(100));
    cv::RNG rng(12345);
    for (int i = 0; i < 160; i++) {
        int x = rng.uniform(8, 248), y = rng.uniform(8, 248);
        cv::circle(bg, {x, y}, rng.uniform(2, 5), cv::Scalar(rng.uniform(150, 240)), cv::FILLED);
    }
    cv::GaussianBlur(bg, bg, {5, 5}, 0);
    return bg;
}

cv::Mat translate(const cv::Mat &src, double dx, double dy) {
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy), out;
    cv::warpAffine(src, out, M, src.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return out;
}

void blob(cv::Mat &img, int cx, int cy) {
    cv::circle(img, {cx, cy}, 6, cv::Scalar(255), cv::FILLED);
}

float window_max(const cv::Mat &m, int cx, int cy, int r) {
    int x0 = std::max(0, cx - r), y0 = std::max(0, cy - r);
    int x1 = std::min(m.cols, cx + r), y1 = std::min(m.rows, cy + r);
    double mx = 0; cv::minMaxLoc(m(cv::Rect(x0, y0, x1 - x0, y1 - y0)), nullptr, &mx);
    return (float)mx;
}

// median local-max over a grid of STATIC background points (away from the mover's
// horizontal band + the borders) — robust to a single high-residual pixel.
float static_bg_median(const cv::Mat &m) {
    std::vector<float> v;
    for (int y = 30; y < 226; y += 12)
        for (int x = 30; x < 226; x += 12)
            if (std::abs(y - 180) > 25) v.push_back(window_max(m, x, y, 3));
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.f : v[v.size() / 2];
}

TEST(independent_mover_peaks_above_static_background) {
    cv::Mat bg = make_bg();
    // camera pan: refs are the background shifted (two different "past" offsets)
    cv::Mat cur = bg.clone();
    cv::Mat ref_a = translate(bg, 6, 4);
    cv::Mat ref_b = translate(bg, 12, 8);
    // a blob moving independently of the pan (different position in each frame)
    blob(cur,   180, 180);
    blob(ref_a, 150, 180);
    blob(ref_b, 120, 180);

    cv::Mat res = nvmm::motion::independent_motion_residual(cur, ref_a, ref_b);
    ASSERT_TRUE(!res.empty());                       // a homography was fit

    float mover = window_max(res, 180, 180, 10);     // residual on the independent mover
    float bg_median = static_bg_median(res);          // typical pan-explained background residual
    printf("[mover=%.1f bg_median=%.1f] ", mover, bg_median);

    ASSERT_TRUE(mover > 60.0f);                       // mover clearly lights up
    ASSERT_TRUE(mover > 5.0f * (bg_median + 1.0f));   // and dominates the static background
}

TEST(textureless_input_returns_empty) {
    cv::Mat flat(256, 256, CV_8U, cv::Scalar(120));   // no features -> no homography
    cv::Mat res = nvmm::motion::independent_motion_residual(flat, flat, flat);
    ASSERT_TRUE(res.empty());
}

}  // namespace

int main() {
    printf("== analytics/dual_homography ==\n");
    return tests_failed > 0 ? 1 : 0;
}
