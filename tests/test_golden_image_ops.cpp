/// Golden comparisons for analytics/image_ops.hpp + the fused Sobel stage —
/// OpenCV is the reference oracle.
///
/// Tolerances (documented per stage):
///  - gaussian_kernel: 1e-6 vs cv::getGaussianKernel — same fixed tables /
///    sigma formula, only float-vs-double rounding differs.
///  - gaussian_blur (float): 2e-3 absolute on inputs up to ~1442 (the Sobel
///    magnitude ceiling). Same separable kernel and REFLECT_101 border; the
///    difference is accumulation order (row-then-col float accumulate vs
///    OpenCV's SIMD filter engine).
///  - sobel_magnitude: 2e-3 absolute — integer taps are exact, both sides take
///    a float sqrt; differences come from cv::magnitude's vectorised sqrt.
#include "low_texture_motion.hpp"
#include "analytics_scene.h"
#include "golden_util.h"
#include "test_harness.h"

#include <cmath>
#include <vector>

namespace {

nvmm::img::Image<uint8_t> random_frame(int w, int h, unsigned seed) {
    scene::Rng rng(seed);
    nvmm::img::Image<uint8_t> f(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) f.at(y, x) = (uint8_t)rng.uniform(0, 256);
    // a few smooth structures so it isn't only white noise
    for (int i = 0; i < 20; i++)
        scene::fill_circle(f, rng.uniform(10, w - 10), rng.uniform(10, h - 10),
                           rng.uniform(3, 9), (uint8_t)rng.uniform(0, 256));
    return f;
}

TEST(gaussian_kernel_matches_getGaussianKernel) {
    for (int k : {1, 3, 5, 7, 9, 15, 31}) {
        const std::vector<float> ours = nvmm::img::gaussian_kernel(k);
        cv::Mat ref = cv::getGaussianKernel(k, 0, CV_64F);
        for (int i = 0; i < k; i++)
            ASSERT_NEAR(ours[(size_t)i], ref.at<double>(i), 1e-6);
    }
}

TEST(gaussian_blur_matches_GaussianBlur_float) {
    nvmm::img::Image<uint8_t> u8 = random_frame(157, 121, 11);   // odd sizes on purpose
    nvmm::img::Image<float> in(u8.width(), u8.height());
    for (int y = 0; y < u8.height(); y++)
        for (int x = 0; x < u8.width(); x++) in.at(y, x) = 5.5f * u8.at(y, x);
    cv::Mat cv_in = golden::to_cv(in);

    for (int k : {3, 5, 7, 9, 15}) {
        nvmm::img::Image<float> ours(in.width(), in.height()), tmp;
        nvmm::img::gaussian_blur<float>(in.view(), tmp, ours.view(), k);
        cv::Mat ref;
        cv::GaussianBlur(cv_in, ref, cv::Size(k, k), 0);
        const double d = golden::max_abs_diff(ours, ref);
        printf("[k=%d max|d|=%.2e] ", k, d);
        ASSERT_TRUE(d <= 2e-3);
    }
}

TEST(sobel_magnitude_matches_cv_chain) {
    nvmm::img::Image<uint8_t> f = random_frame(200, 150, 23);
    nvmm::img::Image<float> ours(200, 150);
    nvmm::motion::detail::sobel_magnitude(f.view(), ours.view());

    cv::Mat cf = golden::to_cv(f), gx, gy, ref;
    cv::Sobel(cf, gx, CV_32F, 1, 0);
    cv::Sobel(cf, gy, CV_32F, 0, 1);
    cv::magnitude(gx, gy, ref);
    const double d = golden::max_abs_diff(ours, ref);
    printf("[max|d|=%.2e] ", d);
    ASSERT_TRUE(d <= 2e-3);
}

TEST(window_max_matches_minMaxLoc) {
    scene::Rng rng(5);
    nvmm::img::Image<float> m(97, 83);
    for (int y = 0; y < 83; y++)
        for (int x = 0; x < 97; x++) m.at(y, x) = rng.gauss(40.f);
    cv::Mat cm = golden::to_cv(m);
    for (int i = 0; i < 200; i++) {
        const int cx = rng.uniform(-5, 102), cy = rng.uniform(-5, 88), r = rng.uniform(1, 9);
        const int x0 = std::max(0, cx - r), y0 = std::max(0, cy - r);
        const int x1 = std::min(97, cx + r), y1 = std::min(83, cy + r);
        const float ours = nvmm::img::window_max(m.view(), (float)cx, (float)cy, r);
        if (x1 <= x0 || y1 <= y0) { ASSERT_NEAR(ours, 0.f, 0.f); continue; }
        double mx;
        cv::minMaxLoc(cm(cv::Rect(x0, y0, x1 - x0, y1 - y0)), nullptr, &mx);
        ASSERT_NEAR(ours, mx, 0.0);
    }
}

}  // namespace

int main() {
    printf("== golden: image_ops vs OpenCV ==\n");
    return tests_failed > 0 ? 1 : 0;
}
