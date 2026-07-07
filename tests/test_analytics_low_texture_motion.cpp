/// Synthetic unit test for analytics/low_texture_motion.hpp.
///
/// A blob that appears in the low-texture background must light up; an identical blob
/// inside a high-texture patch must be suppressed (the whole point of the mask).
#include "low_texture_motion.hpp"
#include "test_harness.h"

#include <opencv2/opencv.hpp>

namespace {

float wmax(const cv::Mat &m, int cx, int cy, int r) {
    cv::Rect roi(cx - r, cy - r, 2 * r, 2 * r);
    double mx = 0; cv::minMaxLoc(m(roi & cv::Rect(0, 0, m.cols, m.rows)), nullptr, &mx);
    return (float)mx;
}

// low-texture background (flat + faint noise) with one high-texture checkerboard patch.
cv::Mat make_scene() {
    cv::Mat f(256, 256, CV_8U, cv::Scalar(110));
    cv::RNG rng(3); cv::Mat noise(256, 256, CV_8U); rng.fill(noise, cv::RNG::NORMAL, 0, 3);
    f += noise;
    // a genuinely high-texture patch: per-pixel random noise (high gradient everywhere)
    cv::Mat patch(60, 60, CV_8U); rng.fill(patch, cv::RNG::UNIFORM, 0, 256);
    patch.copyTo(f(cv::Rect(30, 30, 60, 60)));
    return f;
}

// The guarantee: frame-diff is kept only where the CURRENT frame is low-texture.
// Apply a uniform difference everywhere and check it survives in the flat background
// but is masked out over the high-texture patch.
TEST(diff_kept_in_low_texture_masked_over_textured_patch) {
    cv::Mat cur = make_scene();
    cv::Mat ref = cur - cv::Scalar(30);              // uniform offset -> |diff| == 30 everywhere

    cv::Mat m = nvmm::motion::low_texture_motion(cur, ref, ref);
    float low_tex = wmax(m, 185, 185, 8);            // flat background -> diff kept
    float high_tex = wmax(m, 60, 60, 8);             // inside the textured patch -> masked
    printf("[low_tex=%.1f high_tex=%.1f] ", low_tex, high_tex);

    ASSERT_TRUE(low_tex > 20.0f);                    // ~30 diff survives over low texture
    ASSERT_TRUE(high_tex < 5.0f);                    // suppressed where the frame is textured
}

}  // namespace

int main() {
    printf("== analytics/low_texture_motion ==\n");
    return tests_failed > 0 ? 1 : 0;
}
