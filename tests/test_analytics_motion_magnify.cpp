/// Synthetic unit test for analytics/motion_magnify.hpp.
///
/// Drive a frame whose intensity oscillates at a known frequency and measure the
/// output peak-to-peak vs the input. An IN-BAND oscillation must be amplified; an
/// OUT-OF-BAND one must pass through ~unchanged.
#include "motion_magnify.hpp"
#include "test_harness.h"

#include <opencv2/opencv.hpp>
#include <cmath>

namespace {

// Run a uniform frame oscillating at f0 Hz through the magnifier; return the
// steady-state output peak-to-peak amplitude (input amplitude is fixed at 20 -> pp 40).
float steady_pp(float f0, float low, float high, float alpha) {
    nvmm::motion::MagnifyParams p; p.fps = 30.f; p.low_hz = low; p.high_hz = high; p.alpha = alpha;
    nvmm::motion::MotionMagnifier mag(p);
    const int N = 150, settle = 100;
    float lo = 1e9f, hi = -1e9f;
    for (int n = 0; n < N; n++) {
        float v = 128.f + 20.f * std::sin(2.f * (float)CV_PI * f0 * n / p.fps);
        cv::Mat f(16, 16, CV_8U, cv::Scalar(cv::saturate_cast<uchar>(v)));
        cv::Mat out = mag.process(f);
        float c = out.at<float>(8, 8);
        if (n >= settle) { lo = std::min(lo, c); hi = std::max(hi, c); }
    }
    return hi - lo;
}

TEST(in_band_oscillation_is_amplified) {
    float pp = steady_pp(/*f0=*/4.f, /*low=*/2.f, /*high=*/8.f, /*alpha=*/10.f);
    printf("[in-band pp=%.1f vs input 40] ", pp);
    ASSERT_TRUE(pp > 80.0f);   // clearly amplified beyond the input pp of ~40
}

TEST(out_of_band_oscillation_is_not_amplified) {
    float pp = steady_pp(/*f0=*/0.2f, /*low=*/2.f, /*high=*/8.f, /*alpha=*/10.f);
    printf("[out-of-band pp=%.1f vs input 40] ", pp);
    ASSERT_TRUE(pp < 60.0f);   // a slow (sub-band) oscillation passes ~through (input pp ~40, clamped)
}

}  // namespace

int main() {
    printf("== analytics/motion_magnify ==\n");
    return tests_failed > 0 ? 1 : 0;
}
