/// Golden comparison for analytics/motion_magnify.hpp — the fused IIR pass vs
/// the original cv::Mat-expression implementation, frame by frame over a
/// mixed-frequency sequence.
///
/// Tolerance: 1e-2 absolute after 120 frames. The per-pixel recurrence is
/// identical; drift comes from float rounding-order differences (cv::Mat
/// expressions produce different intermediate temporaries) compounding through
/// the IIR state. With blur enabled the blur tolerance (2e-3/frame) feeds the
/// same recurrence, hence the shared bound.
#include "motion_magnify.hpp"
#include "analytics_scene.h"
#include "golden_util.h"
#include "test_harness.h"

#include <cmath>

namespace {

// the original OpenCV implementation, verbatim, as the oracle
class ReferenceMagnifier {
public:
    explicit ReferenceMagnifier(const nvmm::motion::MagnifyParams &p) : p_(p) {
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

void run_case(int blur) {
    nvmm::motion::MagnifyParams p;
    p.fps = 30.f; p.low_hz = 1.f; p.high_hz = 6.f; p.alpha = 8.f; p.blur = blur;
    nvmm::motion::MotionMagnifier ours(p);
    ReferenceMagnifier ref(p);

    scene::Rng rng(77);
    nvmm::img::Image<uint8_t> base(64, 48);
    for (int y = 0; y < 48; y++)
        for (int x = 0; x < 64; x++) base.at(y, x) = (uint8_t)rng.uniform(30, 220);

    double worst = 0;
    for (int n = 0; n < 120; n++) {
        // global oscillation + a local one at a different frequency
        const float g = 6.f * std::sin(2.f * 3.14159265f * 3.f * n / 30.f);
        const float l = 9.f * std::sin(2.f * 3.14159265f * 0.4f * n / 30.f);
        nvmm::img::Image<uint8_t> f(64, 48);
        for (int y = 0; y < 48; y++)
            for (int x = 0; x < 64; x++)
                f.at(y, x) = scene::clamp_u8((float)base.at(y, x) + g + (x < 32 ? l : 0.f));
        nvmm::img::Image<float> o = ours.process(f);
        cv::Mat r = ref.process(golden::to_cv(f));
        worst = std::max(worst, golden::max_abs_diff(o, r));
    }
    printf("[blur=%d worst|d|=%.2e] ", blur, worst);
    ASSERT_TRUE(worst <= 1e-2);
}

TEST(fused_pass_matches_reference_no_blur) { run_case(0); }
TEST(fused_pass_matches_reference_with_blur) { run_case(5); }

}  // namespace

int main() {
    printf("== golden: motion_magnify vs OpenCV ==\n");
    return tests_failed > 0 ? 1 : 0;
}
