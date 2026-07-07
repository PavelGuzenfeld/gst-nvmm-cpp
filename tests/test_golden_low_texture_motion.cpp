/// Golden comparison for analytics/low_texture_motion.hpp vs the original
/// OpenCV op chain (inlined below as the oracle).
///
/// Metrics (documented per stage):
///  - mask stage (blur -> threshold -> morphological close): DISAGREEMENT RATE,
///    not value diff — the threshold is a hard nonlinearity, so a ~1e-3 float
///    difference in the blurred gradient near grad_thresh legitimately flips a
///    pixel. Bound: <= 0.2% of pixels. The box-sum close itself is exact on a
///    binary mask, so all disagreement stems from near-threshold flips.
///  - full component: value tolerance 0.05 wherever both sides KEPT the pixel,
///    plus <= 0.5% of pixels allowed to differ by more (mask-edge flips fed
///    through the output blur).
#include "low_texture_motion.hpp"
#include "analytics_scene.h"
#include "golden_util.h"
#include "test_harness.h"

#include <cmath>

namespace {

using nvmm::motion::LowTextureMotionParams;

// original OpenCV implementation, verbatim, as the oracle
cv::Mat reference_mask(const cv::Mat &cur, const LowTextureMotionParams &p)
{
    cv::Mat gx, gy, grad, gblur, low;
    cv::Sobel(cur, gx, CV_32F, 1, 0);
    cv::Sobel(cur, gy, CV_32F, 0, 1);
    cv::magnitude(gx, gy, grad);
    cv::GaussianBlur(grad, gblur, cv::Size(p.grad_blur, p.grad_blur), 0);
    low = (gblur < p.grad_thresh);
    cv::morphologyEx(low, low, cv::MORPH_CLOSE, cv::Mat::ones(p.close_k, p.close_k, CV_8U));
    return low;
}

cv::Mat reference_low_texture_motion(const cv::Mat &cur, const cv::Mat &ref_a,
                                     const cv::Mat &ref_b, const LowTextureMotionParams &p)
{
    cv::Mat low = reference_mask(cur, p);
    cv::Mat da, db, dmin, out;
    cv::absdiff(cur, ref_a, da);
    cv::absdiff(cur, ref_b, db);
    dmin = cv::min(da, db);
    dmin.convertTo(dmin, CV_32F);
    out = cv::Mat::zeros(cur.size(), CV_32F);
    dmin.copyTo(out, low);
    if (p.diff_blur > 0) cv::GaussianBlur(out, out, cv::Size(p.diff_blur, p.diff_blur), 0);
    if (p.border > 0) {
        out.rowRange(0, std::min(p.border, out.rows)) = 0;
        out.rowRange(std::max(0, out.rows - p.border), out.rows) = 0;
        out.colRange(0, std::min(p.border, out.cols)) = 0;
        out.colRange(std::max(0, out.cols - p.border), out.cols) = 0;
    }
    return out;
}

// mixed scene: flat regions, noise patch, smooth blobs — plenty of mask boundary
nvmm::img::Image<uint8_t> make_scene(unsigned seed) {
    scene::Rng rng(seed);
    nvmm::img::Image<uint8_t> f(256, 256);
    for (int y = 0; y < 256; y++)
        for (int x = 0; x < 256; x++) f.at(y, x) = scene::clamp_u8(110.f + rng.gauss(3.f));
    for (int y = 30; y < 90; y++)
        for (int x = 30; x < 90; x++) f.at(y, x) = (uint8_t)rng.uniform(0, 256);
    for (int i = 0; i < 40; i++)
        scene::fill_circle(f, rng.uniform(8, 248), rng.uniform(8, 248),
                           rng.uniform(2, 6), (uint8_t)rng.uniform(60, 200));
    return f;
}

TEST(mask_stage_disagreement_below_bound) {
    for (unsigned seed : {3u, 17u, 99u}) {
        nvmm::img::Image<uint8_t> cur = make_scene(seed);
        LowTextureMotionParams p;
        nvmm::img::Image<uint8_t> ours = nvmm::motion::detail::low_texture_mask(cur, p);
        cv::Mat ref = reference_mask(golden::to_cv(cur), p);
        const double frac = golden::mask_disagree_frac(ours, ref);
        printf("[seed=%u disagree=%.4f%%] ", seed, 100.0 * frac);
        ASSERT_TRUE(frac <= 0.002);
    }
}

TEST(component_matches_reference) {
    nvmm::img::Image<uint8_t> cur = make_scene(7);
    scene::Rng rng(1234);
    nvmm::img::Image<uint8_t> ra(256, 256), rb(256, 256);
    for (int y = 0; y < 256; y++)
        for (int x = 0; x < 256; x++) {
            ra.at(y, x) = scene::clamp_u8((float)cur.at(y, x) - 25.f + rng.gauss(2.f));
            rb.at(y, x) = scene::clamp_u8((float)cur.at(y, x) - 10.f + rng.gauss(2.f));
        }

    LowTextureMotionParams p;
    nvmm::img::Image<float> ours = nvmm::motion::low_texture_motion(cur, ra, rb, p);
    cv::Mat ref = reference_low_texture_motion(golden::to_cv(cur), golden::to_cv(ra),
                                               golden::to_cv(rb), p);
    long over = 0;
    double worst_ok = 0;
    for (int y = 0; y < 256; y++)
        for (int x = 0; x < 256; x++) {
            const double d = std::fabs((double)ours.at(y, x) - ref.at<float>(y, x));
            if (d > 0.5) over++;
            else worst_ok = std::max(worst_ok, d);
        }
    const double frac = (double)over / (256.0 * 256.0);
    printf("[>0.5 diff: %.4f%%, worst elsewhere %.2e] ", 100.0 * frac, worst_ok);
    ASSERT_TRUE(frac <= 0.005);      // mask-edge flips only
    ASSERT_TRUE(worst_ok <= 0.05);   // agreement wherever the masks agree
}

}  // namespace

int main() {
    printf("== golden: low_texture_motion vs OpenCV ==\n");
    return tests_failed > 0 ? 1 : 0;
}
