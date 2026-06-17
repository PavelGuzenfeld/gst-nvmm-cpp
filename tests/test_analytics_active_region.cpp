/// Synthetic unit test for analytics/active_region.hpp.
#include "active_region.hpp"
#include "test_harness.h"

#include <opencv2/opencv.hpp>

namespace {

// dark-but-textured content (IR-like) between two UNIFORM (bright) side bars.
// The range-based detector must keep the dark content and trim the bright bars —
// a mean/brightness threshold would do the opposite.
TEST(trims_uniform_bars_keeps_dark_textured_content) {
    cv::Mat f(200, 200, CV_8U, cv::Scalar(128));   // uniform mid-gray everywhere (the "bars")
    cv::RNG rng(7);
    for (int i = 0; i < 120; i++) {                 // dark textured content in x[40,160)
        int x = rng.uniform(42, 158), y = rng.uniform(4, 196);
        cv::circle(f, {x, y}, rng.uniform(2, 4), cv::Scalar(rng.uniform(10, 60)), cv::FILLED);
    }
    cv::GaussianBlur(f, f, {3, 3}, 0);

    cv::Rect r = nvmm::video::active_region(f);
    printf("[x=%d w=%d] ", r.x, r.width);
    ASSERT_TRUE(r.x >= 38 && r.x <= 44);            // left bar trimmed to ~x=40
    ASSERT_TRUE(r.x + r.width >= 156 && r.x + r.width <= 162);  // right bar trimmed to ~x=160
}

TEST(uniform_frame_returns_full) {
    cv::Mat f(120, 120, CV_8U, cv::Scalar(50));     // nothing but a bar -> degenerate, return full
    cv::Rect r = nvmm::video::active_region(f);
    ASSERT_TRUE(r.width == f.cols && r.height == f.rows);
}

}  // namespace

int main() {
    printf("== analytics/active_region ==\n");
    return tests_failed > 0 ? 1 : 0;
}
