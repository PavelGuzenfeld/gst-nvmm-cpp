#include "active_region.hpp"
#include "analytics_scene.h"
#include "golden_util.h"
#include "test_harness.h"

namespace {

cv::Rect reference_active_region(const cv::Mat &gray, int bar_range)
{
    cv::Mat cmin, cmax, rmin, rmax;
    cv::reduce(gray, cmax, 0, cv::REDUCE_MAX);
    cv::reduce(gray, cmin, 0, cv::REDUCE_MIN);
    cv::reduce(gray, rmax, 1, cv::REDUCE_MAX);
    cv::reduce(gray, rmin, 1, cv::REDUCE_MIN);
    auto crange = [&](int i){ return (int)cmax.at<uchar>(i) - (int)cmin.at<uchar>(i); };
    auto rrange = [&](int i){ return (int)rmax.at<uchar>(i) - (int)rmin.at<uchar>(i); };
    int x0 = 0;               while (x0 < gray.cols - 1 && crange(x0) < bar_range) x0++;
    int x1 = gray.cols - 1;   while (x1 > x0            && crange(x1) < bar_range) x1--;
    int y0 = 0;               while (y0 < gray.rows - 1 && rrange(y0) < bar_range) y0++;
    int y1 = gray.rows - 1;   while (y1 > y0            && rrange(y1) < bar_range) y1--;
    if (x1 <= x0 || y1 <= y0) return cv::Rect(0, 0, gray.cols, gray.rows);
    return cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
}

/// Both sides do integer min/max, so the rectangle must match exactly.
TEST(rect_matches_reference_exactly) {
    scene::Rng rng(31);
    for (int trial = 0; trial < 40; trial++) {
        const int w = rng.uniform(40, 300), h = rng.uniform(40, 300);
        nvmm::img::Image<uint8_t> f(w, h, (uint8_t)rng.uniform(0, 256));
        const int bx = rng.uniform(0, w / 2), bw = rng.uniform(1, w - bx);
        const int by = rng.uniform(0, h / 2), bh = rng.uniform(1, h - by);
        for (int y = by; y < by + bh; y++)
            for (int x = bx; x < bx + bw; x++)
                f.at(y, x) = (uint8_t)rng.uniform(0, 256);

        nvmm::video::ActiveRegionParams p;
        p.bar_range = rng.uniform(2, 60);
        const nvmm::img::Rect ours = nvmm::video::active_region(f, p);
        const cv::Rect ref = reference_active_region(golden::to_cv(f), p.bar_range);
        ASSERT_EQ(ours.x, ref.x);
        ASSERT_EQ(ours.y, ref.y);
        ASSERT_EQ(ours.w, ref.width);
        ASSERT_EQ(ours.h, ref.height);
    }
}

}

int main() {
    printf("== golden: active_region vs OpenCV ==\n");
    return tests_failed > 0 ? 1 : 0;
}
