/// Integration test for analytics/detection_motion_gate.hpp.
///
/// A panning textured scene with (a) an independently-moving target and (b) a static
/// clutter detection that the "detector" also reports every frame. The gate must
/// confirm the MOVER and never the static clutter.
#include "detection_motion_gate.hpp"
#include "test_harness.h"

#include <opencv2/opencv.hpp>
#include <vector>

namespace {

cv::Mat make_bg() {
    cv::Mat bg(256, 256, CV_8U, cv::Scalar(100));
    cv::RNG rng(99);
    for (int i = 0; i < 160; i++) {
        int x = rng.uniform(8, 248), y = rng.uniform(8, 248);
        cv::circle(bg, {x, y}, rng.uniform(2, 5), cv::Scalar(rng.uniform(150, 240)), cv::FILLED);
    }
    cv::GaussianBlur(bg, bg, {5, 5}, 0);
    return bg;
}

// frame at "time" t: background panned diagonally + a target translating horizontally
// (independent of the pan). Returns the frame and the target centre.
cv::Mat frame_at(const cv::Mat &bg, int t, cv::Point &target) {
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 3.0 * t, 0, 1, 2.0 * t), f;
    cv::warpAffine(bg, f, M, bg.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    target = {70 + 6 * t, 180};
    cv::circle(f, target, 6, cv::Scalar(255), cv::FILLED);
    return f;
}

TEST(confirms_independent_mover_not_static_clutter) {
    cv::Mat bg = make_bg();
    nvmm::motion::MovingObjectGate gate;   // default params (min_age 6, min_support 4)

    int confirmed_idx = -2;
    cv::Point tgt, tgt_a, tgt_b;
    for (int t = 2; t <= 16; t++) {
        cv::Mat cur = frame_at(bg, t, tgt);
        cv::Mat ref_a = frame_at(bg, t - 1, tgt_a);
        cv::Mat ref_b = frame_at(bg, t - 2, tgt_b);
        // detector reports the (moving) target AND a fixed static clutter point
        std::vector<nvmm::track::Detection> dets = {
            { (float)tgt.x, (float)tgt.y, 0.9f, false },   // index 0: the independent mover
            { 200.f,        60.f,        0.9f, false },    // index 1: static background clutter
        };
        int r = gate.update(dets, cur, ref_a, ref_b);
        if (r >= 0 && confirmed_idx == -2) confirmed_idx = r;   // record the FIRST confirmation
        ASSERT_TRUE(r != 1);                                    // NEVER confirm the static clutter
    }
    printf("[first_confirm_idx=%d] ", confirmed_idx);
    ASSERT_TRUE(confirmed_idx == 0);     // the independent mover was confirmed
    ASSERT_TRUE(gate.locked());
}

}  // namespace

int main() {
    printf("== analytics/detection_motion_gate ==\n");
    return tests_failed > 0 ? 1 : 0;
}
