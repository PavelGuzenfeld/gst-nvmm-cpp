/// Integration test for analytics/detection_motion_gate.hpp.
///
/// A panning textured scene with (a) an independently-moving target and (b) a static
/// clutter detection that the "detector" also reports every frame. The gate must
/// confirm the MOVER and never the static clutter.
#include "detection_motion_gate.hpp"
#include "analytics_scene.h"
#include "test_harness.h"

#include <vector>

namespace {

// frame at "time" t: background panned diagonally + a target translating horizontally
// (independent of the pan). Returns the frame; the target centre is (70 + 6t, 180).
nvmm::img::Image<uint8_t> frame_at(const nvmm::img::Image<uint8_t> &bg, int t) {
    nvmm::img::Image<uint8_t> f = scene::translate(bg, 3.0 * t, 2.0 * t);
    scene::fill_circle(f, 70 + 6 * t, 180, 6, 255);
    return f;
}

TEST(confirms_independent_mover_not_static_clutter) {
    nvmm::img::Image<uint8_t> bg = scene::textured_bg(256, 99);
    nvmm::motion::MovingObjectGate gate;   // default params (min_age 6, min_support 4)

    int confirmed_idx = -2;
    for (int t = 2; t <= 16; t++) {
        nvmm::img::Image<uint8_t> cur = frame_at(bg, t);
        nvmm::img::Image<uint8_t> ref_a = frame_at(bg, t - 1);
        nvmm::img::Image<uint8_t> ref_b = frame_at(bg, t - 2);
        // detector reports the (moving) target AND a fixed static clutter point
        std::vector<nvmm::track::Detection> dets = {
            { (float)(70 + 6 * t), 180.f, 0.9f, false },   // index 0: the independent mover
            { 200.f,               60.f,  0.9f, false },   // index 1: static background clutter
        };
        const int r = gate.update(dets, cur, ref_a, ref_b);
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
