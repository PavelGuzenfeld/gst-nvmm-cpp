#include "detection_motion_gate.hpp"
#include "analytics_scene.h"
#include "test_harness.h"

#include <vector>

namespace {

nvmm::img::Image<uint8_t> frame_at(const nvmm::img::Image<uint8_t> &bg, int t) {
    nvmm::img::Image<uint8_t> f = scene::translate(bg, 3.0 * t, 2.0 * t);
    scene::fill_circle(f, 70 + 6 * t, 180, 6, 255);
    return f;
}

TEST(confirms_independent_mover_not_static_clutter) {
    nvmm::img::Image<uint8_t> bg = scene::textured_bg(256, 99);
    nvmm::motion::MovingObjectGate gate;

    int confirmed_idx = -2;
    for (int t = 2; t <= 16; t++) {
        nvmm::img::Image<uint8_t> cur = frame_at(bg, t);
        nvmm::img::Image<uint8_t> ref_a = frame_at(bg, t - 1);
        nvmm::img::Image<uint8_t> ref_b = frame_at(bg, t - 2);
        std::vector<nvmm::track::Detection> dets = {
            { (float)(70 + 6 * t), 180.f, 0.9f, false },
            { 200.f,               60.f,  0.9f, false },
        };
        const int r = gate.update(dets, cur, ref_a, ref_b);
        if (r >= 0 && confirmed_idx == -2) confirmed_idx = r;
        ASSERT_TRUE(r != 1);
    }
    printf("[first_confirm_idx=%d] ", confirmed_idx);
    ASSERT_TRUE(confirmed_idx == 0);
    ASSERT_TRUE(gate.locked());
}

}

int main() {
    printf("== analytics/detection_motion_gate ==\n");
    return tests_failed > 0 ? 1 : 0;
}
