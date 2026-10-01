#include "nvmm_motion.hpp"
#include "test_harness.h"

#include <cmath>
#include <vector>

using nvmm::MotionEntry;

namespace {
constexpr int GRID = 4, MVW = 8, MVH = 8;
constexpr int FW = MVW * GRID, FH = MVH * GRID;

std::vector<int16_t> half_moving_field() {
    std::vector<int16_t> f(MVW * MVH * 2, 0);
    for (int y = 0; y < MVH; y++)
        for (int x = MVW / 2; x < MVW; x++)
            f[(y * MVW + x) * 2 + 0] = 64;
    return f;
}

NvmmDetObject box(float x, float y, float w, float h) {
    NvmmDetObject o{};
    o.left = x; o.top = y; o.width = w; o.height = h;
    return o;
}
}

TEST(static_vs_moving_box) {
    auto f = half_moving_field();
    NvmmDetObject objs[2] = { box(0, 0, 12, 12), box(20, 0, 12, 12) };
    MotionEntry out[2];
    ASSERT_EQ(nvmm::compute_box_motion(f.data(), MVW, MVH, GRID, FW, FH,
                                       objs, 2, 1.0f, out), 2u);
    ASSERT_NEAR(out[0].mean_px, 0.0f, 1e-4);
    ASSERT_EQ(out[0].moving, 0u);
    ASSERT_NEAR(out[1].mean_px, 2.0f, 1e-4);
    ASSERT_EQ(out[1].moving, 1u);
}

TEST(magnitude_is_euclidean) {
    std::vector<int16_t> f(MVW * MVH * 2, 0);
    f[0] = 96; f[1] = 128;
    NvmmDetObject o = box(0, 0, GRID, GRID);
    MotionEntry e;
    nvmm::compute_box_motion(f.data(), MVW, MVH, GRID, FW, FH, &o, 1, 1.0f, &e);
    ASSERT_NEAR(e.mean_px, 5.0f, 1e-4);
    ASSERT_EQ(e.moving, 1u);
}

TEST(straddling_box_averages) {
    auto f = half_moving_field();
    NvmmDetObject o = box(8, 8, 16, 8);
    MotionEntry e;
    nvmm::compute_box_motion(f.data(), MVW, MVH, GRID, FW, FH, &o, 1, 0.5f, &e);
    ASSERT_NEAR(e.mean_px, 1.0f, 1e-4);
    ASSERT_EQ(e.moving, 1u);
}

TEST(out_of_frame_clamped) {
    auto f = half_moving_field();
    NvmmDetObject o = box(-10, -10, 14, 14);
    MotionEntry e;
    ASSERT_EQ(nvmm::compute_box_motion(f.data(), MVW, MVH, GRID, FW, FH,
                                       &o, 1, 1.0f, &e), 1u);
    ASSERT_NEAR(e.mean_px, 0.0f, 1e-4);
}

TEST(tiny_box_reads_a_cell) {
    auto f = half_moving_field();
    NvmmDetObject o = box(29, 29, 1, 1);
    MotionEntry e;
    nvmm::compute_box_motion(f.data(), MVW, MVH, GRID, FW, FH, &o, 1, 1.0f, &e);
    ASSERT_NEAR(e.mean_px, 2.0f, 1e-4);
    ASSERT_EQ(e.moving, 1u);
}

TEST(zero_objects) {
    auto f = half_moving_field();
    ASSERT_EQ(nvmm::compute_box_motion(f.data(), MVW, MVH, GRID, FW, FH,
                                       nullptr, 0, 1.0f, nullptr), 0u);
}

int main() {
    printf("=== nvmm motion compute tests ===\n");
    return tests_failed > 0 ? 1 : 0;
}
