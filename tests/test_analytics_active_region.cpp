#include "active_region.hpp"
#include "analytics_scene.h"
#include "test_harness.h"

namespace {

TEST(trims_uniform_bars_keeps_dark_textured_content) {
    nvmm::img::Image<uint8_t> f(200, 200, 128);
    scene::Rng rng(7);
    for (int i = 0; i < 120; i++) {
        const int x = rng.uniform(42, 158), y = rng.uniform(4, 196);
        scene::fill_circle(f, x, y, rng.uniform(2, 4), (uint8_t)rng.uniform(10, 60));
    }
    scene::gaussian_blur_u8(f, 3);

    nvmm::img::Rect r = nvmm::video::active_region(f);
    printf("[x=%d w=%d] ", r.x, r.w);
    ASSERT_TRUE(r.x >= 38 && r.x <= 44);
    ASSERT_TRUE(r.x + r.w >= 156 && r.x + r.w <= 162);
}

TEST(uniform_frame_returns_full) {
    nvmm::img::Image<uint8_t> f(120, 120, 50);
    nvmm::img::Rect r = nvmm::video::active_region(f);
    ASSERT_TRUE(r.w == f.width() && r.h == f.height());
}

}

int main() {
    printf("== analytics/active_region ==\n");
    return tests_failed > 0 ? 1 : 0;
}
