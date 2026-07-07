/// Synthetic unit test for analytics/low_texture_motion.hpp.
///
/// A blob that appears in the low-texture background must light up; an identical blob
/// inside a high-texture patch must be suppressed (the whole point of the mask).
#include "low_texture_motion.hpp"
#include "analytics_scene.h"
#include "test_harness.h"

namespace {

// low-texture background (flat + faint noise) with one high-texture patch
// (per-pixel random noise -> high gradient everywhere inside it).
nvmm::img::Image<uint8_t> make_scene() {
    scene::Rng rng(3);
    nvmm::img::Image<uint8_t> f(256, 256);
    for (int y = 0; y < 256; y++)
        for (int x = 0; x < 256; x++)
            f.at(y, x) = scene::clamp_u8(110.f + rng.gauss(3.f));
    for (int y = 30; y < 90; y++)
        for (int x = 30; x < 90; x++) f.at(y, x) = (uint8_t)rng.uniform(0, 256);
    return f;
}

// The guarantee: frame-diff is kept only where the CURRENT frame is low-texture.
// Apply a uniform difference everywhere and check it survives in the flat background
// but is masked out over the high-texture patch.
TEST(diff_kept_in_low_texture_masked_over_textured_patch) {
    nvmm::img::Image<uint8_t> cur = make_scene();
    nvmm::img::Image<uint8_t> ref(256, 256);        // cur - 30 -> |diff| == 30 everywhere
    for (int y = 0; y < 256; y++)
        for (int x = 0; x < 256; x++)
            ref.at(y, x) = (uint8_t)(cur.at(y, x) < 30 ? 0 : cur.at(y, x) - 30);

    nvmm::img::Image<float> m = nvmm::motion::low_texture_motion(cur, ref, ref);
    const float low_tex = nvmm::img::window_max(m.view(), 185, 185, 8);   // flat bg -> kept
    const float high_tex = nvmm::img::window_max(m.view(), 60, 60, 8);    // textured -> masked
    printf("[low_tex=%.1f high_tex=%.1f] ", low_tex, high_tex);

    ASSERT_TRUE(low_tex > 20.0f);                    // ~30 diff survives over low texture
    ASSERT_TRUE(high_tex < 5.0f);                    // suppressed where the frame is textured
}

}  // namespace

int main() {
    printf("== analytics/low_texture_motion ==\n");
    return tests_failed > 0 ? 1 : 0;
}
