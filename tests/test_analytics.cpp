#include "persistence_gate.hpp"
#include "dual_homography.hpp"
#include "analytics_scene.h"
#include "low_texture_motion.hpp"
#include "active_region.hpp"
#include "detection_motion_gate.hpp"
#include "motion_magnify.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "test_harness.h"

namespace {

/// Input amplitude is 20, so an unamplified output has peak-to-peak 40. Defined first:
/// on a fresh heap glibc aborts on a one-row overrun in the magnifier; later it may not.
float steady_pp(float f0, float low, float high, float alpha) {
    nvmm::motion::MagnifyParams p; p.fps = 30.f; p.low_hz = low; p.high_hz = high; p.alpha = alpha;
    nvmm::motion::MotionMagnifier mag(p);
    const int N = 150, settle = 100;
    float lo = 1e9f, hi = -1e9f;
    for (int n = 0; n < N; n++) {
        const float v = 128.f + 20.f * std::sin(2.f * 3.14159265f * f0 * n / p.fps);
        nvmm::img::Image<uint8_t> f(16, 16, scene::clamp_u8(v));
        nvmm::img::Image<float> out = mag.process(f);
        const float c = out.at(8, 8);
        if (n >= settle) { lo = std::min(lo, c); hi = std::max(hi, c); }
    }
    return hi - lo;
}

TEST(in_band_oscillation_is_amplified) {
    float pp = steady_pp(4.f, 2.f, 8.f, 10.f);
    printf("[in-band pp=%.1f vs input 40] ", pp);
    ASSERT_TRUE(pp > 80.0f);
}

TEST(out_of_band_oscillation_is_not_amplified) {
    float pp = steady_pp(0.2f, 2.f, 8.f, 10.f);
    printf("[out-of-band pp=%.1f vs input 40] ", pp);
    ASSERT_TRUE(pp < 60.0f);
}

}

namespace {

using nvmm::track::Detection;
using nvmm::track::PersistenceGate;
using nvmm::track::PersistenceParams;

std::vector<Detection> one(float x, float y, bool supported) {
    return { Detection{x, y, 0.9f, supported} };
}

TEST(persistent_supported_confirms_then_latches) {
    PersistenceGate g{PersistenceParams{}};
    int first = -1;
    for (int f = 1; f <= 5; f++) ASSERT_TRUE(g.update(one(100, 100, true)) == -1);
    first = g.update(one(100, 100, true));
    ASSERT_TRUE(first == 0);
    ASSERT_TRUE(g.locked());
    ASSERT_TRUE(g.update(one(101, 100, true)) == 0);
}

TEST(unsupported_never_confirms) {
    PersistenceGate g{PersistenceParams{}};
    for (int f = 1; f <= 30; f++) ASSERT_TRUE(g.update(one(100, 100, false)) == -1);
    ASSERT_TRUE(!g.locked());
}

TEST(flickering_support_never_confirms) {
    PersistenceGate g{PersistenceParams{}};
    for (int f = 1; f <= 40; f++)
        ASSERT_TRUE(g.update(one(100, 100, f % 2 == 0)) == -1);
    ASSERT_TRUE(!g.locked());
}

}

namespace {

using nvmm::img::Image;
using nvmm::img::window_max;

float static_bg_median(const Image<float> &m) {
    std::vector<float> v;
    for (int y = 30; y < 226; y += 12)
        for (int x = 30; x < 226; x += 12)
            if (std::abs(y - 180) > 25) v.push_back(window_max(m.view(), (float)x, (float)y, 3));
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.f : v[v.size() / 2];
}

void pan_scene(Image<uint8_t> &cur, Image<uint8_t> &ref_a, Image<uint8_t> &ref_b) {
    Image<uint8_t> bg = scene::textured_bg(256, 12345);
    cur = bg;
    ref_a = scene::translate(bg, 6, 4);
    ref_b = scene::translate(bg, 12, 8);
    scene::fill_circle(cur, 180, 180, 6, 255);
    scene::fill_circle(ref_a, 150, 180, 6, 255);
    scene::fill_circle(ref_b, 120, 180, 6, 255);
}

void check_mover_dominates(nvmm::motion::FeaturePipeline pl) {
    Image<uint8_t> cur, ref_a, ref_b;
    pan_scene(cur, ref_a, ref_b);
    nvmm::motion::DualHomographyParams p;
    p.pipeline = pl;
    Image<float> res = nvmm::motion::independent_motion_residual(cur, ref_a, ref_b, p);
    ASSERT_TRUE(!res.empty());

    const float mover = window_max(res.view(), 180, 180, 10);
    const float bg_median = static_bg_median(res);
    printf("[mover=%.1f bg_median=%.1f] ", mover, bg_median);

    ASSERT_TRUE(mover > 60.0f);
    ASSERT_TRUE(mover > 5.0f * (bg_median + 1.0f));
}

TEST(independent_mover_peaks_small_motion_pipeline) {
    check_mover_dominates(nvmm::motion::FeaturePipeline::small_motion);
}

TEST(independent_mover_peaks_orb_pipeline) {
    check_mover_dominates(nvmm::motion::FeaturePipeline::orb);
}

/// The mover bar is lower than the pan-only scene's: min-combining against the
/// H2-warped reference also bounds the mover's absolute residual.
TEST(genuine_parallax_plane_absorbed_mover_survives) {
    Image<uint8_t> B = scene::textured_bg(256, 42);
    Image<uint8_t> F = scene::textured_bg(256, 4242);
    auto composite = [&](Image<uint8_t> bg, const Image<uint8_t> &fg) {
        for (int y = 0; y < bg.height(); y++)
            for (int x = 0; x < 90; x++) bg.at(y, x) = fg.at(y, x);
        return bg;
    };
    Image<uint8_t> cur = composite(B, F);
    Image<uint8_t> ref_a = composite(scene::translate(B, 6, 4), scene::translate(F, 10, 7));
    Image<uint8_t> ref_b = composite(scene::translate(B, 12, 8), scene::translate(F, 20, 14));
    scene::fill_circle(cur, 180, 180, 6, 255);
    scene::fill_circle(ref_a, 150, 180, 6, 255);
    scene::fill_circle(ref_b, 120, 180, 6, 255);

    for (int pl = 0; pl < 2; pl++) {
        nvmm::motion::DualHomographyParams p;
        p.pipeline = pl == 0 ? nvmm::motion::FeaturePipeline::small_motion
                             : nvmm::motion::FeaturePipeline::orb;
        Image<float> res = nvmm::motion::independent_motion_residual(cur, ref_a, ref_b, p);
        ASSERT_TRUE(!res.empty());
        const float mover = window_max(res.view(), 180, 180, 10);
        std::vector<float> fg_v;
        for (int y = 30; y < 226; y += 12)
            for (int x = 30; x < 80; x += 12) fg_v.push_back(window_max(res.view(), (float)x, (float)y, 3));
        std::sort(fg_v.begin(), fg_v.end());
        const float fg_median = fg_v[fg_v.size() / 2];
        const float bg_median = static_bg_median(res);
        printf("[p%d mover=%.1f fg=%.1f bg=%.1f] ", pl, mover, fg_median, bg_median);
        ASSERT_TRUE(mover > 30.0f);
        ASSERT_TRUE(mover > 5.0f * (fg_median + 1.0f));
        ASSERT_TRUE(mover > 5.0f * (bg_median + 1.0f));
    }
}

TEST(textureless_input_returns_empty) {
    Image<uint8_t> flat(256, 256, 120);
    Image<float> res = nvmm::motion::independent_motion_residual(flat, flat, flat);
    ASSERT_TRUE(res.empty());
}

constexpr int lone_corner_w = 64, lone_corner_h = 48, patch_radius = 15;

std::vector<nvmm::motion::detail::OrbFeature> orb_on_lone_bright_pixel(int x, int y) {
    Image<uint8_t> im(lone_corner_w, lone_corner_h, 0);
    im.at(y, x) = 255;
    nvmm::motion::detail::OrbParams p;
    p.nlevels = 1;
    return nvmm::motion::detail::orb_detect(im.view(), p);
}

void assert_single_keypoint_at(int x, int y) {
    const std::vector<nvmm::motion::detail::OrbFeature> f = orb_on_lone_bright_pixel(x, y);
    ASSERT_EQ(f.size(), (size_t)1);
    ASSERT_EQ(f[0].x, (float)x);
    ASSERT_EQ(f[0].y, (float)y);
}

TEST(orb_keeps_corner_exactly_patch_radius_from_left_edge) {
    assert_single_keypoint_at(patch_radius, lone_corner_h / 2);
}

TEST(orb_keeps_corner_exactly_patch_radius_from_right_edge) {
    assert_single_keypoint_at(lone_corner_w - 1 - patch_radius, lone_corner_h / 2);
}

TEST(orb_keeps_corner_exactly_patch_radius_from_top_edge) {
    assert_single_keypoint_at(lone_corner_w / 2, patch_radius);
}

TEST(orb_keeps_corner_exactly_patch_radius_from_bottom_edge) {
    assert_single_keypoint_at(lone_corner_w / 2, lone_corner_h - 1 - patch_radius);
}

TEST(orb_drops_corner_whose_patch_crosses_left_edge) {
    ASSERT_TRUE(orb_on_lone_bright_pixel(patch_radius - 1, lone_corner_h / 2).empty());
}

TEST(orb_drops_corner_whose_patch_crosses_right_edge) {
    ASSERT_TRUE(orb_on_lone_bright_pixel(lone_corner_w - patch_radius, lone_corner_h / 2).empty());
}

TEST(orb_drops_corner_whose_patch_crosses_top_edge) {
    ASSERT_TRUE(orb_on_lone_bright_pixel(lone_corner_w / 2, patch_radius - 1).empty());
}

TEST(orb_drops_corner_whose_patch_crosses_bottom_edge) {
    ASSERT_TRUE(orb_on_lone_bright_pixel(lone_corner_w / 2, lone_corner_h - patch_radius).empty());
}

int brief_pair_end_max_squared_radius(int end) {
    const int8_t *pat = nvmm::motion::detail::brief_pattern();
    int max_r2 = 0;
    for (int bit = 0; bit < 256; bit++) {
        const int8_t *q = pat + bit * 4 + end * 2;
        max_r2 = std::max(max_r2, q[0] * q[0] + q[1] * q[1]);
    }
    return max_r2;
}

TEST(brief_pair_first_ends_reach_exactly_the_patch_radius) {
    ASSERT_EQ(brief_pair_end_max_squared_radius(0), patch_radius * patch_radius);
}

TEST(brief_pair_second_ends_reach_exactly_the_patch_radius) {
    ASSERT_EQ(brief_pair_end_max_squared_radius(1), patch_radius * patch_radius);
}

}

namespace {

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

TEST(diff_kept_in_low_texture_masked_over_textured_patch) {
    nvmm::img::Image<uint8_t> cur = make_scene();
    nvmm::img::Image<uint8_t> ref(256, 256);
    for (int y = 0; y < 256; y++)
        for (int x = 0; x < 256; x++)
            ref.at(y, x) = (uint8_t)(cur.at(y, x) < 30 ? 0 : cur.at(y, x) - 30);

    nvmm::img::Image<float> m = nvmm::motion::low_texture_motion(cur, ref, ref);
    const float low_tex = nvmm::img::window_max(m.view(), 185, 185, 8);
    const float high_tex = nvmm::img::window_max(m.view(), 60, 60, 8);
    printf("[low_tex=%.1f high_tex=%.1f] ", low_tex, high_tex);

    ASSERT_TRUE(low_tex > 20.0f);
    ASSERT_TRUE(high_tex < 5.0f);
}

}

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
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
