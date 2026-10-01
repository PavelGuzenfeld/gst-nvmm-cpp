#include "dual_homography.hpp"
#include "analytics_scene.h"
#include "test_harness.h"

#include <algorithm>
#include <cmath>
#include <vector>

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

int main() {
    printf("== analytics/dual_homography ==\n");
    return tests_failed > 0 ? 1 : 0;
}
