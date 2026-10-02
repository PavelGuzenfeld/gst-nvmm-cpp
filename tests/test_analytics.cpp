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

#include <sys/mman.h>
#include <unistd.h>

#include "test_harness.h"

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

using nvmm::motion::detail::Plane;
using nvmm::motion::detail::Pt;

constexpr int plane_frame = 256;
constexpr float mover_dx = 30.f;

struct Correspondences {
    std::vector<Pt> cur, ref;
    void add(float x, float y, float dx, float dy) {
        cur.push_back(Pt{x, y});
        ref.push_back(Pt{x + dx, y + dy});
    }
};

Correspondences dominant_pan() {
    Correspondences c;
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++) c.add(16.f + 30.f * i, 16.f + 30.f * j, 6.f, 4.f);
    return c;
}

void fit_two_planes(const Correspondences &c, const nvmm::motion::DualHomographyParams &p,
                    Plane &pl1, Plane &pl2) {
    nvmm::motion::detail::two_planes(c.cur, c.ref, p, plane_frame, plane_frame, pl1, pl2);
}

TEST(mover_cluster_with_stray_inliers_sharing_h2_is_rejected) {
    Correspondences c = dominant_pan();
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++) c.add(100.f + 3.f * i, 100.f + 3.f * j, mover_dx, 0.f);
    const Pt strays[] = {{20.f, 200.f}, {160.f, 30.f}, {175.f, 220.f}, {190.f, 150.f},
                         {205.f, 60.f}, {220.f, 180.f}, {240.f, 40.f}};
    for (const Pt &s : strays) c.add(s.x, s.y, mover_dx, 0.f);
    Plane pl1, pl2;
    fit_two_planes(c, nvmm::motion::DualHomographyParams{}, pl1, pl2);
    ASSERT_TRUE(pl1.ok);
    ASSERT_TRUE(!pl2.ok);
}

Plane parallax_grid_plane(float step_x, float step_y) {
    Correspondences c = dominant_pan();
    for (int j = 0; j < 5; j++)
        for (int i = 0; i < 5; i++) c.add(40.f + step_x * i, 40.f + step_y * j, mover_dx, 0.f);
    nvmm::motion::DualHomographyParams p;
    p.min_plane_extent = 0.25f;
    Plane pl1, pl2;
    fit_two_planes(c, p, pl1, pl2);
    ASSERT_TRUE(pl1.ok);
    return pl2;
}

TEST(parallax_plane_whose_shortest_half_spans_exactly_the_extent_in_x_is_kept) {
    ASSERT_TRUE(parallax_grid_plane(32.f, 4.f).ok);
}

TEST(parallax_plane_whose_shortest_half_spans_exactly_the_extent_in_y_is_kept) {
    ASSERT_TRUE(parallax_grid_plane(4.f, 32.f).ok);
}

TEST(parallax_plane_whose_shortest_half_falls_short_of_the_extent_is_rejected) {
    ASSERT_TRUE(!parallax_grid_plane(31.f, 31.f).ok);
}

Plane corner_cluster_plus_spread_strays(int cluster, int strays) {
    Correspondences c = dominant_pan();
    for (int i = 0; i < cluster; i++)
        c.add(20.f + 3.f * (i % 4), 20.f + 3.f * (i / 4), mover_dx, 0.f);
    for (int i = 0; i < strays; i++)
        c.add(100.f + 14.f * i, 100.f + (float)((i * 37) % 120), mover_dx, 0.f);
    Plane pl1, pl2;
    fit_two_planes(c, nvmm::motion::DualHomographyParams{}, pl1, pl2);
    ASSERT_TRUE(pl1.ok);
    return pl2;
}

TEST(mover_cluster_holding_eleven_of_twenty_one_consensus_points_is_rejected) {
    ASSERT_TRUE(!corner_cluster_plus_spread_strays(11, 10).ok);
}

TEST(plane_whose_cluster_holds_ten_of_twenty_one_consensus_points_is_kept) {
    ASSERT_TRUE(corner_cluster_plus_spread_strays(10, 11).ok);
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
        const uint8_t shade = (uint8_t)rng.uniform(10, 60);
        const int radius = rng.uniform(2, 4);
        scene::fill_circle(f, x, y, radius, shade);
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

namespace {

/// Input amplitude is 20, so an unamplified output has peak-to-peak 40.
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

/// The frame's last row ends where a PROT_NONE page begins, so reading one row past
/// frame.height faults deterministically instead of reading whatever the heap holds.
TEST(magnifier_reads_no_row_past_the_frame) {
    const long page = sysconf(_SC_PAGESIZE);
    void *mem = mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_TRUE(mem != MAP_FAILED);
    uint8_t *guard = static_cast<uint8_t *>(mem) + page;
    ASSERT_EQ(mprotect(guard, page, PROT_NONE), 0);
    constexpr int kW = 16, kH = 16;
    uint8_t *pixels = guard - kW * kH;
    std::fill(pixels, guard, (uint8_t)128);

    nvmm::motion::MotionMagnifier mag;
    const nvmm::img::View<const uint8_t> frame(pixels, kW, kH, kW);
    for (int n = 0; n < 2; n++) ASSERT_EQ(mag.process(frame).height(), kH);
    munmap(mem, 2 * page);
}

}

namespace {

namespace det = nvmm::motion::detail;
using nvmm::img::Image;
using nvmm::img::Rect;
using nvmm::img::View;

/// FAST's radius-3 Bresenham circle, clockwise from 12 o'clock (Rosten & Drummond 2006).
const int fast_circle_dx[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
const int fast_circle_dy[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
constexpr int fast_bg = 100, fast_thresh = 20, fast_bright = 200, fast_dark = 30;

/// A 7x7 frame with margin 3 leaves (3,3) the only FAST candidate.
Image<uint8_t> centre_with_circle_arc(int start, int len, int value) {
    Image<uint8_t> im(7, 7, fast_bg);
    for (int k = 0; k < len; k++) {
        const int i = (start + k) % 16;
        im.at(3 + fast_circle_dy[i], 3 + fast_circle_dx[i]) = (uint8_t)value;
    }
    return im;
}

std::vector<det::Corner> fast_on(const Image<uint8_t> &im, int max_corners = 64) {
    return det::fast_corners(im.view(), fast_thresh, max_corners, 3);
}

TEST(fast_accepts_nine_pixel_arc_at_every_rotation) {
    for (int value : {fast_bright, fast_dark})
        for (int start = 0; start < 16; start++) {
            const std::vector<det::Corner> c = fast_on(centre_with_circle_arc(start, 9, value));
            ASSERT_EQ(c.size(), (size_t)1);
            ASSERT_EQ(c[0].x, 3);
            ASSERT_EQ(c[0].y, 3);
        }
}

TEST(fast_rejects_eight_pixel_arc_at_every_rotation) {
    for (int value : {fast_bright, fast_dark})
        for (int start = 0; start < 16; start++)
            ASSERT_TRUE(fast_on(centre_with_circle_arc(start, 8, value)).empty());
}

TEST(fast_rejects_ninth_arc_pixel_that_differs_by_exactly_the_threshold) {
    for (int sign : {1, -1}) {
        Image<uint8_t> im = centre_with_circle_arc(0, 8, sign > 0 ? fast_bright : fast_dark);
        im.at(3 + fast_circle_dy[8], 3 + fast_circle_dx[8]) = (uint8_t)(fast_bg + sign * fast_thresh);
        ASSERT_TRUE(fast_on(im).empty());
    }
}

TEST(fast_rejects_two_compass_points_without_a_contiguous_arc) {
    for (int value : {fast_bright, fast_dark}) {
        Image<uint8_t> im(7, 7, fast_bg);
        for (int i : {0, 4}) im.at(3 + fast_circle_dy[i], 3 + fast_circle_dx[i]) = (uint8_t)value;
        ASSERT_TRUE(fast_on(im).empty());
    }
}

float only_corner_score(const Image<uint8_t> &im) {
    const std::vector<det::Corner> c = fast_on(im);
    ASSERT_EQ(c.size(), (size_t)1);
    return c[0].score;
}

/// The score sums |I(p) - I(c)| - t over every circle pixel where that is positive.
TEST(fast_score_sums_each_circle_pixel_excess_over_the_threshold) {
    ASSERT_EQ(only_corner_score(centre_with_circle_arc(0, 9, fast_bright)),
              9.f * (fast_bright - fast_bg - fast_thresh));
    ASSERT_EQ(only_corner_score(centre_with_circle_arc(0, 9, fast_dark)),
              9.f * (fast_bg - fast_dark - fast_thresh));
}

constexpr int nms_size = 32, nms_c = 16;
const int neighbour_dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
const int neighbour_dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

std::vector<det::Corner> fast_on_adjacent_pair(int dir, int centre_value, int neighbour_value) {
    Image<uint8_t> im(nms_size, nms_size, 0);
    im.at(nms_c, nms_c) = (uint8_t)centre_value;
    im.at(nms_c + neighbour_dy[dir], nms_c + neighbour_dx[dir]) = (uint8_t)neighbour_value;
    return fast_on(im);
}

TEST(nms_keeps_exactly_one_of_two_equal_adjacent_corners) {
    for (int dir = 0; dir < 8; dir++)
        ASSERT_EQ(fast_on_adjacent_pair(dir, 200, 200).size(), (size_t)1);
}

TEST(nms_drops_a_corner_with_a_stronger_neighbour_in_any_direction) {
    for (int dir = 0; dir < 8; dir++) {
        const std::vector<det::Corner> c = fast_on_adjacent_pair(dir, 150, 200);
        ASSERT_EQ(c.size(), (size_t)1);
        ASSERT_EQ(c[0].x, nms_c + neighbour_dx[dir]);
        ASSERT_EQ(c[0].y, nms_c + neighbour_dy[dir]);
    }
}

TEST(fast_keeps_the_strongest_max_corners_strongest_first) {
    Image<uint8_t> im(nms_size, nms_size, 0);
    im.at(8, 8) = 160;
    im.at(8, 20) = 200;
    im.at(20, 14) = 180;
    const std::vector<det::Corner> c = fast_on(im, 2);
    ASSERT_EQ(c.size(), (size_t)2);
    ASSERT_EQ(c[0].x, 20);
    ASSERT_EQ(c[0].y, 8);
    ASSERT_EQ(c[1].x, 14);
    ASSERT_EQ(c[1].y, 20);
}

TEST(fast_corner_inside_the_margin_does_not_suppress_its_neighbour) {
    constexpr int margin = 8, last = nms_size - margin - 1;
    for (int axis = 0; axis < 2; axis++) {
        Image<uint8_t> im(nms_size, nms_size, 0);
        const int bx = axis == 0 ? last + 1 : nms_c, by = axis == 0 ? nms_c : last + 1;
        im.at(by, bx) = 200;
        im.at(axis == 0 ? nms_c : last, axis == 0 ? last : nms_c) = 150;
        const std::vector<det::Corner> c = det::fast_corners(im.view(), fast_thresh, 64, margin);
        ASSERT_EQ(c.size(), (size_t)1);
        ASSERT_EQ(c[0].x, axis == 0 ? last : nms_c);
        ASSERT_EQ(c[0].y, axis == 0 ? nms_c : last);
    }
}

constexpr int orient_size = 48, orient_c = 24;

void orientation_of_lone_pixel(int dx, int dy, float &ca, float &sa) {
    Image<uint8_t> im(orient_size, orient_size, 0);
    im.at(orient_c + dy, orient_c + dx) = 200;
    det::orb_orientation(im.view(), orient_c, orient_c, ca, sa);
}

/// Every offset is an integer point on the radius-15 circle, so cos and sin are exact quotients.
TEST(orientation_points_at_a_lone_pixel_on_the_disc_rim) {
    const int rim[8][2] = {{15, 0}, {-15, 0}, {0, 15}, {0, -15},
                           {12, 9}, {-9, 12}, {-12, -9}, {9, -12}};
    for (const auto &p : rim) {
        float ca = 0.f, sa = 0.f;
        orientation_of_lone_pixel(p[0], p[1], ca, sa);
        ASSERT_EQ(ca, (float)p[0] / 15.f);
        ASSERT_EQ(sa, (float)p[1] / 15.f);
    }
}

TEST(orientation_ignores_a_pixel_just_outside_the_disc) {
    float ca = 0.f, sa = 0.f;
    orientation_of_lone_pixel(-13, -9, ca, sa);
    ASSERT_EQ(ca, 1.f);
    ASSERT_EQ(sa, 0.f);
}

constexpr int ramp_size = 64, ramp_c = 32;

/// The ramp's step across the FAST circle stays under the threshold, so the spike is the only corner.
Image<uint8_t> ramp_with_spike() {
    Image<uint8_t> im(ramp_size, ramp_size);
    for (int y = 0; y < ramp_size; y++)
        for (int x = 0; x < ramp_size; x++)
            im.at(y, x) = scene::clamp_u8(128.f + 3.f * (x - ramp_c) + 4.f * (y - ramp_c));
    im.at(ramp_c, ramp_c) = 255;
    return im;
}

bool within_blur_of_the_spike(int x, int y) {
    return std::max(std::abs(x - ramp_c), std::abs(y - ramp_c)) <= 3;
}

/// Steered BRIEF (Rublee 2011): bit i is 1 iff the smoothed image is darker at pair i's first end.
/// The dyadic 7-tap blur leaves a ramp exact, so away from the spike raw pixels are the oracle.
TEST(descriptor_bit_set_when_first_sample_is_darker) {
    const Image<uint8_t> im = ramp_with_spike();
    det::OrbParams p;
    p.nlevels = 1;
    const std::vector<det::OrbFeature> f = det::orb_detect(im.view(), p);
    ASSERT_EQ(f.size(), (size_t)1);
    float ca = 0.f, sa = 0.f;
    det::orb_orientation(im.view(), ramp_c, ramp_c, ca, sa);
    const int8_t *pat = det::brief_pattern();
    int checked = 0;
    for (int bit = 0; bit < 256; bit++) {
        const int8_t *q = pat + bit * 4;
        const int x0 = ramp_c + (int)std::lround(ca * q[0] - sa * q[1]);
        const int y0 = ramp_c + (int)std::lround(sa * q[0] + ca * q[1]);
        const int x1 = ramp_c + (int)std::lround(ca * q[2] - sa * q[3]);
        const int y1 = ramp_c + (int)std::lround(sa * q[2] + ca * q[3]);
        if (within_blur_of_the_spike(x0, y0) || within_blur_of_the_spike(x1, y1)) continue;
        const bool set = ((f[0].desc[bit >> 6] >> (bit & 63)) & 1u) != 0;
        ASSERT_EQ(set, im.at(y0, x0) < im.at(y1, x1));
        checked++;
    }
    printf("[checked=%d] ", checked);
    ASSERT_TRUE(checked >= 128);
}

/// The 7-tap blur darkens exactly the 7x7 block around a dark spike, and the spike leaves the
/// moments zero, so the pattern is unsteered and a pair sets its bit iff only its first end is in it.
TEST(descriptor_bit_set_only_where_the_first_end_falls_in_a_dark_spike_blur) {
    Image<uint8_t> im(ramp_size, ramp_size, 255);
    im.at(ramp_c, ramp_c) = 0;
    det::OrbParams p;
    p.nlevels = 1;
    const std::vector<det::OrbFeature> f = det::orb_detect(im.view(), p);
    ASSERT_EQ(f.size(), (size_t)1);
    const int8_t *pat = det::brief_pattern();
    for (int bit = 0; bit < 256; bit++) {
        const int8_t *q = pat + bit * 4;
        const bool first_in = within_blur_of_the_spike(ramp_c + q[0], ramp_c + q[1]);
        const bool second_in = within_blur_of_the_spike(ramp_c + q[2], ramp_c + q[3]);
        if (first_in && second_in) continue;
        const bool set = ((f[0].desc[bit >> 6] >> (bit & 63)) & 1u) != 0;
        ASSERT_EQ(set, first_in);
    }
}

TEST(descriptor_bit_set_only_where_the_second_end_falls_in_a_bright_spike_blur) {
    Image<uint8_t> im(ramp_size, ramp_size, 0);
    im.at(ramp_c, ramp_c) = 255;
    det::OrbParams p;
    p.nlevels = 1;
    const std::vector<det::OrbFeature> f = det::orb_detect(im.view(), p);
    ASSERT_EQ(f.size(), (size_t)1);
    const int8_t *pat = det::brief_pattern();
    for (int bit = 0; bit < 256; bit++) {
        const int8_t *q = pat + bit * 4;
        const bool first_in = within_blur_of_the_spike(ramp_c + q[0], ramp_c + q[1]);
        const bool second_in = within_blur_of_the_spike(ramp_c + q[2], ramp_c + q[3]);
        if (first_in && second_in) continue;
        const bool set = ((f[0].desc[bit >> 6] >> (bit & 63)) & 1u) != 0;
        ASSERT_EQ(set, second_in);
    }
}

Image<uint8_t> lone_pixel_grid(int w, int h, int spacing) {
    Image<uint8_t> im(w, h, 0);
    for (int y = 16; y < h - 15; y += spacing)
        for (int x = 16; x < w - 15; x += spacing) im.at(y, x) = 255;
    return im;
}

det::OrbParams two_level_orb(int nfeatures) {
    det::OrbParams p;
    p.nfeatures = nfeatures;
    p.nlevels = 2;
    p.scale_factor = 1.2f;
    return p;
}

/// Level i's share of nfeatures is proportional to its area, scale^-2i: 50 / (1 + 1.2^-2) = 29.5.
/// A 36-px-wide frame has a 30-px level 1, too narrow for the patch, so only level 0 contributes.
TEST(orb_first_level_takes_its_area_share_of_nfeatures) {
    const Image<uint8_t> im = lone_pixel_grid(36, 200, 4);
    ASSERT_EQ(det::orb_detect(im.view(), two_level_orb(50)).size(), (size_t)30);
}

/// Shares 29.5 and 20.5 round to 30 and 20.
TEST(orb_two_levels_together_take_nfeatures) {
    const Image<uint8_t> im = lone_pixel_grid(120, 120, 6);
    ASSERT_EQ(det::orb_detect(im.view(), two_level_orb(50)).size(), (size_t)50);
}

std::vector<det::OrbFeature> orb_on_lone_pixel_frame(int size) {
    Image<uint8_t> im(size, size, 0);
    im.at(18, 18) = 255;
    return det::orb_detect(im.view(), two_level_orb(100));
}

/// 37 px scales to a 31-px level 1 whose only FAST candidate (15,15) samples source pixel 18.
TEST(orb_reports_an_upper_level_keypoint_in_level_zero_coordinates) {
    const std::vector<det::OrbFeature> f = orb_on_lone_pixel_frame(37);
    ASSERT_EQ(f.size(), (size_t)2);
    const float level_one = (float)(15 * (double)1.2f);
    ASSERT_EQ(f[1].x, level_one);
    ASSERT_EQ(f[1].y, level_one);
}

TEST(orb_skips_a_level_no_wider_than_twice_the_patch_radius) {
    ASSERT_EQ(orb_on_lone_pixel_frame(36).size(), (size_t)1);
}

TEST(brief_pattern_spans_the_full_minus_13_to_13_window) {
    const int8_t *pat = det::brief_pattern();
    const auto mm = std::minmax_element(pat, pat + 256 * 4);
    ASSERT_EQ(*mm.first, -13);
    ASSERT_EQ(*mm.second, 13);
}

TEST(brief_pair_ends_are_distinct) {
    const int8_t *pat = det::brief_pattern();
    for (int bit = 0; bit < 256; bit++) {
        const int8_t *q = pat + bit * 4;
        ASSERT_TRUE(q[0] != q[2] || q[1] != q[3]);
    }
}

TEST(hamming_sums_differing_bits_over_all_four_words) {
    const uint64_t a[4] = {0xFF, 0x1, 0x3, 0x7};
    const uint64_t zero[4] = {0, 0, 0, 0};
    ASSERT_EQ(det::hamming256(a, zero), 14);
    ASSERT_EQ(det::hamming256(zero, a), 14);
}

det::OrbFeature orb_feature(float x, float y, uint64_t word0) {
    det::OrbFeature f;
    f.x = x;
    f.y = y;
    f.desc[0] = word0;
    return f;
}

std::vector<det::Pt> orb_match_targets(const std::vector<det::OrbFeature> &ref, float ratio = 0.75f) {
    std::vector<det::Pt> p1, p2;
    det::orb_match({orb_feature(1.f, 2.f, 0)}, ref, ratio, p1, p2);
    ASSERT_EQ(p1.size(), p2.size());
    for (const det::Pt &p : p1) {
        ASSERT_EQ(p.x, 1.f);
        ASSERT_EQ(p.y, 2.f);
    }
    return p2;
}

TEST(orb_match_needs_two_reference_features) {
    ASSERT_TRUE(orb_match_targets({orb_feature(5.f, 6.f, 0)}).empty());
}

TEST(orb_match_pairs_with_the_nearest_of_two_reference_features) {
    const std::vector<det::Pt> m = orb_match_targets({orb_feature(5.f, 6.f, 0), orb_feature(7.f, 8.f, 0xFF)});
    ASSERT_EQ(m.size(), (size_t)1);
    ASSERT_EQ(m[0].x, 5.f);
    ASSERT_EQ(m[0].y, 6.f);
}

TEST(orb_match_pairs_with_the_nearest_wherever_it_sits) {
    const std::vector<det::Pt> m = orb_match_targets(
        {orb_feature(5.f, 6.f, 0x1F), orb_feature(7.f, 8.f, 0x1), orb_feature(9.f, 10.f, 0x1FF)});
    ASSERT_EQ(m.size(), (size_t)1);
    ASSERT_EQ(m[0].x, 7.f);
    ASSERT_EQ(m[0].y, 8.f);
}

/// Lowe's ratio test is strict: 3 < 0.75 * 4 fails.
TEST(orb_match_rejects_nearest_at_exactly_ratio_times_second) {
    ASSERT_TRUE(orb_match_targets({orb_feature(5.f, 6.f, 0x7), orb_feature(7.f, 8.f, 0xF)}).empty());
}

TEST(orb_match_ratio_uses_a_runner_up_seen_after_the_nearest) {
    ASSERT_TRUE(orb_match_targets({orb_feature(5.f, 6.f, 0xF), orb_feature(7.f, 8.f, 0x1FF),
                                   orb_feature(9.f, 10.f, 0x1F)}).empty());
}

const int zncc_patch[9] = {10, 40, 20, 70, 30, 90, 50, 60, 80};

/// The patch sums to 9 * 50, so every mean-removed moment stays an exact double.
Image<uint8_t> frame_with_patch(int cx, int cy, int gain, int offset) {
    Image<uint8_t> im(8, 8, 0);
    for (int k = 0; k < 9; k++)
        im.at(cy - 1 + k / 3, cx - 1 + k % 3) = (uint8_t)(gain * zncc_patch[k] + offset);
    return im;
}

TEST(zncc_is_one_under_gain_and_offset) {
    const Image<uint8_t> a = frame_with_patch(2, 3, 1, 0), b = frame_with_patch(5, 4, 2, 10);
    ASSERT_EQ(det::zncc_at(a.view(), 2, 3, b.view(), 5, 4, 1), 1.f);
}

TEST(zncc_is_minus_one_for_an_inverted_patch) {
    const Image<uint8_t> a = frame_with_patch(2, 3, 1, 0), b = frame_with_patch(5, 4, -1, 255);
    ASSERT_EQ(det::zncc_at(a.view(), 2, 3, b.view(), 5, 4, 1), -1.f);
}

TEST(zncc_is_minus_one_when_either_patch_is_flat) {
    const Image<uint8_t> textured = frame_with_patch(3, 3, 1, 0), flat(8, 8, 77);
    ASSERT_EQ(det::zncc_at(textured.view(), 3, 3, flat.view(), 4, 4, 1), -1.f);
    ASSERT_EQ(det::zncc_at(flat.view(), 4, 4, textured.view(), 3, 3, 1), -1.f);
}

TEST(zncc_window_reaches_every_corner_of_the_patch) {
    for (int dy : {-2, 2})
        for (int dx : {-2, 2}) {
            Image<uint8_t> a(10, 10, 0), b(10, 10, 0);
            a.at(4 + dy, 3 + dx) = 90;
            b.at(5 + dy, 6 + dx) = 90;
            ASSERT_EQ(det::zncc_at(a.view(), 3, 4, b.view(), 6, 5, 2), 1.f);
        }
}

Image<uint8_t> random_frame(int w, int h, unsigned seed) {
    Image<uint8_t> im(w, h);
    scene::Rng rng(seed);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) im.at(y, x) = (uint8_t)rng.uniform(0, 256);
    return im;
}

TEST(resize_by_half_averages_each_2x2_block_rounding_half_up) {
    const Image<uint8_t> src = random_frame(8, 6, 11);
    const Image<uint8_t> dst = det::resize_bilinear(src.view(), 4, 3);
    for (int y = 0; y < 3; y++)
        for (int x = 0; x < 4; x++) {
            const int sum = src.at(2 * y, 2 * x) + src.at(2 * y, 2 * x + 1) +
                            src.at(2 * y + 1, 2 * x) + src.at(2 * y + 1, 2 * x + 1);
            ASSERT_EQ((int)dst.at(y, x), (sum + 2) / 4);
        }
}

TEST(resize_by_three_samples_each_block_centre) {
    const Image<uint8_t> src = random_frame(12, 9, 12);
    const Image<uint8_t> dst = det::resize_bilinear(src.view(), 4, 3);
    for (int y = 0; y < 3; y++)
        for (int x = 0; x < 4; x++) ASSERT_EQ(dst.at(y, x), src.at(3 * y + 1, 3 * x + 1));
}

TEST(resize_up_repeats_the_last_row_and_column) {
    const Image<uint8_t> src = random_frame(2, 2, 13);
    const Image<uint8_t> dst = det::resize_bilinear(src.view(), 4, 4);
    ASSERT_EQ(dst.at(3, 3), src.at(1, 1));
}

det::SmallMotionParams small_search() {
    det::SmallMotionParams p;
    p.search_radius = 6;
    return p;
}

void small_motion_on(View<const uint8_t> cur, View<const uint8_t> ref,
                     const std::vector<det::Corner> &corners, std::vector<det::Pt> &p1, std::vector<det::Pt> &p2) {
    det::small_motion_matches(cur, ref, corners, small_search(), p1, p2);
}

TEST(small_motion_recovers_translations_between_coarse_grid_points) {
    Image<uint8_t> base = random_frame(96, 96, 21);
    scene::gaussian_blur_u8(base, 7);
    const std::vector<det::Corner> corners = {{20, 20, 1.f}, {32, 40, 1.f}, {44, 28, 1.f}};
    const int shifts[3][2] = {{1, -2}, {-4, 5}, {5, 4}};
    for (const auto &s : shifts) {
        const View<const uint8_t> cur = base.view().sub(Rect{16, 16, 64, 64});
        const View<const uint8_t> ref = base.view().sub(Rect{16 - s[0], 16 - s[1], 64, 64});
        std::vector<det::Pt> p1, p2;
        small_motion_on(cur, ref, corners, p1, p2);
        ASSERT_EQ(p1.size(), corners.size());
        for (size_t i = 0; i < corners.size(); i++) {
            ASSERT_EQ(p1[i].x, (float)corners[i].x);
            ASSERT_EQ(p1[i].y, (float)corners[i].y);
            ASSERT_EQ(p2[i].x, (float)(corners[i].x + s[0]));
            ASSERT_EQ(p2[i].y, (float)(corners[i].y + s[1]));
        }
    }
}

TEST(small_motion_rejects_a_corner_on_texture_repeating_within_the_search) {
    const Image<uint8_t> tile = random_frame(6, 6, 22);
    Image<uint8_t> im(64, 64);
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++) im.at(y, x) = tile.at(y % 6, x % 6);
    std::vector<det::Pt> p1, p2;
    small_motion_on(im.view(), im.view(), {{32, 32, 1.f}}, p1, p2);
    ASSERT_TRUE(p1.empty());
}

constexpr int edge_frame = 64, patch_r = 5;

std::vector<det::Pt> small_motion_identity_matches(const std::vector<det::Corner> &corners) {
    const Image<uint8_t> im = random_frame(edge_frame, edge_frame, 23);
    std::vector<det::Pt> p1, p2;
    small_motion_on(im.view(), im.view(), corners, p1, p2);
    for (size_t i = 0; i < p1.size(); i++) {
        ASSERT_EQ(p2[i].x, p1[i].x);
        ASSERT_EQ(p2[i].y, p1[i].y);
    }
    return p1;
}

TEST(small_motion_keeps_a_corner_exactly_patch_radius_from_each_edge) {
    const int far = edge_frame - 1 - patch_r, mid = edge_frame / 2;
    for (const det::Corner &c : {det::Corner{patch_r, mid, 1.f}, det::Corner{far, mid, 1.f},
                                 det::Corner{mid, patch_r, 1.f}, det::Corner{mid, far, 1.f}}) {
        const std::vector<det::Pt> m = small_motion_identity_matches({c});
        ASSERT_EQ(m.size(), (size_t)1);
        ASSERT_EQ(m[0].x, (float)c.x);
        ASSERT_EQ(m[0].y, (float)c.y);
    }
}

Image<uint8_t> smooth_frame(int size, unsigned seed) {
    Image<uint8_t> im = random_frame(size, size, seed);
    scene::gaussian_blur_u8(im, 7);
    return im;
}

/// cur and ref are windows of one larger frame, so memory past cur's edge holds a true match.
std::vector<det::Pt> small_motion_in_windows(const Image<uint8_t> &base, int sx, int sy,
                                             const det::Corner &c, const det::SmallMotionParams &p) {
    const View<const uint8_t> cur = base.view().sub(Rect{16, 16, edge_frame, edge_frame});
    const View<const uint8_t> ref = base.view().sub(Rect{16 - sx, 16 - sy, edge_frame, edge_frame});
    std::vector<det::Pt> p1, p2;
    det::small_motion_matches(cur, ref, {c}, p, p1, p2);
    return p2;
}

TEST(small_motion_skips_a_corner_whose_patch_crosses_an_edge) {
    const Image<uint8_t> base = smooth_frame(96, 24);
    const int far = edge_frame - patch_r, mid = edge_frame / 2;
    ASSERT_TRUE(small_motion_in_windows(base, 2, 0, {patch_r - 1, mid, 1.f}, small_search()).empty());
    ASSERT_TRUE(small_motion_in_windows(base, -2, 0, {far, mid, 1.f}, small_search()).empty());
    ASSERT_TRUE(small_motion_in_windows(base, 0, 2, {mid, patch_r - 1, 1.f}, small_search()).empty());
    ASSERT_TRUE(small_motion_in_windows(base, 0, -2, {mid, far, 1.f}, small_search()).empty());
}

bool patch_inside_frame(const det::Pt &p) {
    return p.x >= patch_r && p.y >= patch_r && p.x < edge_frame - patch_r && p.y < edge_frame - patch_r;
}

TEST(small_motion_never_reports_a_match_whose_patch_leaves_the_reference_frame) {
    const Image<uint8_t> base = smooth_frame(96, 25);
    const int near = patch_r, far = edge_frame - patch_r - 1, mid = edge_frame / 2;
    const int cases[4][4] = {{near, mid, -1, 0}, {far, mid, 1, 0}, {mid, near, 0, -1}, {mid, far, 0, 1}};
    for (const auto &k : cases)
        for (const det::Pt &m : small_motion_in_windows(base, k[2], k[3], {k[0], k[1], 1.f}, small_search()))
            ASSERT_TRUE(patch_inside_frame(m));
}

TEST(small_motion_accepts_a_match_whose_zncc_equals_the_minimum) {
    det::SmallMotionParams p = small_search();
    p.zncc_min = 1.f;
    const Image<uint8_t> base = smooth_frame(96, 26);
    ASSERT_EQ(small_motion_in_windows(base, 0, 0, {32, 32, 1.f}, p).size(), (size_t)1);
}

/// The reference frame ends where a PROT_NONE page begins, so a search candidate whose
/// patch reaches past the last row or column faults instead of reading the heap.
TEST(small_motion_reads_no_reference_pixel_past_the_frame) {
    const long page = sysconf(_SC_PAGESIZE);
    const long frame_bytes = (long)edge_frame * edge_frame;
    const long pages = (frame_bytes + page - 1) / page;
    void *mem = mmap(nullptr, (size_t)(pages + 1) * page, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_TRUE(mem != MAP_FAILED);
    uint8_t *guard = static_cast<uint8_t *>(mem) + pages * page;
    ASSERT_EQ(mprotect(guard, page, PROT_NONE), 0);
    uint8_t *pixels = guard - frame_bytes;
    const Image<uint8_t> cur = smooth_frame(edge_frame, 27);
    std::copy(cur.data(), cur.data() + frame_bytes, pixels);
    const View<const uint8_t> ref(pixels, edge_frame, edge_frame, edge_frame);
    const int grid_hits_last_column = edge_frame - patch_r - 3, last_row_patch = edge_frame - patch_r - 1;
    const std::vector<det::Corner> corners = {{grid_hits_last_column, grid_hits_last_column, 1.f},
                                              {grid_hits_last_column, last_row_patch, 1.f}};
    std::vector<det::Pt> p1, p2;
    det::small_motion_matches(cur.view(), ref, corners, small_search(), p1, p2);
    ASSERT_EQ(p2.size(), corners.size());
    for (size_t i = 0; i < corners.size(); i++) {
        ASSERT_EQ(p2[i].x, (float)corners[i].x);
        ASSERT_EQ(p2[i].y, (float)corners[i].y);
    }
    munmap(mem, (size_t)(pages + 1) * page);
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
