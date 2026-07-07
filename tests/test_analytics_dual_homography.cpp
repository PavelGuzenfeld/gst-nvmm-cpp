/// Synthetic unit test for analytics/dual_homography.hpp.
///
/// Build a textured background, simulate a camera pan (a global translation) into
/// two reference frames, and add a blob that moves INDEPENDENTLY of that pan. The
/// dual-homography residual must peak on the independent mover and stay low on the
/// (pan-explained) static background — for BOTH feature pipelines. A third scene
/// adds a genuine second plane (foreground moving differently from the pan) to
/// verify H2 still absorbs real parallax after the mover-cluster gates.
#include "dual_homography.hpp"
#include "analytics_scene.h"
#include "test_harness.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using nvmm::img::Image;
using nvmm::img::window_max;

// median local-max over a grid of STATIC background points (away from the mover's
// horizontal band + the borders) — robust to a single high-residual pixel.
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
    // camera pan: refs are the background shifted (two different "past" offsets)
    ref_a = scene::translate(bg, 6, 4);
    ref_b = scene::translate(bg, 12, 8);
    // a blob moving independently of the pan (different position in each frame)
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
    ASSERT_TRUE(!res.empty());                        // a homography was fit

    const float mover = window_max(res.view(), 180, 180, 10);
    const float bg_median = static_bg_median(res);
    printf("[mover=%.1f bg_median=%.1f] ", mover, bg_median);

    ASSERT_TRUE(mover > 60.0f);                       // mover clearly lights up
    ASSERT_TRUE(mover > 5.0f * (bg_median + 1.0f));   // and dominates the static background
}

TEST(independent_mover_peaks_small_motion_pipeline) {
    check_mover_dominates(nvmm::motion::FeaturePipeline::small_motion);
}

TEST(independent_mover_peaks_orb_pipeline) {
    check_mover_dominates(nvmm::motion::FeaturePipeline::orb);
}

// Genuine parallax: a foreground plane (left strip, its own texture) moves
// differently from the background pan. H2 must absorb it (low residual there)
// while the independent mover still dominates. The mover bar is RELATIVE here:
// under a second plane, min-combining bounds the mover's absolute residual by
// its contrast against the H2-warped reference too.
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
        ASSERT_TRUE(mover > 5.0f * (fg_median + 1.0f));   // parallax plane suppressed
        ASSERT_TRUE(mover > 5.0f * (bg_median + 1.0f));   // background suppressed
    }
}

TEST(textureless_input_returns_empty) {
    Image<uint8_t> flat(256, 256, 120);               // no features -> no homography
    Image<float> res = nvmm::motion::independent_motion_residual(flat, flat, flat);
    ASSERT_TRUE(res.empty());
}

}  // namespace

int main() {
    printf("== analytics/dual_homography ==\n");
    return tests_failed > 0 ? 1 : 0;
}
