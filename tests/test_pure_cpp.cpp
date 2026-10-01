#include "gmc_backend.hpp"
#include "gmc_mask.hpp"
#include "phase_correlation.hpp"
#include "samurai_gmc.hpp"
#include "samurai_view.hpp"
#include "vit_grid.hpp"
#include "xfeat_motion.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "test_harness.h"

namespace {

using nvmm::GmcBackend;

std::vector<uint8_t> lcg_noise(int w, int h) {
    std::vector<uint8_t> v((size_t)w * h);
    uint32_t s = 0x9e3779b9u;
    for (auto &p : v) { s = s * 1664525u + 1013904223u; p = (uint8_t)((s >> 24) & 0xFF); }
    return v;
}

uint8_t circular_at(const std::vector<uint8_t> &img, int w, int h, int x, int y) {
    x = ((x % w) + w) % w; y = ((y % h) + h) % h;
    return img[(size_t)y * w + x];
}

/// Sub-pixel precision was certified against cv::phaseCorrelate (0.009 px). The
/// 0.15 px bound only has to catch sign flips, a missing fft-shift and a wrong
/// peak, which are all whole-pixel errors.
TEST(phase_correlation_recovers_known_circular_shifts) {
    constexpr int W = 128, H = 64;
    const std::vector<uint8_t> base = lcg_noise(W, H);
    nvmm::PhaseCorrelator pc(W, H);
    const int shifts[][2] = {{0, 0}, {1, 0}, {0, 1}, {3, -2}, {-5, 4}, {7, -6}};
    for (auto &sh : shifts) {
        const int sx = sh[0], sy = sh[1];
        std::vector<float> a((size_t)W * H), b((size_t)W * H);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                a[(size_t)y * W + x] = circular_at(base, W, H, x, y);
                b[(size_t)y * W + x] = circular_at(base, W, H, x - sx, y - sy);
            }
        const nvmm::PhaseCorrelator::Shift r = pc.correlate(a.data(), b.data());
        ASSERT_TRUE(std::fabs(r.x - sx) < 0.15 && std::fabs(r.y - sy) < 0.15);
        ASSERT_TRUE(r.response > 0.80);
    }
}

TEST(gmc_auto_backend_prefers_fft_cuda_then_pva_then_fft_cpu) {
    ASSERT_TRUE(resolve_gmc_backend(GmcBackend::Auto, true, true) == GmcBackend::FftCuda);
    ASSERT_TRUE(resolve_gmc_backend(GmcBackend::Auto, false, true) == GmcBackend::Pva);
    ASSERT_TRUE(resolve_gmc_backend(GmcBackend::Auto, false, false) == GmcBackend::FftCpu);
}

TEST(gmc_unavailable_explicit_backend_degrades_to_fft_cpu_never_pva) {
    ASSERT_TRUE(resolve_gmc_backend(GmcBackend::FftCuda, false, true) == GmcBackend::FftCpu);
    ASSERT_TRUE(resolve_gmc_backend(GmcBackend::FftCuda, false, false) == GmcBackend::FftCpu);
    ASSERT_TRUE(resolve_gmc_backend(GmcBackend::Pva, false, false) == GmcBackend::FftCpu);
    ASSERT_TRUE(resolve_gmc_backend(GmcBackend::Ncc, true, true) == GmcBackend::Ncc);
    ASSERT_TRUE(resolve_gmc_backend(GmcBackend::FftCpu, true, true) == GmcBackend::FftCpu);
}

TEST(gmc_patch_size_is_256_for_pva_else_128) {
    ASSERT_TRUE(nvmm::gmc_patch_size(GmcBackend::Ncc) == 128);
    ASSERT_TRUE(nvmm::gmc_patch_size(GmcBackend::FftCpu) == 128);
    ASSERT_TRUE(nvmm::gmc_patch_size(GmcBackend::Pva) == 256);
}

/// NCC and FFT have different native sign conventions. A flipped sign fed to
/// kf.shift() doubles camera motion instead of cancelling it.
TEST(gmc_ncc_and_fft_agree_in_sign_and_value) {
    constexpr int N = 128;
    const std::vector<uint8_t> base = lcg_noise(N, N);
    nvmm::PhaseCorrelator pc(N, N);
    const int shifts[][2] = {{0, 0}, {2, 0}, {0, 3}, {3, -2}, {-4, 5}, {6, -3}};
    for (auto &sh : shifts) {
        const int sx = sh[0], sy = sh[1];
        std::vector<uint8_t> prev((size_t)N * N), curr((size_t)N * N);
        std::vector<float>   prevf((size_t)N * N), currf((size_t)N * N);
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                const uint8_t p = circular_at(base, N, N, x, y);
                const uint8_t c = circular_at(base, N, N, x - sx, y - sy);
                const size_t i = (size_t)y * N + x;
                prev[i] = p; curr[i] = c;
                prevf[i] = (float)p; currf[i] = (float)c;
            }
        const nvmm::GmcShift ncc = nvmm::estimate_shift(prev.data(), curr.data(), N, 24);
        const nvmm::PhaseCorrelator::Shift fft = pc.correlate(prevf.data(), currf.data());

        ASSERT_TRUE(std::fabs(ncc.dx - sx) < 0.5f && std::fabs(ncc.dy - sy) < 0.5f);
        ASSERT_TRUE(std::fabs(fft.x - sx) < 0.15 && std::fabs(fft.y - sy) < 0.15);
        ASSERT_TRUE(std::fabs(ncc.dx - fft.x) < 0.5 && std::fabs(ncc.dy - fft.y) < 0.5);
    }
}

using nvmm::gmc_map_box_to_patch;
using nvmm::gmc_mask_box_to_mean;

/// A diverged Kalman state hands the mapper NaN/Inf; its isfinite guard is what
/// keeps the double->int cast from being UB.
TEST(gmc_mask_non_finite_or_degenerate_box_never_overlaps) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    constexpr double kInf = std::numeric_limits<double>::infinity();
    ASSERT_TRUE(!gmc_map_box_to_patch(kNaN, 0, 10, 10, 1920, 1080, 512, 128).overlaps);
    ASSERT_TRUE(!gmc_map_box_to_patch(0, kInf, 10, 10, 1920, 1080, 512, 128).overlaps);
    ASSERT_TRUE(!gmc_map_box_to_patch(0, 0, kNaN, 10, 1920, 1080, 512, 128).overlaps);
    ASSERT_TRUE(!gmc_map_box_to_patch(0, 0, 10, kInf, 1920, 1080, 512, 128).overlaps);
    ASSERT_TRUE(!gmc_map_box_to_patch(0, 0, 0, 10, 1920, 1080, 512, 128).overlaps);
    ASSERT_TRUE(!gmc_map_box_to_patch(0, 0, 10, -5, 1920, 1080, 512, 128).overlaps);
    ASSERT_TRUE(!gmc_map_box_to_patch(0, 0, 10, 10, 1920, 1080, 0, 128).overlaps);
    ASSERT_TRUE(!gmc_map_box_to_patch(-10000, -10000, 10, 10, 1920, 1080, 1080, 128).overlaps);
}

TEST(gmc_mask_centered_box_maps_to_patch_center) {
    const nvmm::GmcMaskBox mb =
        gmc_map_box_to_patch(960 - 5, 540 - 5, 10, 10, 1920, 1080, 1080, 128, 1.0);
    ASSERT_TRUE(mb.overlaps);
    const int cx = (mb.x0 + mb.x1) / 2, cy = (mb.y0 + mb.y1) / 2;
    ASSERT_TRUE(std::abs(cx - 64) <= 1 && std::abs(cy - 64) <= 1);
}

TEST(gmc_mask_fills_region_with_patch_mean_only) {
    constexpr int N = 16;
    std::vector<uint8_t> patch(N * N);
    for (int i = 0; i < N * N; i++) patch[i] = (uint8_t)((i * 37) % 256);
    uint64_t sum = 0;
    for (int i = 0; i < N * N; i++) sum += patch[i];
    const uint8_t mean = (uint8_t)(sum / (N * N));

    const std::vector<uint8_t> before = patch;
    gmc_mask_box_to_mean(patch.data(), N, 4, 4, 10, 10);
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            const bool inside = x >= 4 && x < 10 && y >= 4 && y < 10;
            ASSERT_EQ(patch[y * N + x], inside ? mean : before[y * N + x]);
        }

    const std::vector<uint8_t> unchanged = patch;
    gmc_mask_box_to_mean(patch.data(), N, 10, 10, 4, 4);
    ASSERT_TRUE(patch == unchanged);
    gmc_mask_box_to_mean(patch.data(), N, 1000, 1000, 2000, 2000);
    ASSERT_TRUE(patch == unchanged);
}

constexpr int kViewW = 1920, kViewH = 1080, kCrop = 512;

struct ViewCase {
    float box_x, box_y, box_w, box_h;
    float want_view_x, want_view_y;
};

/// From the Python samurai.tracker.get_view_around_bbox (view_golden.py).
const ViewCase kViewCases[] = {
    {1015, 446, 18, 12, 768, 196},
    {0,    0,   10, 10, 0,   0},
    {1910, 1070,10, 10, 1408,568},
    {940,  520, 40, 40, 704, 284},
    {5,    500, 10, 20, 0,   254},
    {1900, 500, 15, 20, 1408,254},
    {500,  2,   20, 10, 254, 0},
    {500,  1065,20, 13, 254, 568},
};

TEST(samurai_view_matches_python_golden_and_stays_in_frame) {
    for (const ViewCase &c : kViewCases) {
        nvmm::SamuraiView v = nvmm::get_view_around_bbox(c.box_x, c.box_y, c.box_w, c.box_h,
                                                         kCrop, kViewW, kViewH);
        ASSERT_NEAR(v.x, c.want_view_x, 1e-4);
        ASSERT_NEAR(v.y, c.want_view_y, 1e-4);
        ASSERT_NEAR(v.width, (float)kCrop, 1e-4);
        ASSERT_NEAR(v.height, (float)kCrop, 1e-4);
        ASSERT_TRUE(v.x >= 0.f && v.y >= 0.f);
        ASSERT_TRUE(v.x + v.width <= (float)kViewW);
        ASSERT_TRUE(v.y + v.height <= (float)kViewH);
    }
}

TEST(samurai_view_clamps_oversized_crop) {
    nvmm::SamuraiView v = nvmm::get_view_around_bbox(100, 100, 10, 10, 4096, kViewW, kViewH);
    ASSERT_NEAR(v.width, (float)kViewW, 1e-4);
    ASSERT_NEAR(v.height, (float)kViewH, 1e-4);
}

TEST(vit_grid_counts_tokens_with_truncating_division) {
    using nvmm::vit_grid_side;
    using nvmm::vit_grid_tokens;
    ASSERT_EQ(vit_grid_side(512, 16), 32);
    ASSERT_EQ(vit_grid_tokens(512, 16), 1024);
    ASSERT_EQ(vit_grid_side(384, 16), 24);
    ASSERT_EQ(vit_grid_tokens(384, 16), 576);
    ASSERT_EQ(vit_grid_side(256, 16), 16);
    ASSERT_EQ(vit_grid_tokens(256, 16), 256);
    ASSERT_EQ(vit_grid_tokens(224, 14), 256);
    ASSERT_EQ(vit_grid_tokens(224, 16), 196);
    ASSERT_EQ(vit_grid_side(520, 16), 32);
    ASSERT_EQ(vit_grid_tokens(520, 16), 1024);
    ASSERT_EQ(vit_grid_side(512, 0), 0);
    ASSERT_EQ(vit_grid_tokens(512, 0), 0);
}

}

namespace xfeat_scene {

using namespace nvmm::motion;
using nvmm::xfeat::Pt2;

constexpr double DX = 7.0, DY = -3.0;
const Box kMover{ 285, 135, 40, 40 };

std::vector<MatchPair> background_grid_plus_independent_mover() {
    std::vector<MatchPair> m;
    int idx = 0;
    for (int gy = 0; gy < 20; ++gy)
        for (int gx = 0; gx < 10; ++gx) {
            Pt2 a{ 20.0 + gx * 40, 15.0 + gy * 12 };
            m.push_back({ idx++, a, Pt2{ a.x + DX, a.y + DY } });
        }
    for (int k = 0; k < 12; ++k) {
        Pt2 a{ 290.0 + (k % 4) * 8, 140.0 + (k / 4) * 8 };
        m.push_back({ idx++, a, Pt2{ a.x + DX + 25, a.y + DY - 20 } });
    }
    return m;
}

TEST(xfeat_motion_primitives_recover_planted_scene) {
    const std::vector<MatchPair> m = background_grid_plus_independent_mover();
    Box mover = kMover;

    GmcEstimate g = global_translation_median(m, &mover, 2.0);
    ASSERT_TRUE(g.ok && std::fabs(g.dx - DX) < 1e-6 && std::fabs(g.dy - DY) < 1e-6);
    ASSERT_TRUE(g.inlier_frac > 0.99);

    AffineFit fit = ransac_affine(m, &mover, 300, 1.0, 12, 12345);
    ASSERT_TRUE(fit.ok && fit.inliers >= 190);

    RegionResidual rr = region_max_residual(fit.M, m, mover, 3);
    ASSERT_TRUE(rr.ok && rr.max_resid > 20.0);

    Box bg{ 20, 15, 60, 60 };
    RegionResidual rbg = region_max_residual(fit.M, m, bg, 3);
    ASSERT_TRUE(rbg.ok && rbg.max_resid < 1.0);

    std::vector<MatchPair> mover_static_in_ref_b;
    for (const auto& mp : m) {
        Pt2 b = mover.contains(mp.a) ? Pt2{ mp.a.x + DX, mp.a.y + DY } : mp.b;
        mover_static_in_ref_b.push_back({ mp.idx, mp.a, b });
    }
    AffineFit fitb = ransac_affine(mover_static_in_ref_b, &mover, 300, 1.0, 12, 12345);
    RegionResidual r2 =
        region_max_residual_2ref(fit.M, m, fitb.M, mover_static_in_ref_b, mover, 3);
    ASSERT_TRUE(r2.ok && r2.max_resid < 1.0);

    MotionBlob blob = cluster_moving(m, fit.M, 10.0, 4, 16.0);
    ASSERT_TRUE(blob.ok && blob.n >= 8 && blob.box.x > 280 && blob.box.x < 300);

    ASSERT_TRUE(!global_translation_median({}, nullptr, 2.0).ok);
    ASSERT_TRUE(!region_max_residual(fit.M, {}, mover, 3).ok);
    ASSERT_TRUE(!ransac_affine({}, nullptr).ok);
    ASSERT_TRUE(!cluster_moving({}, fit.M, 10.0).ok);
}

}

int main() {
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
