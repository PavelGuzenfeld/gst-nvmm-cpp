// Pure-host test for gst/common/xfeat_motion.hpp — the OpenCV-free sparse
// independent-motion analytics (GMC median, RANSAC background affine, in-region
// residual, motion-blob clustering). No TRT/CUDA/OpenCV, runs on the x86 CI build.
//
// Synthetic scene: a background grid moved by a known camera translation, plus a
// planted cluster that moves independently. Verifies each primitive recovers the
// ground truth and that scarcity resolves to the safe "no verdict" path.
#include "xfeat_motion.hpp"
#include <cstdio>
#include <cassert>
#include <cmath>

using namespace nvmm::motion;
using nvmm::xfeat::Pt2;

int main() {
    const double DX = 7.0, DY = -3.0;
    std::vector<MatchPair> m;
    int idx = 0;
    for (int gy = 0; gy < 20; ++gy)
        for (int gx = 0; gx < 10; ++gx) {
            Pt2 a{ 20.0 + gx * 40, 15.0 + gy * 12 };
            Pt2 b{ a.x + DX, a.y + DY };            // background: moves with camera
            m.push_back({ idx++, a, b });
        }
    Box mover{ 285, 135, 40, 40 };
    for (int k = 0; k < 12; ++k) {
        Pt2 a{ 290.0 + (k % 4) * 8, 140.0 + (k / 4) * 8 };
        Pt2 b{ a.x + DX + 25, a.y + DY - 20 };      // independent motion on top of camera
        m.push_back({ idx++, a, b });
    }

    // 1) GMC median (mover excluded) recovers the camera translation exactly.
    GmcEstimate g = global_translation_median(m, &mover, 2.0);
    assert(g.ok && std::fabs(g.dx - DX) < 1e-6 && std::fabs(g.dy - DY) < 1e-6);
    assert(g.inlier_frac > 0.99);

    // 2) RANSAC background affine: all background matches are inliers.
    AffineFit fit = ransac_affine(m, &mover, 300, 1.0, 12, 12345);
    assert(fit.ok && fit.inliers >= 190);

    // 3) The mover region shows high residual under the background model.
    RegionResidual rr = region_max_residual(fit.M, m, mover, 3);
    assert(rr.ok && rr.max_resid > 20.0);

    // 4) A static background box shows ~zero residual.
    Box bg{ 20, 15, 60, 60 };
    RegionResidual rbg = region_max_residual(fit.M, m, bg, 3);
    assert(rbg.ok && rbg.max_resid < 1.0);

    // 5) Two-reference min-combine: build a second ref where the mover is static so
    //    its per-keypoint MIN residual collapses to ~0 (rejected — must move vs BOTH).
    std::vector<MatchPair> mb;
    for (const auto& mp : m) {
        Pt2 b = mover.contains(mp.a) ? Pt2{ mp.a.x + DX, mp.a.y + DY } : mp.b;  // mover static here
        mb.push_back({ mp.idx, mp.a, b });
    }
    AffineFit fitb = ransac_affine(mb, &mover, 300, 1.0, 12, 12345);
    RegionResidual r2 = region_max_residual_2ref(fit.M, m, fitb.M, mb, mover, 3);
    assert(r2.ok && r2.max_resid < 1.0);            // static under ref B => combined min ~0

    // 6) Motion blob clusters the movers into a plausible bbox.
    MotionBlob blob = cluster_moving(m, fit.M, 10.0, 4, 16.0);
    assert(blob.ok && blob.n >= 8 && blob.box.x > 280 && blob.box.x < 300);

    // 7) "No verdict" parity: empty / too-few inputs resolve to not-ok (safe default).
    assert(!global_translation_median({}, nullptr, 2.0).ok);
    assert(!region_max_residual(fit.M, {}, mover, 3).ok);
    assert(!ransac_affine({}, nullptr).ok);
    assert(!cluster_moving({}, fit.M, 10.0).ok);

    printf("xfeat_motion: all assertions passed\n");
    return 0;
}
