/// xfeat_motion.hpp — sparse independent-motion analytics over XFeat matches.
///
/// The OpenCV-free replacement for common/dual_homography.hpp. Where the OpenCV
/// path built a DENSE per-pixel residual map (ORB + dual-homography + warp), this
/// works on SPARSE matched keypoint pairs from the XFeat/LightGlue matcher:
///
///   - GMC: dominant camera translation = MEDIAN of match displacements, with the
///     tracked box EXCLUDED (so the target's own motion never biases the camera
///     estimate). Analog of the phaseCorrelate background-dominated peak.
///   - Independent motion: fit a robust background transform (RANSAC affine) from
///     matches OUTSIDE a region, then measure the reprojection residual of matches
///     INSIDE it. Low residual => the region moves with the scene (static); high =>
///     it moves independently. Analog of the dual-homography residual + minMaxLoc.
///   - Two references (cur vs -dlt and cur vs -2*dlt) are combined by per-keypoint
///     MIN residual — a static edge that aligns under one reference but not the
///     other is rejected. This mirrors the OpenCV cv::min over two warped residuals.
///   - Motion blob: cluster the independently-moving keypoints (no connected-
///     components / NPP CCL needed) into a bbox for detector-independent seeding.
///
/// IMPORTANT: residuals here are GEOMETRIC pixel displacement (in registration
/// space, e.g. 480x270), NOT intensity absdiff. The OpenCV thresholds (val_rmin=12,
/// rmin=12, ...) do NOT transfer — every gate must be re-derived empirically.
///
/// Pure host, std-only, OpenCV-free. Header-only, unit-testable off-device.
#pragma once
#include "xfeat_register.hpp"   // nvmm::xfeat::Pt2, affine_from_3pts
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace nvmm { namespace motion {

using nvmm::xfeat::Pt2;

/// A matched keypoint pair. `a` is the point in the anchor frame (current), `b` in
/// the other frame (a past reference). `idx` is the anchor-frame keypoint index, so
/// the same keypoint can be tied across two reference match-sets (two-ref combine).
struct MatchPair { int idx; Pt2 a; Pt2 b; };

/// Axis-aligned box in the same coordinate space as the match points (registration
/// space). Top-left + dims.
struct Box {
    double x = 0, y = 0, w = 0, h = 0;
    bool contains(const Pt2& p) const {
        return p.x >= x && p.x <= x + w && p.y >= y && p.y <= y + h;
    }
};

// ---- small helpers ----------------------------------------------------------

inline Pt2 apply_affine(const double M[6], const Pt2& p) {
    return { M[0]*p.x + M[1]*p.y + M[2], M[3]*p.x + M[4]*p.y + M[5] };
}

/// reprojection residual of a match under a candidate transform M (anchor->ref).
inline double residual(const double M[6], const MatchPair& mp) {
    Pt2 q = apply_affine(M, mp.a);
    double dx = q.x - mp.b.x, dy = q.y - mp.b.y;
    return std::sqrt(dx*dx + dy*dy);
}

// ---- GMC: median global translation (box excluded) --------------------------

struct GmcEstimate {
    double dx = 0, dy = 0;   // dominant camera translation (anchor->ref units)
    double inlier_frac = 0;  // fraction of used matches within `tol` of the median
    int    n = 0;            // number of matches used (outside the excluded box)
    bool   ok = false;       // n >= min_n
};

/// Dominant translation = componentwise median of match displacements, ignoring
/// matches whose anchor point falls in `exclude` (the tracked target). `tol` is the
/// displacement agreement radius used only to report an inlier fraction / confidence.
inline GmcEstimate global_translation_median(const std::vector<MatchPair>& m,
                                             const Box* exclude = nullptr,
                                             double tol = 2.0, int min_n = 8) {
    std::vector<double> dxs, dys;
    dxs.reserve(m.size()); dys.reserve(m.size());
    for (const auto& mp : m) {
        if (exclude && exclude->contains(mp.a)) continue;
        dxs.push_back(mp.b.x - mp.a.x);
        dys.push_back(mp.b.y - mp.a.y);
    }
    GmcEstimate e;
    e.n = (int)dxs.size();
    if (e.n < min_n) return e;
    auto median = [](std::vector<double>& v) {
        std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
        double m1 = v[v.size()/2];
        if (v.size() % 2) return m1;
        double m0 = *std::max_element(v.begin(), v.begin() + v.size()/2);
        return 0.5 * (m0 + m1);
    };
    e.dx = median(dxs);
    e.dy = median(dys);
    int inl = 0;
    for (const auto& mp : m) {
        if (exclude && exclude->contains(mp.a)) continue;
        double ex = (mp.b.x - mp.a.x) - e.dx, ey = (mp.b.y - mp.a.y) - e.dy;
        if (std::sqrt(ex*ex + ey*ey) <= tol) inl++;
    }
    e.inlier_frac = (double)inl / (double)e.n;
    e.ok = true;
    return e;
}

// ---- RANSAC affine (background model, region excluded) ----------------------

struct AffineFit {
    double M[6] = {1,0,0, 0,1,0};  // identity default
    int    inliers = 0;
    int    n = 0;                  // candidate matches (outside excluded box)
    bool   ok = false;
};

/// deterministic xorshift PRNG (no <random> global state; reproducible for tests).
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed = 0x9e3779b9u) : s(seed ? seed : 0x9e3779b9u) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    int below(int n) { return n <= 0 ? 0 : (int)(next() % (uint32_t)n); }
};

/// Robust background transform fit from matches OUTSIDE `exclude`, mirroring
/// findHomography(..., RANSAC) on the background. Affine (6-DOF): sparse matches
/// rarely justify an 8-DOF homography, and camera motion on a distant background is
/// well-approximated by affine. `tol` = inlier reprojection threshold (px).
inline AffineFit ransac_affine(const std::vector<MatchPair>& m,
                               const Box* exclude = nullptr,
                               int iters = 200, double tol = 2.0,
                               int min_inliers = 12, uint32_t seed = 0x9e3779b9u) {
    AffineFit fit;
    std::vector<int> cand;
    cand.reserve(m.size());
    for (int i = 0; i < (int)m.size(); ++i)
        if (!(exclude && exclude->contains(m[i].a))) cand.push_back(i);
    fit.n = (int)cand.size();
    if (fit.n < 3) return fit;

    Rng rng(seed);
    int best_inl = -1;
    double bestM[6];
    for (int it = 0; it < iters; ++it) {
        // sample 3 distinct candidates
        int i0 = cand[rng.below(fit.n)];
        int i1 = cand[rng.below(fit.n)];
        int i2 = cand[rng.below(fit.n)];
        if (i1 == i0 || i2 == i0 || i2 == i1) continue;
        std::array<Pt2,3> src{ m[i0].a, m[i1].a, m[i2].a };
        std::array<Pt2,3> dst{ m[i0].b, m[i1].b, m[i2].b };
        double M[6];
        if (!nvmm::xfeat::affine_from_3pts(src, dst, M)) continue;
        int inl = 0;
        for (int c : cand) if (residual(M, m[c]) <= tol) inl++;
        if (inl > best_inl) { best_inl = inl; std::copy(M, M+6, bestM); }
    }
    if (best_inl >= min_inliers) {
        std::copy(bestM, bestM+6, fit.M);
        fit.inliers = best_inl;
        fit.ok = true;
    }
    return fit;
}

// ---- region residual (independent-motion test) ------------------------------

struct RegionResidual {
    double max_resid = 0;  // largest reprojection residual among in-box matches
    int    n = 0;          // number of in-box matches contributing
    bool   ok = false;     // n >= min_in_box (else "no verdict")
};

/// Max reprojection residual of matches whose anchor point lies in `box`, under the
/// background transform `M`. Analog of cv::minMaxLoc max over the rad-4 ROI. High =>
/// the region moves independently of the background.
inline RegionResidual region_max_residual(const double M[6],
                                          const std::vector<MatchPair>& m,
                                          const Box& box, int min_in_box = 3) {
    RegionResidual r;
    for (const auto& mp : m) {
        if (!box.contains(mp.a)) continue;
        r.max_resid = std::max(r.max_resid, residual(M, mp));
        r.n++;
    }
    r.ok = r.n >= min_in_box;
    return r;
}

/// Two-reference combine: for each ANCHOR keypoint present in BOTH match-sets, take
/// MIN of its residual under the two background transforms; then report the MAX over
/// in-box keypoints of that per-keypoint min. Mirrors OpenCV cv::min over the two
/// warped residuals then minMaxLoc. A keypoint present in only one set is skipped
/// (it did not survive both references — cannot corroborate independent motion).
inline RegionResidual region_max_residual_2ref(const double Ma[6],
                                               const std::vector<MatchPair>& ma,
                                               const double Mb[6],
                                               const std::vector<MatchPair>& mb,
                                               const Box& box, int min_in_box = 3) {
    // residual-by-anchor-index for the second set
    std::vector<std::pair<int,double>> rb;
    rb.reserve(mb.size());
    for (const auto& mp : mb) if (box.contains(mp.a)) rb.push_back({mp.idx, residual(Mb, mp)});
    std::sort(rb.begin(), rb.end());
    auto find_b = [&](int idx) -> double {
        auto lo = std::lower_bound(rb.begin(), rb.end(), std::make_pair(idx, -1e300));
        if (lo != rb.end() && lo->first == idx) return lo->second;
        return -1.0;  // absent
    };
    RegionResidual r;
    for (const auto& mp : ma) {
        if (!box.contains(mp.a)) continue;
        double rbv = find_b(mp.idx);
        if (rbv < 0) continue;                       // must survive both references
        double comb = std::min(residual(Ma, mp), rbv);
        r.max_resid = std::max(r.max_resid, comb);
        r.n++;
    }
    r.ok = r.n >= min_in_box;
    return r;
}

// ---- combined two-reference residual list (for the drone gate) --------------

struct ResidPt { Pt2 pt; double resid; };  // anchor point + combined 2-ref residual

/// Per-anchor combined residual over the WHOLE frame (not a single box): for each
/// anchor keypoint present in BOTH match-sets, the MIN of its residual under the two
/// background transforms. `pt` is the anchor coordinate (registration space). The
/// drone gate samples this list near det centers and clusters it for motion blobs.
inline std::vector<ResidPt> combined_residuals_2ref(const double Ma[6],
                                                    const std::vector<MatchPair>& ma,
                                                    const double Mb[6],
                                                    const std::vector<MatchPair>& mb) {
    std::vector<std::pair<int,double>> rb;
    rb.reserve(mb.size());
    for (const auto& mp : mb) rb.push_back({mp.idx, residual(Mb, mp)});
    std::sort(rb.begin(), rb.end());
    std::vector<ResidPt> out;
    out.reserve(ma.size());
    for (const auto& mp : ma) {
        auto lo = std::lower_bound(rb.begin(), rb.end(), std::make_pair(mp.idx, -1e300));
        if (lo == rb.end() || lo->first != mp.idx) continue;   // must survive both refs
        out.push_back({ mp.a, std::min(residual(Ma, mp), lo->second) });
    }
    return out;
}

// ---- motion blob (cluster of independently-moving keypoints) ----------------

struct MotionBlob {
    Box  box;         // bbox of the largest independently-moving cluster (anchor coords)
    int  n = 0;       // keypoints in the cluster
    bool ok = false;
};

/// Cluster the ANCHOR points whose residual under `M` exceeds `resid_thresh` into
/// grid cells of side `cell`, union-find over 8-connected occupied cells, and return
/// the bbox of the largest cluster if it has >= `min_pts`. Replaces
/// connectedComponentsWithStats (NPP has no CCL) on sparse points directly.
inline MotionBlob cluster_moving(const std::vector<MatchPair>& m, const double M[6],
                                 double resid_thresh, int min_pts = 4, double cell = 16.0) {
    MotionBlob out;
    // moving points (anchor coords) + their cell coords
    std::vector<Pt2> pts;
    for (const auto& mp : m)
        if (residual(M, mp) >= resid_thresh) pts.push_back(mp.a);
    const int n = (int)pts.size();
    if (n < min_pts) return out;

    // union-find over points; connect points whose cells are within 1 (8-neighbourhood)
    std::vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;
    auto find = [&](int i){ while (parent[i]!=i){ parent[i]=parent[parent[i]]; i=parent[i]; } return i; };
    auto unite = [&](int a, int b){ a=find(a); b=find(b); if (a!=b) parent[a]=b; };
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            if (std::fabs(pts[i].x - pts[j].x) <= cell && std::fabs(pts[i].y - pts[j].y) <= cell)
                unite(i, j);
        }
    // largest component
    std::vector<int> cnt(n, 0);
    for (int i = 0; i < n; ++i) cnt[find(i)]++;
    int root = -1, best = 0;
    for (int i = 0; i < n; ++i) if (cnt[i] > best) { best = cnt[i]; root = i; }
    if (root < 0 || best < min_pts) return out;

    double x0=1e300, y0=1e300, x1=-1e300, y1=-1e300;
    for (int i = 0; i < n; ++i) if (find(i) == root) {
        x0 = std::min(x0, pts[i].x); y0 = std::min(y0, pts[i].y);
        x1 = std::max(x1, pts[i].x); y1 = std::max(y1, pts[i].y);
    }
    out.box = { x0, y0, x1 - x0, y1 - y0 };
    out.n = best;
    out.ok = true;
    return out;
}

}} // namespace nvmm::motion
