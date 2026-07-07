/// Plane+parallax independent-motion residual for a moving (panning) camera.
///
/// Frame differencing and single-homography background subtraction both fail on a
/// translating camera: a single dominant-plane homography leaves strong residual
/// wherever the scene has depth (near-field parallax), which then masquerades as
/// motion. This computes a DUAL-homography residual instead:
///
///   1. Match the current frame to a reference frame and fit H1 (RANSAC) — the
///      dominant background plane.
///   2. Re-fit H2 on the RANSAC OUTLIERS — a second plane that captures the
///      parallax structure the first plane missed.
///   3. Compare the reference against the current frame under each plane and keep,
///      per pixel, the residual that survives BOTH planes (element-wise min).
///
/// A pixel that is explained by either plane (flat background OR near-field parallax)
/// is suppressed; only genuinely independent motion — an object moving relative to
/// the whole scene — keeps a high residual. Two reference frames (e.g. two different
/// past frames) are combined the same way so a static edge that aligns under one
/// reference but not the other is also rejected.
///
/// The OpenCV op chain per plane (warpPerspective, absdiff, a second ones-warp,
/// compare, erode, masked copy — ~6 sweeps, two of them full warps) is fused into
/// ONE inverse-warp pass per reference pair: each output pixel projects through H,
/// samples the reference bilinearly, differences, and applies validity as a
/// source-margin test — no warped frame, no mask buffer, no erode. Feature
/// matching is selectable (small_motion / orb — see detail/features.hpp); current-
/// frame features are extracted once and reused for both references.
///
/// Pure C++14, header-only, no dependencies — unit-testable on the host.
/// (Distinct from common/nvmm_motion.hpp, which scores boxes from a precomputed
///  optical-flow field; this works directly on the raw grayscale frames.)
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "detail/features.hpp"
#include "detail/homography.hpp"
#include "image.hpp"
#include "image_ops.hpp"

namespace nvmm {
namespace motion {

enum class FeaturePipeline {
    small_motion,  // FAST + bounded ZNCC search — near-consecutive frames, small pan
    orb            // pyramid ORB + Hamming KNN — rotation/scale-robust, heavier
};

struct DualHomographyParams {
    int    features = 4000;      // ORB keypoint budget (small_motion uses max_corners)
    double ransac_thresh = 3.0;  // RANSAC reprojection threshold (px)
    int    min_matches = 20;     // need at least this many good matches to fit a plane
    int    valid_margin = 15;    // drop pixels whose reference sample is this close to
                                 // the reference edge (replaces the warp-mask erode)
    int    blur = 3;             // Gaussian blur applied to the residual (odd; 0 = none)
    int    border = 12;          // zero out this many px around the frame border
    float  ratio = 0.75f;        // Lowe ratio for the ORB match
    FeaturePipeline pipeline = FeaturePipeline::small_motion;
    int    fast_thresh = 20;     // FAST threshold (both pipelines)
    int    max_corners = 800;    // small_motion corner budget
    int    search_radius = 32;   // small_motion max displacement (px)
    float  zncc_min = 0.6f;      // small_motion match acceptance
    float  min_plane_extent = 0.15f;  // the central 60% of H2's consensus must span at
                                      // least this fraction of the frame in x or y —
                                      // real parallax structure is extended; a compact
                                      // cluster (+ a few chance agreers) is usually the
                                      // independent mover, which H2 must not absorb
};

namespace detail {

struct Plane {
    Mat3 H;           // cur -> ref
    bool ok = false;
};

/// H1 = dominant plane of the correspondences, H2 = parallax plane (RANSAC outliers).
/// `fw`/`fh` are the frame dimensions (for the H2 extent gate).
inline void two_planes(const std::vector<Pt> &p1, const std::vector<Pt> &p2,
                       const DualHomographyParams &p, int fw, int fh,
                       Plane &pl1, Plane &pl2)
{
    pl1.ok = pl2.ok = false;
    if ((int)p1.size() < p.min_matches) return;
    std::vector<uint8_t> mask;
    pl1.ok = find_homography_ransac(p1, p2, p.ransac_thresh, pl1.H, mask);
    if (!pl1.ok) return;
    std::vector<Pt> o1, o2;
    for (size_t i = 0; i < p1.size(); i++)
        if (!mask[i]) { o1.push_back(p1[i]); o2.push_back(p2[i]); }
    if ((int)o1.size() >= p.min_matches) {
        std::vector<uint8_t> omask;
        pl2.ok = find_homography_ransac(o1, o2, p.ransac_thresh, pl2.H, omask);
        // A real parallax plane explains a BROAD, EXTENDED consensus. A compact
        // coherent cluster of outliers is typically the independent mover itself —
        // if H2 locked onto that, the min-combine would veto exactly the motion
        // this residual exists to expose. Demand min_matches consensus AND a
        // TRIMMED consensus span (central 60% per axis) covering a real fraction
        // of the frame: a bounding box can be faked by the mover cluster plus a
        // couple of chance agreers, a trimmed span cannot.
        if (pl2.ok) {
            std::vector<float> xs, ys;
            for (size_t i = 0; i < o1.size(); i++)
                if (omask[i]) { xs.push_back(o1[i].x); ys.push_back(o1[i].y); }
            bool keep = (int)xs.size() >= p.min_matches;
            if (keep) {
                std::sort(xs.begin(), xs.end());
                std::sort(ys.begin(), ys.end());
                const size_t i0 = (xs.size() - 1) / 5, i1 = (xs.size() - 1) * 4 / 5;
                keep = xs[i1] - xs[i0] >= p.min_plane_extent * (float)fw ||
                       ys[i1] - ys[i0] >= p.min_plane_extent * (float)fh;
            }
            pl2.ok = keep;
        }
    }
}

/// One fused pass: residual of `cur` against ref_a via Ha min ref_b via Hb.
/// Per output pixel: project through H (incremental along the row), sample the
/// reference bilinearly, |diff|, valid only when the SOURCE point keeps
/// `margin` px from the reference edge; invalid or missing-plane samples
/// contribute 0 exactly like the warp-mask erode + masked-copy chain did.
/// False when neither plane is available.
inline bool plane_pair_residual(img::View<const uint8_t> cur,
                                img::View<const uint8_t> ref_a, const Plane &pa,
                                img::View<const uint8_t> ref_b, const Plane &pb,
                                int margin, img::Image<float> &out)
{
    if (!pa.ok && !pb.ok) return false;
    const int w = cur.width, h = cur.height;
    if (out.width() != w || out.height() != h) out = img::Image<float>(w, h);

    // >= 1 so the bilinear taps at ix+1 / iy+1 stay in bounds
    const int mg = std::max(1, margin);
    const double lo = mg, hix = (double)w - 1 - mg, hiy = (double)h - 1 - mg;
    auto sample = [&](img::View<const uint8_t> ref, const Mat3 &H, int x, int y,
                      double &nx, double &ny, double &dn, bool init, float cv) -> float {
        if (init) {
            nx = H.m[0] * x + H.m[1] * y + H.m[2];
            ny = H.m[3] * x + H.m[4] * y + H.m[5];
            dn = H.m[6] * x + H.m[7] * y + H.m[8];
        }
        float r = 0.f;
        if (std::abs(dn) > 1e-12) {
            const double sx = nx / dn, sy = ny / dn;
            if (sx >= lo && sx <= hix && sy >= lo && sy <= hiy) {
                const int ix = (int)sx, iy = (int)sy;
                const float fx = (float)(sx - ix), fy = (float)(sy - iy);
                const uint8_t *r0 = ref.row(iy), *r1 = ref.row(iy + 1);
                const float v = (1.f - fy) * ((1.f - fx) * r0[ix] + fx * r0[ix + 1]) +
                                fy * ((1.f - fx) * r1[ix] + fx * r1[ix + 1]);
                r = std::abs(cv - v);
            }
        }
        nx += H.m[0]; ny += H.m[3]; dn += H.m[6];
        return r;
    };

    for (int y = 0; y < h; y++) {
        const uint8_t *c = cur.row(y);
        float *o = out.row(y);
        double nxa = 0, nya = 0, dna = 0, nxb = 0, nyb = 0, dnb = 0;
        for (int x = 0; x < w; x++) {
            const float cv = (float)c[x];
            if (pa.ok && pb.ok) {
                const float va = sample(ref_a, pa.H, x, y, nxa, nya, dna, x == 0, cv);
                const float vb = sample(ref_b, pb.H, x, y, nxb, nyb, dnb, x == 0, cv);
                o[x] = std::min(va, vb);
            } else if (pa.ok) {
                o[x] = sample(ref_a, pa.H, x, y, nxa, nya, dna, x == 0, cv);
            } else {
                o[x] = sample(ref_b, pb.H, x, y, nxb, nyb, dnb, x == 0, cv);
            }
        }
    }
    return true;
}

/// Correspondences cur -> ref for the configured pipeline. `cur_corners` /
/// `cur_orb` carry the current frame's features, extracted once by the caller.
inline void pipeline_matches(img::View<const uint8_t> cur, img::View<const uint8_t> ref,
                             const DualHomographyParams &p,
                             const std::vector<Corner> &cur_corners,
                             const std::vector<OrbFeature> &cur_orb,
                             std::vector<Pt> &p1, std::vector<Pt> &p2)
{
    p1.clear(); p2.clear();
    if (p.pipeline == FeaturePipeline::small_motion) {
        SmallMotionParams sp;
        sp.fast_thresh = p.fast_thresh;
        sp.max_corners = p.max_corners;
        sp.search_radius = p.search_radius;
        sp.zncc_min = p.zncc_min;
        small_motion_matches(cur, ref, cur_corners, sp, p1, p2);
    } else {
        OrbParams op;
        op.nfeatures = p.features;
        op.fast_thresh = p.fast_thresh;
        op.ratio = p.ratio;
        std::vector<OrbFeature> ref_orb = orb_detect(ref, op);
        orb_match(cur_orb, ref_orb, p.ratio, p1, p2);
    }
}

}  // namespace detail

/// Independent-motion residual of `cur` against two reference frames.
/// `cur`, `ref_a`, `ref_b` are single-channel u8 (grayscale), same size.
/// Returns float (same size): high where motion is unexplained by either the
/// dominant plane or the parallax plane. Empty if no homography could be fit
/// (e.g. textureless input) — callers should treat empty as "no estimate".
inline img::Image<float> independent_motion_residual(img::View<const uint8_t> cur,
                                                     img::View<const uint8_t> ref_a,
                                                     img::View<const uint8_t> ref_b,
                                                     const DualHomographyParams &p = {})
{
    // current-frame features once, reused against both references
    std::vector<detail::Corner> cur_corners;
    std::vector<detail::OrbFeature> cur_orb;
    if (p.pipeline == FeaturePipeline::small_motion) {
        // corners must keep patch room for the ZNCC window
        cur_corners = detail::fast_corners(cur, p.fast_thresh, p.max_corners, 8);
    } else {
        detail::OrbParams op;
        op.nfeatures = p.features;
        op.fast_thresh = p.fast_thresh;
        op.ratio = p.ratio;
        cur_orb = detail::orb_detect(cur, op);
    }

    std::vector<detail::Pt> p1, p2;
    detail::Plane ha1, ha2, hb1, hb2;
    detail::pipeline_matches(cur, ref_a, p, cur_corners, cur_orb, p1, p2);
    detail::two_planes(p1, p2, p, cur.width, cur.height, ha1, ha2);
    detail::pipeline_matches(cur, ref_b, p, cur_corners, cur_orb, p1, p2);
    detail::two_planes(p1, p2, p, cur.width, cur.height, hb1, hb2);

    img::Image<float> r1, r2;
    if (!detail::plane_pair_residual(cur, ref_a, ha1, ref_b, hb1, p.valid_margin, r1))
        return img::Image<float>();
    const bool have_r2 =
        detail::plane_pair_residual(cur, ref_a, ha2, ref_b, hb2, p.valid_margin, r2);

    if (p.blur > 0) {
        img::Image<float> tmp;
        img::gaussian_blur<float>(r1.view(), tmp, r1.view(), p.blur);
        if (have_r2) {
            // fuse r2's vertical blur pass with the cross-plane min into r1
            const std::vector<float> k = img::gaussian_kernel(p.blur);
            img::convolve_rows<float>(r2.view(), tmp.view(), k);
            img::convolve_cols(tmp.view(), k, [&](int y, const float *row, int w) {
                float *d = r1.row(y);
                for (int x = 0; x < w; x++) d[x] = std::min(d[x], row[x]);
            });
        }
    } else if (have_r2) {
        for (int y = 0; y < r1.height(); y++) {
            float *d = r1.row(y);
            const float *s = r2.row(y);
            for (int x = 0; x < r1.width(); x++) d[x] = std::min(d[x], s[x]);
        }
    }
    img::zero_border(r1.view(), p.border);
    return r1;
}

}  // namespace motion
}  // namespace nvmm
